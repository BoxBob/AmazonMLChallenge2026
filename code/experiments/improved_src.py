# %% [markdown]
# # Business Entity Resolution — Improved pipeline (v2, GPU)
#
# Fixes for the problems found in the baseline error analysis (baseline validation macro F0.5 = 0.8871):
#
# | Problem | Fix in this notebook |
# |---|---|
# | Blocking recall 83.6% (64% of the loss) | 3 GPU kNN channels (name, address, **combined name+address**), larger k, 6 exact keys (incl. first-token + house number, Indian compound numbers `20-158/1/A`), then a GPU **prefilter** model keeping the top-M candidates per S1 |
# | Native-script (Tamil/Devanagari/Telugu/...) names & addresses | **Transliteration lexicon learned from the training ground truth** (`praivet→private`, `bildrs→builders`, ...), applied word by word before normalization |
# | India much worse than US, blank candidate addresses, trade/DBA names | IDF-weighted token overlap, *distinctive-mismatch* features (rare word present on one side only), address-crowding and name-frequency counts, numeric-token overlap, per-country thresholds |
# | Same address, different business (main false-positive source) | rare-token mismatch features + address crowd count |
# | Single-match entities / singletons | second-stage **group-context model** (stacked on out-of-fold stage-1 probabilities: rank within entity, gap to best, #strong candidates, best of other source) |
# | Only 14% of training entities used | 400k training entities (+33%), validation on the *same* 100k entities as the baseline for a fair comparison |
# | Normalization gaps | leading-zero house numbers, plural stripping, more French address/legal forms |
#
# Outputs go to **different files**: `output/candidate_pairs_v2.tsv`, `output/matching_results_v2.tsv`.
# All caches and temp files are on the D: drive (`work/v2`).

# %%
import os
import tempfile
from pathlib import Path

# project root = folder holding dataset/, utils/ and output/; override with the ER_ROOT env var
ROOT = Path(os.environ.get("ER_ROOT", r"D:\AmazonMLChallenge2026"))
DATA = ROOT / "dataset"
WORK = ROOT / "work" / "v2"
TMP = ROOT / "work" / "tmp"
OUT = ROOT / "output"
for p in (WORK, TMP, OUT):
    p.mkdir(parents=True, exist_ok=True)
OUT_CAND = OUT / "candidate_pairs_v2.tsv"
OUT_MATCH = OUT / "matching_results_v2.tsv"

# keep every temp file (joblib memmaps, tempfile, ...) off the C: drive
os.environ["TEMP"] = os.environ["TMP"] = os.environ["JOBLIB_TEMP_FOLDER"] = str(TMP)
tempfile.tempdir = str(TMP)

import csv
import gc
import math
import re
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.parquet as pq
import torch
import xgboost as xgb
from anyascii import anyascii
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

assert torch.cuda.is_available(), "CUDA GPU not visible to torch"
DEVICE = "cuda"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0),
      "| xgboost", xgb.__version__, "| pandas", pd.__version__)

# %% [markdown]
# ## Configuration

# %%
SEED = 42
N_TRAIN_S1 = 400_000        # S1 entities used for training (baseline: 300k)
N_VAL_S1 = 100_000          # same validation entities as the baseline notebook
EMB_DIM = 192
SVD_SAMPLE = 250_000
TFIDF_MAX_FEATURES = 2**18
K = {"core": 15, "addr": 20, "comb": 30}   # kNN neighbours per (S1, source) per channel
EXACT_BLOCK_CAP = 50
PF_TRAIN_S1 = 120_000       # training entities used to fit the prefilter
PF_EVAL_S1 = 30_000         # validation entities used to choose the prefilter cut
PF_RETENTION = 0.995        # keep enough candidates to retain this share of blocked true pairs
PF_M_GRID = list(range(20, 85, 5))
MATCH_CAP = {2: 5, 3: 6}
N_JOBS = 6
QUERY_BATCH = 512
POOL_CHUNK = 300_000
STAGE_CHUNK = 50_000        # S1 queries per candidate/feature chunk
LEX_MIN_COUNT = 3
LEX_MIN_SHARE = 0.5
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def read_source(path, usecols=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, engine="c", usecols=usecols)

# %% [markdown]
# ## 1. Train / validation split of Source-1 entities
# Identical permutation to the baseline notebook: validation = the same 100k entities, training = the
# baseline's 300k + 100k more.

# %%
n_s1_train = len(read_source(DATA / "train" / "train_source1.tsv", usecols=["entity_id"]))
_perm = np.random.default_rng(SEED).permutation(np.flatnonzero(np.ones(n_s1_train, dtype=bool)))
VAL_Q = np.sort(_perm[300_000:300_000 + N_VAL_S1])
TRAIN_Q = np.sort(np.concatenate([_perm[:300_000], _perm[400_000:400_000 + N_TRAIN_S1 - 300_000]]))
assert len(np.intersect1d(VAL_Q, TRAIN_Q)) == 0
log(f"train S1 {len(TRAIN_Q):,} | val S1 {len(VAL_Q):,}")

# %% [markdown]
# ## 2. Transliteration lexicon learned from the training ground truth
# Native-script words in S2/S3 are machine transliterations of the English name. For every true
# (S1, S2/S3) pair whose S2/S3 side contains non-ASCII words we align words and count
# `anyascii(native word) → English token`. Names are aligned by position (same word count); address words by
# co-occurrence with S1 address tokens (with a lift filter against generic words). Validation entities are
# excluded so the validation score stays honest.

# %%
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def _fold(s):
    if not s.isascii():
        s = anyascii(s)
    return s.lower()


def _word_key(w):
    return _NON_ALNUM.sub("", anyascii(w).lower())


def _latin_tokens(s):
    return _NON_ALNUM.sub(" ", _fold(s).replace("&", " and ").replace(".", "").replace("'", "")).split()


def _units(s):
    """word units of a raw string: (token, is_native)"""
    out = []
    for w in s.split():
        if w.isascii():
            out += [(t, False) for t in _latin_tokens(w)]
        else:
            k = _word_key(w)
            if k:
                out.append((k, True))
    return out


def build_lexicons():
    cache = WORK / "lexicons.parquet"
    if cache.exists():
        t = pd.read_parquet(cache)
        return ({r.k: r.v for r in t[t.kind == "name"].itertuples()},
                {r.k: r.v for r in t[t.kind == "addr"].itertuples()})
    gt = read_source(DATA / "train" / "train_ground_truth.tsv")
    s1 = read_source(DATA / "train" / "train_source1.tsv")
    s1 = s1[~s1.index.isin(VAL_Q)]                      # exclude validation entities
    gt = gt[gt.source1_entity_id.isin(s1.entity_id)]
    m = gt.assign(c=gt.matched_entity_ids.str.split(",")).explode("c")
    m = m[m.c != ""][["source1_entity_id", "c"]]
    other = pd.concat([read_source(DATA / "train" / f"train_source{s}.tsv") for s in (2, 3)])
    other = other[~(pac.string_is_ascii(pa.array(other.business_name.tolist())).to_numpy(zero_copy_only=False)
                    & pac.string_is_ascii(pa.array(other.business_address.tolist())).to_numpy(zero_copy_only=False))]
    m = m.merge(other.rename(columns={"entity_id": "c"}), on="c")
    m = m.merge(s1.rename(columns={"entity_id": "source1_entity_id", "business_name": "n1",
                                   "business_address": "a1"})[["source1_entity_id", "n1", "a1"]],
                on="source1_entity_id")
    log(f"lexicon: {len(m):,} true pairs with non-ASCII S2/S3 text")

    name_pair, name_tot = Counter(), Counter()
    addr_pair, addr_tot, addr_base = Counter(), Counter(), Counter()
    n_addr = 0
    for n1, a1, n2, a2 in zip(m.n1, m.a1, m.business_name, m.business_address):
        if not n2.isascii():
            A, B = _latin_tokens(n1), _units(n2)
            if len(A) == len(B):
                for (u, nat), a in zip(B, A):
                    if nat:
                        name_pair[(u, a)] += 1
                        name_tot[u] += 1
        if a2 and not a2.isascii():
            A = set(_latin_tokens(a1))
            B = _units(a2)
            latin_c = {u for u, nat in B if not nat}
            cand = A - latin_c
            n_addr += 1
            addr_base.update(cand)
            for u in {u for u, nat in B if nat}:
                addr_tot[u] += 1
                for a in cand:
                    addr_pair[(u, a)] += 1

    def choose(pairs, tot, base=None, n=None):
        best = {}
        for (u, a), c in pairs.items():
            if c < LEX_MIN_COUNT or c / tot[u] < LEX_MIN_SHARE:
                continue
            if base is not None and c / tot[u] < 3 * base[a] / n:   # lift filter
                continue
            if u not in best or c > best[u][1]:
                best[u] = (a, c)
        return {u: a for u, (a, c) in best.items() if u != a}

    lex_n = choose(name_pair, name_tot)
    lex_a = choose(addr_pair, addr_tot, addr_base, n_addr)
    pd.concat([pd.DataFrame({"kind": "name", "k": list(lex_n), "v": list(lex_n.values())}),
               pd.DataFrame({"kind": "addr", "k": list(lex_a), "v": list(lex_a.values())})]).to_parquet(cache)
    return lex_n, lex_a


LEX_NAME, LEX_ADDR = build_lexicons()
log(f"lexicon sizes: name {len(LEX_NAME):,}, address {len(LEX_ADDR):,}")
print({k: LEX_NAME[k] for k in list(LEX_NAME)[:25]})
print({k: LEX_ADDR[k] for k in list(LEX_ADDR)[:15]})

# %% [markdown]
# ## 3. Normalization (lexicon-aware)

# %%
LEGAL_CANON = {
    "corporation": "corp", "incorporated": "inc", "company": "co", "limited": "ltd",
    "private": "pvt", "pte": "pvt", "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp",
    "pllc": "pllc", "ltd": "ltd", "pvt": "pvt", "inc": "inc", "corp": "corp", "co": "co",
    "sarl": "sarl", "sas": "sas", "sa": "sa", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scp": "scp", "selarl": "selarl", "gmbh": "gmbh", "ag": "ag", "bv": "bv",
    "earl": "earl", "gaec": "gaec", "scop": "scop", "scea": "scea", "eirl": "eirl", "sem": "sem",
}
LEGAL_SET = set(LEGAL_CANON.values())
LEADING_LEGAL = {"sci", "sarl", "sas", "sasu", "eurl", "snc", "scp", "selarl", "ste", "societe", "the",
                 "earl", "gaec", "scop", "scea", "eirl", "smt", "sri", "shri", "m", "s", "ms"}

ADDR_CANON = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd",
    "bd": "blvd", "drive": "dr", "lane": "ln", "place": "pl", "court": "ct", "highway": "hwy",
    "suite": "ste", "apartment": "apt", "floor": "fl", "building": "bldg", "bldg": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "near": "nr", "opposite": "opp",
    "township": "twp", "sector": "sec", "number": "no", "parkway": "pkwy", "circle": "cir",
    "terrace": "ter", "square": "sq", "center": "ctr", "centre": "ctr", "junction": "jn",
    "rue": "r", "chemin": "ch", "allee": "all", "impasse": "imp", "route": "rte", "faubourg": "fg",
    "quai": "qu", "residence": "res", "batiment": "bat", "lieu": "lieu", "dit": "dit",
    "null": "", "none": "", "na": "",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "tamil nadu": "tn", "tamilnadu": "tn", "maharashtra": "mh", "mharastr": "mh", "karnataka": "ka",
    "uttar pradesh": "up", "west bengal": "wb", "gujarat": "gj", "rajasthan": "rj", "kerala": "kl",
    "andhra pradesh": "ap", "telangana": "tg", "madhya pradesh": "mp", "haryana": "hr", "punjab": "pb",
    "bihar": "br", "odisha": "od", "orissa": "od", "jharkhand": "jh", "chhattisgarh": "cg",
    "uttarakhand": "uk", "himachal pradesh": "hp", "assam": "as", "goa": "ga", "new delhi": "delhi",
    "dilli": "delhi",
}
MULTI_REPL = sorted(((f" {k} ", f" {v} ") for k, v in {**US_STATES, **IN_STATES}.items() if " " in k),
                    key=lambda kv: -len(kv[0]))
ADDR_SINGLE = {**ADDR_CANON, **{k: v for k, v in {**US_STATES, **IN_STATES}.items() if " " not in k}}

_WWW = re.compile(r"\bwww\.")
_TLD = re.compile(r"\.(com|net|org|biz|info|io|co\.in|in|us|fr)\b")
_POSTCODE = re.compile(r"(?<!\d)\d{5,6}(?!\d)")
_CNUM = re.compile(r"\d+[a-z]?(?:\s*[-/]\s*\d+[a-z]?)+")


def _fold_lex(s, lex):
    """lowercase ASCII text; transliterate native-script words and map them through the lexicon"""
    if s.isascii():
        return s.lower()
    out = []
    for w in s.split():
        if w.isascii():
            out.append(w.lower())
        else:
            k = _word_key(w)
            out.append(lex.get(k, k))
    return " ".join(out)


def _stem(t):
    return t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t


def norm_name(s):
    """-> (full normalized name, core name without legal forms, legal-suffix string)"""
    s = _fold_lex(s, LEX_NAME).replace("&", " and ").replace("'", "")
    s = _TLD.sub(" ", _WWW.sub(" ", s)).replace(".", "")
    toks = [LEGAL_CANON.get(t, t) for t in _NON_ALNUM.sub(" ", s).split()]
    core = list(toks)
    suffix = []
    while core and core[-1] in LEGAL_SET:
        suffix.insert(0, core.pop())
    while len(core) > 1 and core[0] in LEADING_LEGAL:
        suffix.insert(0, core.pop(0))
    if not core:
        core = toks
    core = [_stem(t) for t in core]
    return " ".join(toks), " ".join(core), " ".join(suffix)


def norm_addr(s):
    """-> (normalized address, postcode, house number, house number + next token, compound number)"""
    f = _fold_lex(s, LEX_ADDR)
    pcs = _POSTCODE.findall(f)
    pc = pcs[-1] if pcs else ""
    cn = _CNUM.search(f)
    cnum = re.sub(r"\s+", "", cn.group(0)) if cn else ""
    a = " " + _NON_ALNUM.sub(" ", f.replace(".", "").replace("'", "")) + " "
    for full, ab in MULTI_REPL:
        if full in a:
            a = a.replace(full, ab)
    toks = [ADDR_SINGLE.get(t, t) for t in a.split()]
    toks = [(t.lstrip("0") or "0") if t.isdigit() else t for t in toks if t]
    hn, hn_next = "", ""
    for i, t in enumerate(toks):
        if t[0].isdigit():
            hn = t
            if i + 1 < len(toks) and toks[i + 1].isalpha():
                hn_next = t + " " + toks[i + 1]
            break
    if hn == pc:
        hn, hn_next = "", ""
    return " ".join(toks), pc, hn, hn_next, cnum


def _normalize_chunk(names, addrs):
    nn = [norm_name(x) for x in names]
    aa = [norm_addr(x) for x in addrs]
    core = [x[1] for x in nn]
    ct = [c.split(" ") for c in core]
    return pa.table({
        "name": [x[0] for x in nn],
        "core": core,
        "suffix": [x[2] for x in nn],
        "squash": [c.replace(" ", "") for c in core],
        "first": [t[0] for t in ct],
        "first2": [" ".join(t[:2]) if len(t) >= 2 else "" for t in ct],
        "addr": [x[0] for x in aa],
        "pc": [x[1] for x in aa],
        "hn": [x[2] for x in aa],
        "hn_next": [x[3] for x in aa],
        "cnum": [x[4] for x in aa],
        "native": [not x.isascii() for x in names],
    })


for raw in ("Future Producer Private Limited", "इंटरनेशनल बाबा सर्विसेज प्राइवेट लिमिटेड", "SCI Ptit Àmicale",
            "maurewilliamscolombier.com"):
    print(raw, "->", norm_name(raw))
print(norm_addr("20-158/1/A, R B Nagar, Shamshabad, Rangareddy, తెలంగాణ"))
print(norm_addr("01260 Jordan Creek Parkway, NULL, West Des Moines, Iowa"))

# %%
class Records:
    """All records of one split (S1+S2+S3) as a pyarrow table + numpy side arrays."""

    def __init__(self, split):
        cache = WORK / f"{split}_records.parquet"
        if cache.exists():
            tbl = pq.read_table(cache)
        else:
            parts = []
            for s in (1, 2, 3):
                df = read_source(DATA / split / f"{split}_source{s}.tsv")
                names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
                step = 100_000
                chunks = Parallel(n_jobs=N_JOBS)(
                    delayed(_normalize_chunk)(names[i:i + step], addrs[i:i + step])
                    for i in range(0, len(df), step))
                t = pa.concat_tables(chunks)
                t = t.append_column("entity_id", pa.array(df["entity_id"].tolist()))
                t = t.append_column("country", pa.array(df["country"].tolist()))
                t = t.append_column("src", pa.array(np.full(len(df), s, dtype=np.int8)))
                parts.append(t)
                log(f"{split} source{s}: {len(df):,} records normalized")
                del df, names, addrs, chunks
            tbl = pa.concat_tables(parts)
            pq.write_table(tbl, cache)
        tbl = tbl.combine_chunks()   # one chunk per column -> consistent dictionary encodings
        self.split = split
        self.tbl = tbl
        self.n = tbl.num_rows
        self.src = tbl["src"].to_numpy()
        self.country_codes, self.countries = pd.factorize(tbl["country"].to_pandas())
        self.country_codes = self.country_codes.astype(np.int16)
        self.id_index = pd.Index(tbl["entity_id"].to_pandas())
        self.core_blank = self._is_blank("core")
        self.addr_blank = self._is_blank("addr")
        self.native = tbl["native"].to_numpy(zero_copy_only=False)
        self._counts()

    def _is_blank(self, name):
        return pac.equal(pac.utf8_length(self.tbl[name]), 0).to_numpy(zero_copy_only=False)

    def _counts(self):
        """log counts of identical (country, address) and (country, core name) among all records"""
        for col, attr in (("addr", "addr_crowd"), ("core", "name_freq")):
            key = pac.binary_join_element_wise(self.tbl["country"], self.tbl[col], "|")
            codes = pac.dictionary_encode(key).combine_chunks().indices.to_numpy()
            cnt = np.bincount(codes)[codes].astype(np.float32)
            blank = self.addr_blank if col == "addr" else self.core_blank
            setattr(self, attr, np.where(blank, 0.0, np.log1p(cnt)).astype(np.float32))

    def col(self, name):
        return self.tbl[name]

    def take(self, name, idx):
        return self.tbl[name].take(pa.array(idx)).to_pylist()

    def idf(self, col):
        toks = pac.list_flatten(pac.utf8_split_whitespace(self.tbl[col]))
        vc = pac.value_counts(toks)
        vals = vc.field("values").to_pylist()
        cnts = vc.field("counts").to_numpy()
        keep = cnts >= 2
        w = np.log(self.n / cnts)
        return {v: float(x) for v, x, k in zip(vals, w, keep) if k}, float(math.log(self.n))


train = Records("train")
log(f"train records: {train.n:,}  countries={list(train.countries)}  native names: {train.native.mean():.2%}")
test = Records("test")
log(f"test records: {test.n:,}  countries={list(test.countries)}  native names: {test.native.mean():.2%}")

# %% [markdown]
# ## 4. Name / address embeddings (TF-IDF → SVD on the GPU)

# %%
def _to_torch_csr(X):
    return torch.sparse_csr_tensor(torch.from_numpy(X.indptr.astype(np.int64)),
                                   torch.from_numpy(X.indices.astype(np.int64)),
                                   torch.from_numpy(X.data.astype(np.float32)),
                                   size=X.shape).to(DEVICE)


def fit_projection(field, ngram):
    rng = np.random.default_rng(SEED)
    sample = []
    for rec in (train, test):
        keep = np.flatnonzero(~(rec.core_blank if field == "core" else rec.addr_blank))
        pick = np.sort(rng.choice(keep, SVD_SAMPLE // 2, replace=False))
        sample += rec.take(field, pick)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, max_features=TFIDF_MAX_FEATURES,
                          min_df=3, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(sample)
    if hasattr(vec, "stop_words_"):
        vec.stop_words_ = None
    Xg = _to_torch_csr(X).to_sparse_coo()
    _, S, V = torch.svd_lowrank(Xg, q=EMB_DIM, niter=4)
    del Xg
    torch.cuda.empty_cache()
    log(f"{field}: TF-IDF vocab {len(vec.vocabulary_):,}, SVD top singular values {S[:3].tolist()}")
    return vec, V.contiguous()


def embed_split(rec, field, vec, V):
    path = WORK / f"{rec.split}_emb_{field}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    texts = rec.col(field).to_pylist()
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(rec.n, EMB_DIM))
    step = 200_000
    spans = [(i, min(i + step, rec.n)) for i in range(0, rec.n, step)]
    gen = Parallel(n_jobs=N_JOBS, return_as="generator")(delayed(vec.transform)(texts[a:b]) for a, b in spans)
    for (a, b), X in zip(spans, gen):
        E = torch.nn.functional.normalize(_to_torch_csr(X) @ V, dim=1)
        mm[a:b] = E.half().cpu().numpy()
    mm.flush()
    del mm, texts
    log(f"{rec.split} {field} embeddings written")
    return np.load(path, mmap_mode="r")


EMB = {}
for field, ngram in (("core", (2, 4)), ("addr", (3, 4))):
    if all((WORK / f"{r.split}_emb_{field}.npy").exists() for r in (train, test)):
        vec = V = None
    else:
        vec, V = fit_projection(field, ngram)
    for rec in (train, test):
        EMB[(rec.split, field)] = embed_split(rec, field, vec, V)
    del vec, V
    gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Blocking: 3-channel GPU kNN + exact keys
# One pass over each pool computes name scores `Sn` and address scores `Sa`; the combined channel is
# `Sn + Sa` (mean cosine), so it costs no extra matrix multiply. Top-k per channel and per source are kept.

# %%
CHANNELS = ("core", "addr", "comb")


@torch.no_grad()
def gpu_knn_multi(rec, qidx):
    """-> {channel: int32 array (len(qidx), 2k)}: S2 neighbours in [:k], S3 in [k:], -1 = none."""
    EN, EA = EMB[(rec.split, "core")], EMB[(rec.split, "addr")]
    out = {ch: np.full((len(qidx), 2 * K[ch]), -1, dtype=np.int32) for ch in CHANNELS}
    for ci in np.unique(rec.country_codes[qidx]):
        qpos = np.flatnonzero(rec.country_codes[qidx] == ci)
        q = qidx[qpos]
        Qn = torch.from_numpy(np.ascontiguousarray(EN[q])).to(DEVICE)
        Qa = torch.from_numpy(np.ascontiguousarray(EA[q])).to(DEVICE)
        for si, s in enumerate((2, 3)):
            pool = np.flatnonzero((rec.src == s) & (rec.country_codes == ci))
            best = {ch: (torch.full((len(q), K[ch]), -9.0, device=DEVICE, dtype=torch.float16),
                         torch.full((len(q), K[ch]), -1, device=DEVICE, dtype=torch.int64)) for ch in CHANNELS}
            for c0 in range(0, len(pool), POOL_CHUNK):
                pidx = pool[c0:c0 + POOL_CHUNK]
                Pn = torch.from_numpy(np.ascontiguousarray(EN[pidx])).to(DEVICE)
                Pa = torch.from_numpy(np.ascontiguousarray(EA[pidx])).to(DEVICE)
                pg = torch.from_numpy(pidx).to(DEVICE)
                for q0 in range(0, len(q), QUERY_BATCH):
                    q1 = min(q0 + QUERY_BATCH, len(q))
                    sn = Qn[q0:q1] @ Pn.T
                    sa = Qa[q0:q1] @ Pa.T
                    for ch, sc in (("core", sn), ("addr", sa), ("comb", sn + sa)):
                        k = K[ch]
                        s_, i_ = sc.topk(min(k, len(pidx)), dim=1)
                        bs, bi = best[ch]
                        s_ = torch.cat([bs[q0:q1], s_], 1)
                        i_ = torch.cat([bi[q0:q1], pg[i_]], 1)
                        ts, pos = s_.topk(k, dim=1)
                        bs[q0:q1] = ts
                        bi[q0:q1] = torch.gather(i_, 1, pos)
                    del sn, sa
                del Pn, Pa, pg
            for ch in CHANNELS:
                k = K[ch]
                out[ch][qpos, si * k:(si + 1) * k] = best[ch][1].cpu().numpy().astype(np.int32)
            del best
            torch.cuda.empty_cache()
        del Qn, Qa
        log(f"  {rec.split} country={rec.countries[ci]}: kNN done for {len(q):,} queries")
    out["core"][rec.core_blank[qidx]] = -1
    out["addr"][rec.addr_blank[qidx]] = -1
    return out


EXACT_KEYS = ("ex_squash", "ex_pc_first", "ex_first_hn", "ex_first2", "ex_cnum", "ex_hn_next")


def _key(rec, name):
    c = rec.col
    if name == "ex_squash":
        return pac.if_else(pac.greater_equal(pac.utf8_length(c("squash")), 4),
                           pac.binary_join_element_wise(c("country"), c("squash"), "|"), "")
    pairs = {"ex_pc_first": ("pc", "first"), "ex_first_hn": ("first", "hn"), "ex_first2": ("first2",),
             "ex_cnum": ("cnum",), "ex_hn_next": ("hn_next",)}[name]
    ok = None
    for p in pairs:
        cond = pac.not_equal(c(p), "")
        ok = cond if ok is None else pac.and_(ok, cond)
    key = pac.binary_join_element_wise(c("country"), *[c(p) for p in pairs], "|")
    return pac.if_else(ok, key, "")


def exact_pairs(rec, qidx):
    """all (q, c, key-type) pairs sharing an exact key, same country, S1 queries vs S2/S3 pool"""
    frames = []
    for ti, name in enumerate(EXACT_KEYS):
        full = _key(rec, name)
        codes = pac.dictionary_encode(full).combine_chunks()
        empty = pac.index(codes.dictionary, "").as_py()      # -1 when no record lacks the key
        idx = codes.indices.to_numpy().astype(np.int64)
        pool = np.flatnonzero((rec.src != 1) & (idx != empty))
        sizes = np.bincount(idx[pool], minlength=len(codes.dictionary))
        pool = pool[sizes[idx[pool]] <= EXACT_BLOCK_CAP]
        qv = qidx[idx[qidx] != empty]
        pdf = pd.DataFrame({"k": idx[pool], "c": pool.astype(np.int32)})
        qdf = pd.DataFrame({"k": idx[qv], "q": qv.astype(np.int32)})
        j = qdf.merge(pdf, on="k")[["q", "c"]]
        j["t"] = np.int8(ti)
        frames.append(j)
        log(f"  exact key {name}: {len(j):,} pairs")
    ex = pd.concat(frames, ignore_index=True).sort_values("q", kind="stable")
    return ex["q"].to_numpy(np.int32), ex["c"].to_numpy(np.int32), ex["t"].to_numpy(np.int8)

# %%
def _row_cos(emb, q, c):
    out = np.empty(len(q), dtype=np.float32)
    step = 1_000_000
    for i in range(0, len(q), step):
        a = torch.from_numpy(np.ascontiguousarray(emb[q[i:i + step]])).to(DEVICE).float()
        b = torch.from_numpy(np.ascontiguousarray(emb[c[i:i + step]])).to(DEVICE).float()
        out[i:i + step] = (a * b).sum(1).cpu().numpy()
    return out


def _group_max(q, v):
    """max of v per q, broadcast back (q sorted)"""
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    mx = np.maximum.reduceat(v, starts)
    return np.repeat(mx, np.diff(np.r_[starts, len(q)]))


def _group_size(q):
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    sz = np.diff(np.r_[starts, len(q)])
    return np.repeat(sz, sz).astype(np.float32)


PF_FEATURES = ["rank_core", "rank_addr", "rank_comb", "cos_core", "cos_addr", "cos_comb", *EXACT_KEYS,
               "src_c", "addr_blank_q", "addr_blank_c", "n1_cands", "rel_cos_core", "rel_cos_addr", "rel_cos_comb"]


def stage1(rec, qchunk, qall, knn, ex):
    """candidate pairs (q-sorted) for the queries in qchunk with prefilter features"""
    pos = np.searchsorted(qall, qchunk)
    qs, cs, rk, chs = [], [], [], []
    for chi, ch in enumerate(CHANNELS):
        sub = knn[ch][pos]
        k2 = sub.shape[1]
        qs.append(np.repeat(qchunk, k2)); cs.append(sub.ravel())
        rk.append(np.tile(np.arange(k2) % K[ch], len(qchunk)).astype(np.int8))
        chs.append(np.full(len(qchunk) * k2, chi, dtype=np.int8))
    a, b = np.searchsorted(ex[0], qchunk[0]), np.searchsorted(ex[0], qchunk[-1], side="right")
    mine = a + np.flatnonzero(np.isin(ex[0][a:b], qchunk))   # the id range also holds other splits' queries
    qs.append(ex[0][mine]); cs.append(ex[1][mine])
    rk.append(np.zeros(len(mine), dtype=np.int8)); chs.append((ex[2][mine] + 3).astype(np.int8))
    q = np.concatenate(qs).astype(np.int64); c = np.concatenate(cs).astype(np.int64)
    rk = np.concatenate(rk); chs = np.concatenate(chs)
    ok = c >= 0
    key = q[ok] * rec.n + c[ok]
    uk, inv = np.unique(key, return_inverse=True)
    rk, chs = rk[ok], chs[ok]
    F = {}
    for chi, ch in enumerate(CHANNELS):
        r = np.full(len(uk), K[ch], dtype=np.int8)
        m = chs == chi
        np.minimum.at(r, inv[m], rk[m])
        F[f"rank_{ch}"] = r.astype(np.float32)
    for ti, name in enumerate(EXACT_KEYS):
        f = np.zeros(len(uk), dtype=np.float32)
        f[inv[chs == ti + 3]] = 1.0
        F[name] = f
    q, c = (uk // rec.n), (uk % rec.n)
    F["cos_core"] = _row_cos(EMB[(rec.split, "core")], q, c)
    F["cos_addr"] = _row_cos(EMB[(rec.split, "addr")], q, c)
    F["cos_comb"] = (F["cos_core"] + F["cos_addr"]) / 2
    F["src_c"] = rec.src[c].astype(np.float32)
    F["addr_blank_q"] = rec.addr_blank[q].astype(np.float32)
    F["addr_blank_c"] = rec.addr_blank[c].astype(np.float32)
    F["n1_cands"] = _group_size(q)
    for ch in CHANNELS:
        F[f"rel_cos_{ch}"] = F[f"cos_{ch}"] - _group_max(q, F[f"cos_{ch}"])
    return q, c, F


def top_m(q, score, m):
    """boolean mask keeping the m best-scored rows per q (q sorted)"""
    order = np.lexsort((-score, q))
    qs = q[order]
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    rank = np.arange(len(qs)) - np.repeat(starts, np.diff(np.r_[starts, len(qs)]))
    keep = np.zeros(len(q), dtype=bool)
    keep[order[rank < m]] = True
    return keep

# %% [markdown]
# ## 6. Ground truth + metric

# %%
gt = read_source(DATA / "train" / "train_ground_truth.tsv")
_m = gt["matched_entity_ids"].str.split(",").explode()
_m = _m[_m != ""]
GT_Q = train.id_index.get_indexer(gt["source1_entity_id"].loc[_m.index])
GT_C = train.id_index.get_indexer(_m.values)
assert (GT_Q >= 0).all() and (GT_C >= 0).all()
GT_KEYS = np.sort(GT_Q.astype(np.int64) * train.n + GT_C)
log(f"ground truth: {len(gt):,} S1, {len(GT_KEYS):,} true pairs")
del gt, _m


def pair_labels(q, c):
    keys = q.astype(np.int64) * train.n + c
    pos = np.minimum(np.searchsorted(GT_KEYS, keys), len(GT_KEYS) - 1)
    return (GT_KEYS[pos] == keys).astype(np.int8)


def n_true(qset):
    return int(np.isin(GT_Q, qset).sum())


def f05_per_entity(qset, pq_, pc_):
    qset = np.sort(qset)
    in_q = np.isin(GT_Q, qset)
    nt = np.bincount(np.searchsorted(qset, GT_Q[in_q]), minlength=len(qset))
    npred = np.bincount(np.searchsorted(qset, pq_), minlength=len(qset))
    hit = pair_labels(pq_, pc_).astype(bool)
    tp = np.bincount(np.searchsorted(qset, pq_[hit]), minlength=len(qset))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, tp / np.maximum(npred, 1), 0.0)
        r = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where((nt == 0) & (npred == 0), 1.0, f)


def macro_f05(qset, pq_, pc_):
    return f05_per_entity(qset, pq_, pc_).mean()

# %% [markdown]
# ## 7. Stage-1 blocking on train/val queries + prefilter model (GPU XGBoost)

# %%
Q_TV = np.union1d(TRAIN_Q, VAL_Q)
knn_cache = WORK / "train_knn.npz"
if knn_cache.exists():
    _z = np.load(knn_cache)
    KNN_TV = {ch: _z[ch] for ch in CHANNELS}
else:
    KNN_TV = gpu_knn_multi(train, Q_TV)
    np.savez(knn_cache, **KNN_TV)
EX_TV = exact_pairs(train, Q_TV)

# stage-1 recall on validation (before prefilter)
_found = 0
for a in range(0, len(VAL_Q), STAGE_CHUNK):
    qv = VAL_Q[a:a + STAGE_CHUNK]
    q_, c_, _ = stage1(train, qv, Q_TV, KNN_TV, EX_TV)
    _found += pair_labels(q_, c_).sum()
    _n1 = len(q_) / len(qv)
log(f"VAL stage-1 blocking recall {_found / n_true(VAL_Q):.4f} (baseline 0.8364), ~{_n1:.0f} cands/S1 before prefilter")

# prefilter: fit on a subset of training entities, choose the cut M on a subset of validation entities
rng = np.random.default_rng(SEED + 1)
pf_tr_q = np.sort(rng.choice(TRAIN_Q, PF_TRAIN_S1, replace=False))
pf_ev_q = np.sort(rng.choice(VAL_Q, PF_EVAL_S1, replace=False))


def stage1_matrix(qset):
    Xs, ys, qs = [], [], []
    for a in range(0, len(qset), STAGE_CHUNK):
        q_, c_, F = stage1(train, qset[a:a + STAGE_CHUNK], Q_TV, KNN_TV, EX_TV)
        Xs.append(np.column_stack([F[f] for f in PF_FEATURES]).astype(np.float32))
        ys.append(pair_labels(q_, c_)); qs.append(q_)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(qs)


Xp, yp, _ = stage1_matrix(pf_tr_q)
Xe, ye, qe = stage1_matrix(pf_ev_q)
PF_PARAMS = dict(objective="binary:logistic", eval_metric="aucpr", device="cuda", tree_method="hist",
                 max_depth=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.9, seed=SEED)
pf_model = xgb.train(PF_PARAMS, xgb.QuantileDMatrix(Xp, yp, feature_names=PF_FEATURES), 400,
                     evals=[(xgb.DMatrix(Xe, ye, feature_names=PF_FEATURES), "val")],
                     early_stopping_rounds=30, verbose_eval=100)
pf_model.save_model(WORK / "prefilter.json")
pe = pf_model.predict(xgb.DMatrix(Xe, feature_names=PF_FEATURES), iteration_range=(0, pf_model.best_iteration + 1))
PF_M = PF_M_GRID[-1]
for m_ in PF_M_GRID:
    ret = ye[top_m(qe, pe, m_)].sum() / ye.sum()
    print(f"  keep top {m_:3d}: retains {ret:.4f} of blocked true pairs")
    if ret >= PF_RETENTION:
        PF_M = m_
        break
log(f"prefilter keeps top {PF_M} candidates per S1")
del Xp, yp, Xe, ye, qe, pe
gc.collect()

# %% [markdown]
# ## 8. Full pair features

# %%
FEATURES = [
    *PF_FEATURES, "pf", "pf_rank",
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw_squash", "n_full_tset", "squash_contains",
    "first_eq", "suffix_eq", "len_core_q", "len_core_c", "native_c",
    "a_ratio", "a_tset", "a_tsort", "pc_eq", "hn_eq", "cnum_eq",
    "t_wjacc_n", "t_onlyq_n", "t_onlyc_n", "t_wjacc_a", "t_num_jacc", "t_onlyc_a", "addr_len_ratio",
    "crowd_addr_q", "crowd_addr_c", "freq_name_c",
    "rel_n_tset", "rel_a_tset", "rel_pf",
]


def _eq_or_missing(a, b):
    a = np.array(a, dtype=object)
    b = np.array(b, dtype=object)
    miss = (a == "") | (b == "")
    return np.where(miss, -1.0, (a == b).astype(np.float32)).astype(np.float32)


def _tok_feats(cq, cc, aq, ac, idf_n, mx_n, idf_a, mx_a):
    n = len(cq)
    out = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        A, B = set(cq[i].split()), set(cc[i].split())
        I = A & B
        wa = sum(idf_n.get(t, mx_n) for t in A)
        wb = sum(idf_n.get(t, mx_n) for t in B)
        wi = sum(idf_n.get(t, mx_n) for t in I)
        den = wa + wb - wi
        out[i, 0] = wi / den if den > 0 else 0.0
        out[i, 1] = max((idf_n.get(t, mx_n) for t in A - B), default=0.0)
        out[i, 2] = max((idf_n.get(t, mx_n) for t in B - A), default=0.0)
        X, Y = set(aq[i].split()), set(ac[i].split())
        if X and Y:
            I = X & Y
            wx = sum(idf_a.get(t, mx_a) for t in X)
            wy = sum(idf_a.get(t, mx_a) for t in Y)
            wi = sum(idf_a.get(t, mx_a) for t in I)
            out[i, 3] = wi / (wx + wy - wi)
            NX = {t for t in X if t[0].isdigit()}
            NY = {t for t in Y if t[0].isdigit()}
            U = NX | NY
            out[i, 4] = len(NX & NY) / len(U) if U else -1.0
            out[i, 5] = max((idf_a.get(t, mx_a) for t in Y - X), default=0.0)
        else:
            out[i, 3] = out[i, 4] = out[i, 5] = -1.0
        out[i, 6] = len(ac[i]) / max(len(aq[i]), 1)
    return out


IDF = {}


def pair_features(rec, q, c, F):
    """full feature matrix (float32, columns = FEATURES) for q-sorted pairs"""
    cq, cc = rec.take("core", q), rec.take("core", c)
    aq, ac = rec.take("addr", q), rec.take("addr", c)
    sq, sc = rec.take("squash", q), rec.take("squash", c)
    fq, fc = rec.take("name", q), rec.take("name", c)
    cd = lambda a, b, scorer: process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    F["n_ratio"] = cd(cq, cc, fuzz.ratio)
    F["n_tset"] = cd(cq, cc, fuzz.token_set_ratio)
    F["n_tsort"] = cd(cq, cc, fuzz.token_sort_ratio)
    F["n_partial"] = cd(cq, cc, fuzz.partial_ratio)
    F["n_jw_squash"] = cd(sq, sc, JaroWinkler.normalized_similarity)
    F["n_full_tset"] = cd(fq, fc, fuzz.token_set_ratio)
    F["a_ratio"] = cd(aq, ac, fuzz.ratio)
    F["a_tset"] = cd(aq, ac, fuzz.token_set_ratio)
    F["a_tsort"] = cd(aq, ac, fuzz.token_sort_ratio)
    F["squash_contains"] = np.fromiter(
        (1.0 if len(a) >= 4 and len(b) >= 4 and (a in b or b in a) else 0.0 for a, b in zip(sq, sc)),
        dtype=np.float32, count=len(q))
    for col, name in (("first", "first_eq"), ("suffix", "suffix_eq"), ("pc", "pc_eq"), ("hn", "hn_eq"),
                      ("cnum", "cnum_eq")):
        F[name] = _eq_or_missing(rec.take(col, q), rec.take(col, c))
    F["len_core_q"] = np.fromiter(map(len, cq), dtype=np.float32, count=len(q))
    F["len_core_c"] = np.fromiter(map(len, cc), dtype=np.float32, count=len(q))
    F["native_c"] = rec.native[c].astype(np.float32)
    idf_n, mx_n = IDF[(rec.split, "core")]
    idf_a, mx_a = IDF[(rec.split, "addr")]
    step = max(1, -(-len(q) // N_JOBS))
    parts = Parallel(n_jobs=N_JOBS)(
        delayed(_tok_feats)(cq[i:i + step], cc[i:i + step], aq[i:i + step], ac[i:i + step], idf_n, mx_n, idf_a, mx_a)
        for i in range(0, len(q), step))
    T = np.vstack(parts)
    for j, name in enumerate(("t_wjacc_n", "t_onlyq_n", "t_onlyc_n", "t_wjacc_a", "t_num_jacc", "t_onlyc_a",
                              "addr_len_ratio")):
        F[name] = T[:, j]
    F["crowd_addr_q"] = rec.addr_crowd[q]
    F["crowd_addr_c"] = rec.addr_crowd[c]
    F["freq_name_c"] = rec.name_freq[c]
    F["rel_n_tset"] = F["n_tset"] - _group_max(q, F["n_tset"])
    F["rel_a_tset"] = F["a_tset"] - _group_max(q, F["a_tset"])
    F["rel_pf"] = F["pf"] - _group_max(q, F["pf"])
    return np.column_stack([F[f] for f in FEATURES]).astype(np.float32)


def candidates_and_features(rec, qchunk, qall, knn, ex):
    """stage-1 → prefilter → top-M → full features for one chunk of S1 queries"""
    q, c, F = stage1(rec, qchunk, qall, knn, ex)
    X1 = np.column_stack([F[f] for f in PF_FEATURES]).astype(np.float32)
    pf = pf_model.predict(xgb.DMatrix(X1, feature_names=PF_FEATURES), iteration_range=(0, pf_model.best_iteration + 1))
    keep = top_m(q, pf, PF_M)
    q, c = q[keep], c[keep]
    F = {k: v[keep] for k, v in F.items()}
    F["pf"] = pf[keep].astype(np.float32)
    order = np.lexsort((-F["pf"], q))
    rank = np.empty(len(q), dtype=np.float32)
    qs = q[order]
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    rank[order] = np.arange(len(qs)) - np.repeat(starts, np.diff(np.r_[starts, len(qs)]))
    F["pf_rank"] = rank
    return q, c, pair_features(rec, q, c, F)


for rec in (train, test):
    for col in ("core", "addr"):
        IDF[(rec.split, col)] = rec.idf(col)
log(f"IDF vocabularies: train core {len(IDF[('train', 'core')][0]):,}, addr {len(IDF[('train', 'addr')][0]):,}")
del test          # free RAM during training; reloaded from the parquet cache for inference
gc.collect()

# %% [markdown]
# ## 9. Build training / validation pairs

# %%
def build_pairs(name, qset):
    cache = WORK / f"{name}_pairs.npz"
    if cache.exists():
        z = np.load(cache)
        return z["q"], z["c"], z["X"], z["y"]
    Q, C, X = [], [], []
    for a in range(0, len(qset), STAGE_CHUNK):
        q, c, x = candidates_and_features(train, qset[a:a + STAGE_CHUNK], Q_TV, KNN_TV, EX_TV)
        Q.append(q.astype(np.int32)); C.append(c.astype(np.int32)); X.append(x)
        log(f"  {name}: {min(a + STAGE_CHUNK, len(qset)):,}/{len(qset):,} S1 done")
    Q, C, X = np.concatenate(Q), np.concatenate(C), np.vstack(X)
    y = pair_labels(Q, C)
    np.savez(cache, q=Q, c=C, X=X, y=y)
    return Q, C, X, y


va_q, va_c, va_X, va_y = build_pairs("val", VAL_Q)
tr_q, tr_c, tr_X, tr_y = build_pairs("train", TRAIN_Q)
for nm, qset, q_, c_, y_ in (("train", TRAIN_Q, tr_q, tr_c, tr_y), ("val", VAL_Q, va_q, va_c, va_y)):
    oracle = macro_f05(qset, q_[y_ == 1], c_[y_ == 1])
    log(f"{nm}: final candidate recall {y_.sum() / n_true(qset):.4f}, {len(q_) / len(qset):.1f} cands/S1, "
        f"oracle macro-F0.5 {oracle:.4f}  (baseline val: 0.8364 / 39.7 / 0.9275)")
del KNN_TV, EX_TV
gc.collect()

# %% [markdown]
# ## 10. Stage-1 matcher (XGBoost on GPU) with out-of-fold predictions

# %%
XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric=["logloss", "aucpr"], device="cuda", tree_method="hist",
    max_depth=9, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    max_bin=256, seed=SEED,
)


def fit(X, y, Xv, yv, names, rounds=3000):
    dtr = xgb.QuantileDMatrix(X, y, feature_names=names)
    dva = xgb.QuantileDMatrix(Xv, yv, ref=dtr, feature_names=names)
    b = xgb.train(XGB_PARAMS, dtr, num_boost_round=rounds, evals=[(dva, "val")],
                  early_stopping_rounds=60, verbose_eval=250)
    del dtr, dva
    gc.collect()
    return b


def predict(b, X, names):
    return b.predict(xgb.DMatrix(X, feature_names=names), iteration_range=(0, b.best_iteration + 1))


fold = (np.random.default_rng(SEED + 2).integers(0, 2, size=train.n)[tr_q]).astype(np.int8)
p_oof = np.zeros(len(tr_q), dtype=np.float32)
for f in (0, 1):
    m_ = fold != f
    b = fit(tr_X[m_], tr_y[m_], va_X, va_y, FEATURES)
    p_oof[~m_] = predict(b, tr_X[~m_], FEATURES)
    log(f"fold {f}: best iteration {b.best_iteration}")
    del b
    gc.collect()
m1 = fit(tr_X, tr_y, va_X, va_y, FEATURES)
m1.save_model(WORK / "m1.json")
p1_va = predict(m1, va_X, FEATURES)
log(f"stage-1 matcher best iteration {m1.best_iteration}")
imp = pd.Series(m1.get_score(importance_type="gain"))
print(imp.sort_values(ascending=False).head(25).round(1).to_string())

# %% [markdown]
# ## 11. Group-context model (stacked on stage-1 probabilities)

# %%
G_FEATURES = ["p1", "g_rank", "g_rank_src", "g_max", "g_second", "g_gap", "g_margin", "g_n05", "g_n08",
              "g_sum", "g_max_other_src", "g_max_same_src"]


def group_features(q, src, p):
    """features of a candidate relative to all candidates of the same S1 (q sorted)"""
    n = len(q)
    order = np.lexsort((-p, q))
    qs, ps, ss = q[order], p[order], src[order]
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    sizes = np.diff(np.r_[starts, n])
    rank_sorted = np.arange(n) - np.repeat(starts, sizes)
    g = {}
    g["g_rank"] = np.empty(n, np.float32); g["g_rank"][order] = rank_sorted
    mx = np.repeat(ps[starts], sizes)
    second = np.where(sizes > 1, ps[np.minimum(starts + 1, n - 1)], 0.0)
    g["g_max"] = np.empty(n, np.float32); g["g_max"][order] = mx
    g["g_second"] = np.empty(n, np.float32); g["g_second"][order] = np.repeat(second, sizes)
    for name, cut in (("g_n05", 0.5), ("g_n08", 0.8)):
        g[name] = np.empty(n, np.float32)
        g[name][order] = np.repeat(np.add.reduceat((ps >= cut).astype(np.float32), starts), sizes)
    g["g_sum"] = np.empty(n, np.float32); g["g_sum"][order] = np.repeat(np.add.reduceat(ps, starts), sizes)
    df = pd.DataFrame({"q": q, "s": src, "p": p})
    g["g_rank_src"] = df.groupby(["q", "s"])["p"].rank(ascending=False, method="first").to_numpy(np.float32) - 1
    same = df.groupby(["q", "s"])["p"].transform("max").to_numpy(np.float32)
    per = df.groupby(["q", "s"])["p"].max().unstack(fill_value=0.0)
    other_s = np.where(src == 2, 3, 2)
    other = np.zeros(n, np.float32)
    for s in (2, 3):
        if s in per.columns:
            mask = other_s == s
            other[mask] = per[s].reindex(q[mask]).fillna(0.0).to_numpy(np.float32)
    g["g_max_same_src"] = same
    g["g_max_other_src"] = other
    g["p1"] = p.astype(np.float32)
    g["g_gap"] = p - g["g_max"]
    g["g_margin"] = g["g_max"] - g["g_second"]
    return np.column_stack([g[f] for f in G_FEATURES]).astype(np.float32)


ALL_FEATURES = FEATURES + G_FEATURES
Gtr = group_features(tr_q, train.src[tr_c], p_oof)
Gva = group_features(va_q, train.src[va_c], p1_va)
m2 = fit(np.hstack([tr_X, Gtr]), tr_y, np.hstack([va_X, Gva]), va_y, ALL_FEATURES)
m2.save_model(WORK / "m2.json")
p2_va = predict(m2, np.hstack([va_X, Gva]), ALL_FEATURES)
log(f"group-context model best iteration {m2.best_iteration}")
del Gtr, tr_X
gc.collect()

# %% [markdown]
# ## 12. Decision rule + per-country threshold tuning on validation (macro F0.5)

# %%
def resolve(q, c, src_c, prob, thr):
    """keep pairs >= their threshold, each S2/S3 record to its best S1 only, per-source caps"""
    df = pd.DataFrame({"q": q, "c": c, "s": src_c, "p": prob, "t": thr})
    df = df[df["p"] >= df["t"]]
    df = df.sort_values("p", ascending=False).drop_duplicates("c")
    df["r"] = df.groupby(["q", "s"]).cumcount()
    cap = df["s"].map(MATCH_CAP).fillna(max(MATCH_CAP.values()))
    df = df[df["r"] < cap]
    return df["q"].to_numpy(), df["c"].to_numpy()


GRID = np.round(np.arange(0.20, 0.96, 0.025), 3)
va_src = train.src[va_c]
va_ctry = np.asarray(train.countries)[train.country_codes[va_q]]


def tune(prob):
    lo = prob >= GRID[0]
    return _tune(va_q[lo], va_c[lo], va_src[lo], prob[lo], va_ctry[lo])


def _tune(va_q, va_c, va_src, prob, va_ctry):
    glob = {t: macro_f05(VAL_Q, *resolve(va_q, va_c, va_src, prob, np.full(len(prob), t))) for t in GRID}
    t_glob = max(glob, key=glob.get)
    per = {}
    for ctry in np.unique(va_ctry):
        qset = VAL_Q[np.asarray(train.countries)[train.country_codes[VAL_Q]] == ctry]
        m_ = va_ctry == ctry
        sc = {t: macro_f05(qset, *resolve(va_q[m_], va_c[m_], va_src[m_], prob[m_], np.full(m_.sum(), t)))
              for t in GRID}
        per[ctry] = max(sc, key=sc.get)
    thr = np.array([per.get(x, t_glob) for x in va_ctry], dtype=np.float32)
    return t_glob, glob[t_glob], per, macro_f05(VAL_Q, *resolve(va_q, va_c, va_src, prob, thr))


results = {}
for nm, prob in (("stage-1 matcher", p1_va), ("group-context model", p2_va)):
    t_glob, f_glob, per, f_per = tune(prob)
    results[nm] = (t_glob, f_glob, per, f_per)
    log(f"{nm}: global threshold {t_glob} → macro F0.5 {f_glob:.4f} | per-country {per} → {f_per:.4f}")
BEST_MODEL = max(results, key=lambda k: max(results[k][1], results[k][3]))
t_glob, f_glob, per, f_per = results[BEST_MODEL]
USE_PER_COUNTRY = f_per > f_glob
THRESH = per if USE_PER_COUNTRY else {}
T_DEFAULT = t_glob
best_prob = p2_va if BEST_MODEL == "group-context model" else p1_va
thr_va = np.array([THRESH.get(x, T_DEFAULT) for x in va_ctry], dtype=np.float32)
fq, fc = resolve(va_q, va_c, va_src, best_prob, thr_va)
f_ent = f05_per_entity(VAL_Q, fq, fc)
log(f"VALIDATION macro F0.5 = {f_ent.mean():.4f} using {BEST_MODEL}, "
    f"{'per-country' if USE_PER_COUNTRY else 'global'} thresholds {THRESH or T_DEFAULT} (baseline 0.8871)")
ctry_val = np.asarray(train.countries)[train.country_codes[VAL_Q]]
nt_val = np.bincount(np.searchsorted(VAL_Q, GT_Q[np.isin(GT_Q, VAL_Q)]), minlength=len(VAL_Q))
print(pd.DataFrame({"country": ctry_val, "f": f_ent}).groupby("country").f.agg(["mean", "size"]).round(4))
print(pd.DataFrame({"n_true": pd.cut(nt_val, [-1, 0, 1, 3, 5, 20], labels=["singleton", "1", "2-3", "4-5", "6+"]),
                    "f": f_ent}).groupby("n_true", observed=True).f.agg(["mean", "size"]).round(4))
pd.DataFrame({k: [v[0], v[1], str(v[2]), v[3]] for k, v in results.items()},
             index=["t_global", "f_global", "t_per_country", "f_per_country"]).to_csv(WORK / "val_scores.csv")
del va_X, Gva
gc.collect()

# %% [markdown]
# ## 13. Test inference → `output/candidate_pairs_v2.tsv`, `output/matching_results_v2.tsv`

# %%
test = Records("test")
test_q = np.flatnonzero(test.src == 1)
knn_cache = WORK / "test_knn.npz"
if knn_cache.exists():
    _z = np.load(knn_cache)
    KNN_TE = {ch: _z[ch] for ch in CHANNELS}
else:
    KNN_TE = gpu_knn_multi(test, test_q)
    np.savez(knn_cache, **KNN_TE)
EX_TE = exact_pairs(test, test_q)

ids_te = test.tbl["entity_id"].to_numpy(zero_copy_only=False)
te_ctry = np.asarray(test.countries)
t_min = min([T_DEFAULT, *THRESH.values()])
kept, n_cands = [], 0
with open(OUT_CAND, "w", encoding="utf-8", newline="") as fcand:
    fcand.write("source1_entity_id\tcandidate_entity_ids\n")
    for a in range(0, len(test_q), STAGE_CHUNK):
        qch = test_q[a:a + STAGE_CHUNK]
        q, c, X = candidates_and_features(test, qch, test_q, KNN_TE, EX_TE)
        p = predict(m1, X, FEATURES)
        if BEST_MODEL == "group-context model":
            p = predict(m2, np.hstack([X, group_features(q, test.src[c], p)]), ALL_FEATURES)
        thr = np.array([THRESH.get(x, T_DEFAULT) for x in te_ctry[test.country_codes[q]]], dtype=np.float32)
        sel = p >= np.minimum(thr, t_min)
        kept.append(pd.DataFrame({"q": q[sel], "c": c[sel], "p": p[sel], "t": thr[sel]}))
        # candidate list = the exact set scored by the matcher
        lists = pd.Series(ids_te[c]).groupby(q).agg(",".join)
        col = pd.Series(qch).map(lists).fillna("")
        pd.DataFrame({"a": ids_te[qch], "b": col.to_numpy()}).to_csv(
            fcand, sep="\t", index=False, header=False, quoting=csv.QUOTE_NONE)
        n_cands += len(q)
        log(f"  test scored {min(a + STAGE_CHUNK, len(test_q)):,}/{len(test_q):,} S1 ({n_cands:,} candidate pairs)")
log(f"wrote {OUT_CAND} ({len(test_q):,} rows, {n_cands / len(test_q):.1f} candidates per S1)")

kept = pd.concat(kept, ignore_index=True)
mq, mc = resolve(kept["q"].to_numpy(), kept["c"].to_numpy(), test.src[kept["c"].to_numpy()],
                 kept["p"].to_numpy(), kept["t"].to_numpy())
lists = pd.Series(ids_te[mc]).groupby(mq).agg(",".join)
pd.DataFrame({"source1_entity_id": ids_te[test_q],
              "matched_entity_ids": pd.Series(test_q).map(lists).fillna("").to_numpy()}).to_csv(
    OUT_MATCH, sep="\t", index=False, quoting=csv.QUOTE_NONE)
log(f"wrote {OUT_MATCH}: {len(mq):,} matches, {len(np.unique(mq)):,} of {len(test_q):,} S1 with >=1 match")
by_c = pd.Series(te_ctry[test.country_codes[test_q]]).value_counts()
has = pd.Series(te_ctry[test.country_codes[np.unique(mq)]]).value_counts()
print(pd.DataFrame({"S1": by_c, "with_match": has, "share": (has / by_c).round(3)}))

# %%
res = subprocess.run([sys.executable, str(ROOT / "utils" / "validate_submission.py"),
                      "--matching", str(OUT_MATCH), "--candidate", str(OUT_CAND),
                      "--test-dir", str(DATA / "test"), "--check-ids"],
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
print(res.stdout[-3000:], res.stderr[-3000:])
log("done")
