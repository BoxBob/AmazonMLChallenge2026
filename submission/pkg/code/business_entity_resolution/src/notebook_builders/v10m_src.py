# %% [markdown]
# # Business Entity Resolution — v10: larger prefilter cut, validation-only experiment
#
# The cross-encoder carries France (T25) but was English-only. Here French pairs are re-scored with the multilingual
# `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0), fine-tuned with the identical recipe (work/v9/mce_model).
#
#
# France has no labels, and v5 showed that an unverified France change can hurt (self-training: 0.972 vs v4 0.973, P19).
# The one trustworthy signal for France is the leaderboard. This notebook fixes US/India at v4 and writes France variants
# that differ **only** in France's decision threshold. Comparing their scores tells whether v4's French probabilities are
# too cautious (a lower threshold wins) or too eager (a higher threshold wins).
#
# Also saves `work/v6probe/france_v4_probs.npz` (every French candidate pair with its v4 probability) for fast variants.

# %%
import os
import tempfile
from pathlib import Path

# project root = folder holding dataset/, utils/ and output/; override with the ER_ROOT env var
ROOT = Path(os.environ.get("ER_ROOT", r"D:\AmazonMLChallenge2026"))
DATA = ROOT / "dataset"
WORK = ROOT / "work" / f"v10m{os.environ.get('PF_M_OVERRIDE', '80')}"
PREV = ROOT / "work" / "v4"      # v4 artifacts reused by v5
SHARED = ROOT / "work" / "v2"   # caches shared with v2 (same normalization/embeddings/kNN)
TMP = ROOT / "work" / "tmp"
OUT = ROOT / "output"
for p in (WORK, SHARED, TMP, OUT, OUT / "lb_probe"):
    p.mkdir(parents=True, exist_ok=True)
OUT_CAND = OUT / "lb_probe" / "candidate_pairs_v4.tsv"
OUT_MATCH = OUT / "lb_probe" / "matching_results_v9usin_raw.tsv"
V4_MATCH = OUT / "matching_results_v4.tsv"
V4_CAND = OUT / "candidate_pairs_v4.tsv"
ST_POS, ST_NEG = 0.97, 0.03      # self-training confidence cuts
ST_ROUNDS = 0                      # probe: no self-training
PROBE_T = [0.55, 0.625, 0.70, 0.775, 0.85]
ST_NEG_KEEP = 0.25   # share of confident French negatives kept (memory)
ST_TRAIN_KEEP = 0.6  # share of training rows kept next to the pseudo-labels (memory)

# keep every temp file (joblib memmaps, tempfile, ...) off the C: drive
os.environ["TEMP"] = os.environ["TMP"] = os.environ["JOBLIB_TEMP_FOLDER"] = str(TMP)
os.environ["HF_HOME"] = str(ROOT / "work" / "hf")   # Hugging Face model cache on D:
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
K_BLANK = 5                 # v3: name kNN neighbours per source among blank-address pool records
BLANK_KEY_CAP = 1000        # v3: block cap of the blank-address squashed-name key
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"   # Apache-2.0, 33M params
CE_TRAIN_S1 = 150_000       # training entities for the cross-encoder (top-8 candidates each)
CE_MAX_LEN = 96
CE_BAND = (0.01, 0.99)      # the cross-encoder re-scores pairs whose GBDT probability is in this band
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
    cache = SHARED / "lexicons.parquet"
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
STREET_STOP = {"r", "rue", "st", "str", "ave", "blvd", "rd", "dr", "ln", "pl", "ct", "hwy", "ste", "apt", "fl", "bldg",
               "ch", "all", "imp", "rte", "fg", "qu", "res", "bat", "no", "n", "s", "e", "w", "ne", "nw", "se", "sw",
               "de", "du", "des", "la", "le", "les", "l", "d", "of", "the", "and", "sq", "pkwy", "cir", "ter", "ctr",
               "jn", "nr", "opp", "sec", "twp", "unit", "road", "street", "avenue", "lane", "cross", "main", "floor",
               "plot", "flat", "house", "door", "shop", "gali", "marg", "nagar", "chemin", "allee", "impasse",
               "boulevard", "place", "quai", "route", "cours", "lieu", "dit", "po", "box", "suite"}


def _hn_street_chunk(addrs, pcs):
    """house number + first distinctive street word; street types and articles of any language are skipped"""
    out = []
    for addr, pc in zip(addrs, pcs):
        key = ""
        toks = addr.split()
        for i, t in enumerate(toks):
            if t[0].isdigit() and t != pc:
                for u in toks[i + 1:i + 6]:
                    if u.isalpha() and len(u) >= 3 and u not in STREET_STOP:
                        key = t + " " + u
                        break
                break
        out.append(key)
    return out


def _chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


class Records:
    """All records of one split (S1+S2+S3) as a pyarrow table + numpy side arrays."""

    def __init__(self, split):
        cache = SHARED / f"{split}_records.parquet"
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
        self._v4_columns()

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

    def _v4_columns(self):
        """v4: house number + first distinctive street word, and S1-side ambiguity counts"""
        hs = Parallel(n_jobs=N_JOBS)(delayed(_hn_street_chunk)(a, p) for a, p in zip(
            _chunks(self.tbl["addr"].to_pylist(), 500_000), _chunks(self.tbl["pc"].to_pylist(), 500_000)))
        self.tbl = self.tbl.append_column("hn_street", pa.array([x for h in hs for x in h]))
        s1 = self.src == 1
        for col, attr, minlen in (("squash", "s1_name", 3), ("addr", "s1_addr", 1)):
            key = pac.binary_join_element_wise(self.tbl["country"], self.tbl[col], "|")
            codes = pac.dictionary_encode(key).combine_chunks().indices.to_numpy()
            blank = pac.less(pac.utf8_length(self.tbl[col]), minlen).to_numpy(zero_copy_only=False)
            n_s1 = np.bincount(codes[s1], minlength=codes.max() + 1)
            n_pool = np.bincount(codes[~s1], minlength=codes.max() + 1)
            setattr(self, attr + "_cnt", np.where(blank, 0, n_s1[codes]).astype(np.float32))
            setattr(self, attr.replace("s1_", "pool_") + "_cnt", np.where(blank, 0, n_pool[codes]).astype(np.float32))

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
    path = SHARED / f"{rec.split}_emb_{field}.npy"
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
    if all((SHARED / f"{r.split}_emb_{field}.npy").exists() for r in (train, test)):
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


@torch.no_grad()
def gpu_knn_blank(rec, qidx):
    """v3: name kNN restricted to blank-address S2/S3 records -> int32 (len(qidx), 2*K_BLANK)"""
    EN = EMB[(rec.split, "core")]
    out = np.full((len(qidx), 2 * K_BLANK), -1, dtype=np.int32)
    for ci in np.unique(rec.country_codes[qidx]):
        qpos = np.flatnonzero(rec.country_codes[qidx] == ci)
        Q = torch.from_numpy(np.ascontiguousarray(EN[qidx[qpos]])).to(DEVICE)
        for si, s in enumerate((2, 3)):
            pool = np.flatnonzero((rec.src == s) & (rec.country_codes == ci) & rec.addr_blank & ~rec.core_blank)
            if len(pool) == 0:
                continue
            P = torch.from_numpy(np.ascontiguousarray(EN[pool])).to(DEVICE)
            k = min(K_BLANK, len(pool))
            for q0 in range(0, len(qpos), 2048):
                idx = (Q[q0:q0 + 2048] @ P.T).topk(k, dim=1).indices.cpu().numpy()
                out[qpos[q0:q0 + 2048], si * K_BLANK:si * K_BLANK + k] = pool[idx]
            del P
        del Q
        torch.cuda.empty_cache()
    out[rec.core_blank[qidx]] = -1
    log(f"  {rec.split}: blank-address name kNN done for {len(qidx):,} queries")
    return out


EXACT_KEYS = ("ex_squash", "ex_pc_first", "ex_first_hn", "ex_first2", "ex_cnum", "ex_hn_next", "ex_blank_squash",
              "ex_name_hn", "ex_hn_street")


def _key(rec, name):
    c = rec.col
    if name in ("ex_squash", "ex_blank_squash"):
        return pac.if_else(pac.greater_equal(pac.utf8_length(c("squash")), 4),
                           pac.binary_join_element_wise(c("country"), c("squash"), "|"), "")
    if name == "ex_name_hn":                      # v4: full name + house number (language independent)
        ok = pac.and_(pac.greater_equal(pac.utf8_length(c("squash")), 4), pac.not_equal(c("hn"), ""))
        return pac.if_else(ok, pac.binary_join_element_wise(c("country"), c("squash"), c("hn"), "|"), "")
    pairs = {"ex_pc_first": ("pc", "first"), "ex_first_hn": ("first", "hn"), "ex_first2": ("first2",),
             "ex_cnum": ("cnum",), "ex_hn_next": ("hn_next",), "ex_hn_street": ("hn_street",)}[name]
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
        pool_mask = (rec.src != 1) & (idx != empty)
        if name == "ex_blank_squash":             # v3: blank-address records get their own key block
            pool_mask &= rec.addr_blank
        pool = np.flatnonzero(pool_mask)
        sizes = np.bincount(idx[pool], minlength=len(codes.dictionary))
        pool = pool[sizes[idx[pool]] <= (BLANK_KEY_CAP if name == "ex_blank_squash" else EXACT_BLOCK_CAP)]
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


STAGE_CHANNELS = CHANNELS + ("blank",)
K["blank"] = K_BLANK
PF_FEATURES = ["rank_core", "rank_addr", "rank_comb", "rank_blank", "cos_core", "cos_addr", "cos_comb", *EXACT_KEYS,
               "src_c", "addr_blank_q", "addr_blank_c", "n1_cands", "rel_cos_core", "rel_cos_addr", "rel_cos_comb"]


def stage1(rec, qchunk, qall, knn, ex):
    """candidate pairs (q-sorted) for the queries in qchunk with prefilter features"""
    pos = np.searchsorted(qall, qchunk)
    qs, cs, rk, chs = [], [], [], []
    for chi, ch in enumerate(STAGE_CHANNELS):
        sub = knn[ch][pos]
        k2 = sub.shape[1]
        qs.append(np.repeat(qchunk, k2)); cs.append(sub.ravel())
        rk.append(np.tile(np.arange(k2) % K[ch], len(qchunk)).astype(np.int8))
        chs.append(np.full(len(qchunk) * k2, chi, dtype=np.int8))
    lo, hi = np.searchsorted(ex[0], qchunk), np.searchsorted(ex[0], qchunk, side="right")   # rows of exactly these queries
    n_ex = hi - lo
    mine = np.repeat(lo, n_ex) + (np.arange(n_ex.sum()) - np.repeat(np.cumsum(n_ex) - n_ex, n_ex))
    qs.append(ex[0][mine]); cs.append(ex[1][mine])
    rk.append(np.zeros(len(mine), dtype=np.int8)); chs.append((ex[2][mine] + len(STAGE_CHANNELS)).astype(np.int8))
    q = np.concatenate(qs).astype(np.int64); c = np.concatenate(cs).astype(np.int64)
    rk = np.concatenate(rk); chs = np.concatenate(chs)
    ok = c >= 0
    key = q[ok] * rec.n + c[ok]
    uk, inv = np.unique(key, return_inverse=True)
    rk, chs = rk[ok], chs[ok]
    F = {}
    for chi, ch in enumerate(STAGE_CHANNELS):
        r = np.full(len(uk), K[ch], dtype=np.int8)
        m = chs == chi
        np.minimum.at(r, inv[m], rk[m])
        F[f"rank_{ch}"] = r.astype(np.float32)
    for ti, name in enumerate(EXACT_KEYS):
        f = np.zeros(len(uk), dtype=np.float32)
        f[inv[chs == ti + len(STAGE_CHANNELS)]] = 1.0
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
knn_cache = SHARED / "train_knn.npz"
if knn_cache.exists():
    _z = np.load(knn_cache)
    KNN_TV = {ch: _z[ch] for ch in CHANNELS}
else:
    KNN_TV = gpu_knn_multi(train, Q_TV)
    np.savez(knn_cache, **KNN_TV)
_bc = PREV / "train_knn_blank.npy"
KNN_TV["blank"] = np.load(_bc) if _bc.exists() else gpu_knn_blank(train, Q_TV)
np.save(_bc, KNN_TV["blank"])
EX_TV = exact_pairs(train, Q_TV)

# v5: reuse the v4 prefilter model and its cut M (read from the v4 run log)
pf_model = xgb.Booster(model_file=str(PREV / "prefilter.json"))
PF_M = int(os.environ.get("PF_M_OVERRIDE", "80"))   # v10: larger prefilter cut (v4 used 40)
log(f"prefilter keeps top {PF_M} candidates per S1")
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
    # v3: numeric-aware address features (Ditto span typing / Linacre: numbers are not typos)
    "hn_lev", "hn_contains", "hn_gap", "num_shared", "num_only_c", "num_only_q", "num_min_lev",
    "street_tset", "name_num_diff",
    # v3: token rarity within the entity's candidate set (Linacre)
    "cs_addr_rare_overlap", "cs_addr_unique_c", "cs_name_rare_overlap", "cs_name_unique_c", "cs_disc_match",
    # v4 (E1): S1-side ambiguity: how many S1 entities share this name / address (over the full S1 table)
    "s1_name_cnt_q", "s1_name_cnt_c", "pool_name_cnt_q", "s1_addr_cnt_q", "s1_addr_cnt_c", "pool_addr_cnt_q",
    "name_unique_pair",
]
_LEAD = re.compile(r"\d+")


def _num_feats(hq, hc, aq, ac, nq, nc):
    """numeric-aware comparison: a digit typo (1604 vs 604) vs a different building (1315 vs 1336)"""
    from rapidfuzz.distance import Levenshtein
    n = len(hq)
    out = np.full((n, 9), -1.0, np.float32)
    for i in range(n):
        a, b = hq[i], hc[i]
        if a and b:
            out[i, 0] = Levenshtein.distance(a, b)
            out[i, 1] = 1.0 if (a in b or b in a) else 0.0
            ma, mb = _LEAD.match(a), _LEAD.match(b)
            if ma and mb:
                ia, ib = int(ma.group(0)[:9]), int(mb.group(0)[:9])
                out[i, 2] = abs(ia - ib) / max(ia, ib, 1)
        X, Y = aq[i].split(), ac[i].split()
        SX = {t for t in X if t[0].isdigit()}
        SY = {t for t in Y if t[0].isdigit()}
        if SX and SY:
            out[i, 3] = len(SX & SY)
            out[i, 4] = len(SY - SX)
            out[i, 5] = len(SX - SY)
            out[i, 6] = max((min(Levenshtein.distance(t, u) for u in SX) for t in SY - SX), default=0)
        tx = " ".join(t for t in X if not t[0].isdigit())
        ty = " ".join(t for t in Y if not t[0].isdigit())
        if tx and ty:
            out[i, 7] = fuzz.token_set_ratio(tx, ty)
        dq = {t for t in nq[i].split() if any(ch.isdigit() for ch in t)}
        dc = {t for t in nc[i].split() if any(ch.isdigit() for ch in t)}
        if dq or dc:
            out[i, 8] = len(dq ^ dc)
    return out


def _candset_feats(bounds, aq, ac, nq, nc):
    """token rarity inside each entity's candidate set: what makes this candidate different from its siblings"""
    out = np.zeros((len(ac), 5), np.float32)
    for a, b in bounds:
        A = [set(x.split()) for x in ac[a:b]]
        Nn = [set(x.split()) for x in nc[a:b]]
        dfa = Counter(t for s in A for t in s)
        dfn = Counter(t for s in Nn for t in s)
        Q, QN, m = set(aq[a].split()), set(nq[a].split()), b - a
        disc = {t for t in Q if 0 < dfa.get(t, 0) < m}
        for j in range(m):
            sa, sn = A[j], Nn[j]
            out[a + j, 0] = sum(1.0 / dfa[t] for t in Q & sa) / max(len(Q), 1)
            out[a + j, 1] = sum(1 for t in sa - Q if dfa[t] == 1)
            out[a + j, 2] = sum(1.0 / dfn[t] for t in QN & sn) / max(len(QN), 1)
            out[a + j, 3] = sum(1 for t in sn - QN if dfn[t] == 1)
            out[a + j, 4] = len(disc & sa) / len(disc) if disc else -1.0
    return out


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
    hq, hc = rec.take("hn", q), rec.take("hn", c)
    NF = np.vstack(Parallel(n_jobs=N_JOBS)(
        delayed(_num_feats)(hq[i:i + step], hc[i:i + step], aq[i:i + step], ac[i:i + step], cq[i:i + step],
                            cc[i:i + step]) for i in range(0, len(q), step)))
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    jobs = []
    for part in np.array_split(np.arange(len(starts)), N_JOBS):
        if len(part):
            lo, hi = starts[part[0]], ends[part[-1]]
            jobs.append(delayed(_candset_feats)([(x - lo, y - lo) for x, y in zip(starts[part], ends[part])],
                                                aq[lo:hi], ac[lo:hi], cq[lo:hi], cc[lo:hi]))
    CS = np.vstack(Parallel(n_jobs=N_JOBS)(jobs))
    for j, name in enumerate(FEATURES[-21:-7]):
        F[name] = (NF if j < 9 else CS)[:, j if j < 9 else j - 9]
    F["s1_name_cnt_q"], F["s1_name_cnt_c"] = np.log1p(rec.s1_name_cnt[q]), np.log1p(rec.s1_name_cnt[c])
    F["pool_name_cnt_q"] = np.log1p(rec.pool_name_cnt[q])
    F["s1_addr_cnt_q"], F["s1_addr_cnt_c"] = np.log1p(rec.s1_addr_cnt[q]), np.log1p(rec.s1_addr_cnt[c])
    F["pool_addr_cnt_q"] = np.log1p(rec.pool_addr_cnt[q])
    F["name_unique_pair"] = ((rec.s1_name_cnt[q] == 1) & (rec.s1_name_cnt[c] <= 1)).astype(np.float32)
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
    cache = PREV / f"{name}_pairs.npz"
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


_prev_cache = PREV
PREV = WORK                                   # v10: rebuild VALIDATION pairs with the new cut
va_q, va_c, va_X, va_y = build_pairs("val", VAL_Q)
PREV = _prev_cache                            # training pairs: v4 cache (models are not retrained)
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


m1 = xgb.Booster(model_file=str(PREV / "m1.json"))      # v5: the v4 stage-1 matcher
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


XS_FEATURES = ["xs_name", "xs_addr", "xs_hn_eq", "ss_name", "ss_addr", "ss_hn_eq"]


def cross_source_feats(rec, q, c, p):
    """v3 (Saeedi et al. / TransClean): agreement of each candidate with the entity's best candidate from the
    other source, and with its best rival from the same source"""
    s = rec.src[c]
    df = pd.DataFrame({"q": q, "c": c, "s": s, "p": p}).sort_values(["q", "s", "p"], ascending=[True, True, False])
    df["rk"] = df.groupby(["q", "s"]).cumcount()
    best = df[df.rk == 0].set_index(["q", "s"])["c"]
    second = df[df.rk == 1].set_index(["q", "s"])["c"]
    b_other = best.reindex(pd.MultiIndex.from_arrays([q, np.where(s == 2, 3, 2)])).to_numpy()
    b_best_same = best.reindex(pd.MultiIndex.from_arrays([q, s])).to_numpy()
    b_second_same = second.reindex(pd.MultiIndex.from_arrays([q, s])).to_numpy()
    b_same = np.where(b_best_same == c, b_second_same, b_best_same)
    out = np.full((len(q), 6), -1.0, np.float32)
    for j, bref in enumerate((b_other, b_same)):
        idx = np.flatnonzero(~np.isnan(bref))
        bi = bref[idx].astype(np.int64)
        for k, col in enumerate(("core", "addr")):
            out[idx, 3 * j + k] = process.cpdist(rec.take(col, c[idx]), rec.take(col, bi),
                                                 scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        h1 = np.array(rec.take("hn", c[idx]), dtype=object)
        h2 = np.array(rec.take("hn", bi), dtype=object)
        out[idx, 3 * j + 2] = np.where((h1 == "") | (h2 == ""), -1.0, (h1 == h2).astype(np.float32))
    return out


def group_ctx(rec, q, c, p):
    return np.hstack([group_features(q, rec.src[c], p), cross_source_feats(rec, q, c, p)])


ALL_FEATURES = FEATURES + G_FEATURES + XS_FEATURES
Gva = group_ctx(train, va_q, va_c, p1_va)
m2 = xgb.Booster(model_file=str(PREV / "m2.json"))      # v5: the v4 group-context model
p2_va = predict(m2, np.hstack([va_X, Gva]), ALL_FEATURES)
log(f"group-context model best iteration {m2.best_iteration}")
gc.collect()

# %% [markdown]
# ## 11b. Ditto-style cross-encoder re-ranker (v3)
# Records are serialized as `[COL] name [VAL] … [COL] addr [VAL] …` with the house number wrapped in
# `[NUM] … [/NUM]` (Ditto's span typing), and an Apache-2.0 MiniLM cross-encoder is fine-tuned on the top-8
# candidates of 150k training entities. It re-scores only the uncertain band of GBDT probabilities, and a
# logistic blend of the two logits (fit on validation half A, reported on half B) gives the final probability.

# %%
import math  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

CE_DIR = ROOT / "work" / "v9" / "mce_model"   # v9: multilingual cross-encoder


def serialize(rec, idx):
    names, addrs, hns = rec.take("name", idx), rec.take("addr", idx), rec.take("hn", idx)
    out = []
    for n, a, h in zip(names, addrs, hns):
        if h:
            a = (" " + a + " ").replace(f" {h} ", f" [NUM] {h} [/NUM] ", 1).strip()
        out.append(f"[COL] name [VAL] {n} [COL] addr [VAL] {a if a else 'none'}")
    return out


def ce_batches(rec, qi, ci, bs):
    for a in range(0, len(qi), bs):
        enc = ce_tok(serialize(rec, qi[a:a + bs]), serialize(rec, ci[a:a + bs]), truncation=True,
                     max_length=CE_MAX_LEN, padding=True, return_tensors="pt")
        yield a, {k: v.to(DEVICE) for k, v in enc.items()}


if (CE_DIR / "config.json").exists():
    ce_tok = AutoTokenizer.from_pretrained(CE_DIR)
    ce_model = AutoModelForSequenceClassification.from_pretrained(CE_DIR).to(DEVICE)
    log(f"cross-encoder loaded from {CE_DIR}")
else:
    ce_tok = AutoTokenizer.from_pretrained(CE_MODEL)
    ce_tok.add_special_tokens({"additional_special_tokens": ["[COL]", "[VAL]", "[NUM]", "[/NUM]"]})
    ce_model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL, num_labels=1)
    ce_model.resize_token_embeddings(len(ce_tok))
    ce_model.to(DEVICE).train()
    ents = np.random.default_rng(3).choice(np.unique(tr_q), CE_TRAIN_S1, replace=False)
    sel = np.flatnonzero(np.isin(tr_q, ents) & (tr_X[:, FEATURES.index("pf_rank")] < 8))
    sel = np.random.default_rng(4).permutation(sel)
    yt = torch.from_numpy(tr_y[sel].astype(np.float32))
    opt = torch.optim.AdamW(ce_model.parameters(), lr=3e-5, weight_decay=0.01)
    steps = math.ceil(len(sel) / 64)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    for step, (a, enc) in enumerate(ce_batches(train, tr_q[sel].astype(np.int64), tr_c[sel].astype(np.int64), 64)):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = ce_model(**enc).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), yt[a:a + 64].to(DEVICE))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(ce_model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 2000 == 0:
            log(f"  cross-encoder step {step:,}/{steps:,} loss {loss.item():.4f}")
    ce_model.save_pretrained(CE_DIR)
    ce_tok.save_pretrained(CE_DIR)
ce_model.eval()
del tr_X
gc.collect()


@torch.no_grad()
def ce_score(rec, q, c):
    out = np.zeros(len(q), np.float32)
    for a, enc in ce_batches(rec, q.astype(np.int64), c.astype(np.int64), 256):
        with torch.autocast("cuda", dtype=torch.float16):
            out[a:a + 256] = torch.sigmoid(ce_model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


def _logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


va_band = np.flatnonzero((p2_va >= CE_BAND[0]) & (p2_va <= CE_BAND[1]))
pce_va = ce_score(train, va_q[va_band], va_c[va_band])
log(f"cross-encoder scored {len(va_band):,} validation band pairs ({len(va_band) / len(p2_va):.1%})")
_half = np.random.default_rng(7).random(len(VAL_Q)) < 0.5            # validation halves A / B
VAL_A, VAL_B = VAL_Q[_half], VAL_Q[~_half]
_inA = np.isin(va_q[va_band], VAL_A)
_Xb = np.column_stack([_logit(p2_va[va_band]), _logit(pce_va)])
blend_A = LogisticRegression(C=10.0).fit(_Xb[_inA], va_y[va_band][_inA])
pb_A = p2_va.copy()
pb_A[va_band] = blend_A.predict_proba(_Xb)[:, 1]
blend = LogisticRegression(C=10.0).fit(_Xb, va_y[va_band])            # refit on all validation for test time
log(f"blend weights (GBDT, cross-encoder) {blend.coef_.round(3).tolist()} bias {blend.intercept_.round(3).tolist()}")
pb_va = p2_va.copy()
pb_va[va_band] = blend.predict_proba(_Xb)[:, 1]

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
for nm, prob in (("stage-1 matcher", p1_va), ("group-context model", p2_va), ("blend", pb_va)):
    t_glob, f_glob, per, f_per = tune(prob)
    results[nm] = (t_glob, f_glob, per, f_per)
    log(f"{nm}: global threshold {t_glob} → macro F0.5 {f_glob:.4f} | per-country {per} → {f_per:.4f}")
BEST_MODEL = max(results, key=lambda k: max(results[k][1], results[k][3]))
# honest estimate: blend + threshold tuned on half A, scored on half B
_mA = np.isin(va_q, VAL_A)
_tA = {t: macro_f05(VAL_A, *resolve(va_q[_mA], va_c[_mA], train.src[va_c[_mA]], pb_A[_mA], np.full(_mA.sum(), t)))
       for t in np.round(np.arange(0.5, 0.91, 0.025), 3)}
_tb = max(_tA, key=_tA.get)
_mB = ~_mA
log(f"HOLD-OUT CHECK: blend tuned on half A (t={_tb}) -> half B macro F0.5 = "
    f"{macro_f05(VAL_B, *resolve(va_q[_mB], va_c[_mB], train.src[va_c[_mB]], pb_A[_mB], np.full(_mB.sum(), _tb))):.4f}")
t_glob, f_glob, per, f_per = results[BEST_MODEL]
USE_PER_COUNTRY = f_per > f_glob
THRESH = per if USE_PER_COUNTRY else {}
T_DEFAULT = t_glob
best_prob = {"stage-1 matcher": p1_va, "group-context model": p2_va, "blend": pb_va}[BEST_MODEL]
thr_va = np.array([THRESH.get(x, T_DEFAULT) for x in va_ctry], dtype=np.float32)
fq, fc = resolve(va_q, va_c, va_src, best_prob, thr_va)
f_ent = f05_per_entity(VAL_Q, fq, fc)
log(f"VALIDATION macro F0.5 = {f_ent.mean():.4f} using {BEST_MODEL}, "
    f"{'per-country' if USE_PER_COUNTRY else 'global'} thresholds {THRESH or T_DEFAULT} (v2 0.9721, baseline 0.8871)")
ctry_val = np.asarray(train.countries)[train.country_codes[VAL_Q]]
nt_val = np.bincount(np.searchsorted(VAL_Q, GT_Q[np.isin(GT_Q, VAL_Q)]), minlength=len(VAL_Q))
print(pd.DataFrame({"country": ctry_val, "f": f_ent}).groupby("country").f.agg(["mean", "size"]).round(4))
print(pd.DataFrame({"n_true": pd.cut(nt_val, [-1, 0, 1, 3, 5, 20], labels=["singleton", "1", "2-3", "4-5", "6+"]),
                    "f": f_ent}).groupby("n_true", observed=True).f.agg(["mean", "size"]).round(4))
pd.DataFrame({k: [v[0], v[1], str(v[2]), v[3]] for k, v in results.items()},
             index=["t_global", "f_global", "t_per_country", "f_per_country"]).to_csv(WORK / "val_scores.csv")
del va_X, Gva
gc.collect()

# %%
log("v10 validation-only experiment done (no test inference)")
