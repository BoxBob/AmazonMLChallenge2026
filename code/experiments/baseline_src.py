# %% [markdown]
# # Business Entity Resolution — Baseline (GPU)
#
# End-to-end baseline for the Amazon ML Challenge 2026 entity-resolution task.
#
# 1. **Load + normalize** all three sources (unicode folding / transliteration, legal-suffix
#    and address-abbreviation canonicalisation, postcode + house-number extraction).
# 2. **Embed** names and addresses: char n-gram TF-IDF → truncated SVD (LSA) computed on the **GPU**.
# 3. **Blocking** within each country: GPU top-k cosine search on name and address embeddings,
#    plus exact keys (squashed name, postcode + first name token).
# 4. **Pair features** (rapidfuzz string similarities, embedding cosines, postcode / house-number
#    agreement, rank features) → **XGBoost on the GPU** (`device="cuda"`).
# 5. **Decision**: one S1 owner per S2/S3 record, per-source caps, threshold tuned for
#    macro F0.5 on a held-out validation split of Source-1 entities.
# 6. **Test inference** → `output/candidate_pairs.tsv` + `output/matching_results.tsv`, then run the validator.
#
# Everything (caches, embeddings, temp files, outputs) is written to the **D:** drive.

# %%
import os
import tempfile
from pathlib import Path

# project root = folder holding dataset/, utils/ and output/; override with the ER_ROOT env var
ROOT = Path(os.environ.get("ER_ROOT", r"D:\AmazonMLChallenge2026"))
DATA = ROOT / "dataset"
WORK = ROOT / "work" / "baseline"
TMP = ROOT / "work" / "tmp"
OUT = ROOT / "output"
for p in (WORK, TMP, OUT):
    p.mkdir(parents=True, exist_ok=True)

# keep every temp file (joblib memmaps, tempfile, ...) off the C: drive
os.environ["TEMP"] = os.environ["TMP"] = os.environ["JOBLIB_TEMP_FOLDER"] = str(TMP)
tempfile.tempdir = str(TMP)

import csv
import gc
import re
import subprocess
import sys
import time

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
N_TRAIN_S1 = 300_000        # S1 entities used to train the pair classifier
N_VAL_S1 = 100_000          # disjoint S1 entities used for threshold tuning / scoring
EMB_DIM = 192               # SVD dimensions of the name / address embeddings
SVD_SAMPLE = 250_000        # texts per field used to fit TF-IDF + SVD
TFIDF_MAX_FEATURES = 2**18
K_NAME = 10                 # top-k neighbours per (S1, source) by name embedding
K_ADDR = 10                 # top-k neighbours per (S1, source) by address embedding
EXACT_BLOCK_CAP = 50        # skip exact-key blocks larger than this (too generic)
MATCH_CAP = {2: 5, 3: 6}    # max matches per S1 per source (max seen in train ground truth)
N_JOBS = 6                  # CPU workers for normalization / TF-IDF
QUERY_BATCH = 1024          # GPU kNN: queries per matmul
POOL_CHUNK = 400_000        # GPU kNN: pool rows resident on the GPU at once
FEATURE_CHUNK = 60_000      # S1 queries per feature/predict chunk (~2M pairs)
T0 = time.time()

def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)

# %% [markdown]
# ## 1. Loading and normalization

# %%
LEGAL_CANON = {
    "corporation": "corp", "incorporated": "inc", "company": "co", "limited": "ltd",
    "private": "pvt", "pte": "pvt", "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp",
    "pllc": "pllc", "ltd": "ltd", "pvt": "pvt", "inc": "inc", "corp": "corp", "co": "co",
    "sarl": "sarl", "sas": "sas", "sa": "sa", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scp": "scp", "selarl": "selarl", "gmbh": "gmbh", "ag": "ag", "bv": "bv",
}
LEGAL_SET = set(LEGAL_CANON.values())
LEADING_LEGAL = {"sci", "sarl", "sas", "sasu", "eurl", "snc", "scp", "selarl", "ste", "societe", "the"}

ADDR_CANON = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd",
    "bd": "blvd", "drive": "dr", "lane": "ln", "place": "pl", "court": "ct", "highway": "hwy",
    "suite": "ste", "apartment": "apt", "floor": "fl", "building": "bldg", "bldg": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "near": "nr", "opposite": "opp",
    "township": "twp", "sector": "sec", "number": "no", "parkway": "pkwy", "circle": "cir",
    "terrace": "ter", "square": "sq", "center": "ctr", "centre": "ctr", "junction": "jn",
    "rue": "r", "chemin": "ch", "allee": "all", "impasse": "imp", "route": "rte",
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
    "tamil nadu": "tn", "maharashtra": "mh", "karnataka": "ka", "uttar pradesh": "up",
    "west bengal": "wb", "gujarat": "gj", "rajasthan": "rj", "kerala": "kl", "andhra pradesh": "ap",
    "telangana": "tg", "madhya pradesh": "mp", "haryana": "hr", "punjab": "pb", "bihar": "br",
    "odisha": "od", "orissa": "od", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk",
    "himachal pradesh": "hp", "assam": "as", "goa": "ga", "new delhi": "delhi",
}
# multi-word replacements, longest first (applied on " token token " strings)
MULTI_REPL = sorted(((f" {k} ", f" {v} ") for k, v in {**US_STATES, **IN_STATES}.items() if " " in k),
                    key=lambda kv: -len(kv[0]))
ADDR_SINGLE = {**ADDR_CANON, **{k: v for k, v in {**US_STATES, **IN_STATES}.items() if " " not in k}}

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_WWW = re.compile(r"\bwww\.")
_TLD = re.compile(r"\.(com|net|org|biz|info|io|co\.in|in|us|fr)\b")
_POSTCODE = re.compile(r"(?<!\d)\d{5,6}(?!\d)")


def _fold(s):
    if not s.isascii():
        s = anyascii(s)
    return s.lower()


def norm_name(s):
    """-> (full normalized name, core name without legal forms, legal-suffix string)"""
    s = _fold(s).replace("&", " and ").replace("'", "")
    s = _TLD.sub(" ", _WWW.sub(" ", s)).replace(".", "")
    toks = [LEGAL_CANON.get(t, t) for t in _NON_ALNUM.sub(" ", s).split()]
    core = list(toks)
    suffix = []
    while core and core[-1] in LEGAL_SET:
        suffix.insert(0, core.pop())
    while core and core[0] in LEADING_LEGAL:
        suffix.insert(0, core.pop(0))
    if not core:  # name consisted only of legal words
        core = toks
    return " ".join(toks), " ".join(core), " ".join(suffix)


def norm_addr(s):
    """-> (normalized address, postcode, house number)"""
    pcs = _POSTCODE.findall(s)
    pc = pcs[-1] if pcs else ""
    a = " " + _NON_ALNUM.sub(" ", _fold(s).replace(".", "").replace("'", "")) + " "
    for full, ab in MULTI_REPL:
        if full in a:
            a = a.replace(full, ab)
    toks = [ADDR_SINGLE.get(t, t) for t in a.split()]
    hn = next((t for t in toks if t[0].isdigit()), "")
    if hn == pc:
        hn = ""
    return " ".join(toks), pc, hn


def _normalize_chunk(names, addrs):
    nn = [norm_name(x) for x in names]
    aa = [norm_addr(x) for x in addrs]
    core = [x[1] for x in nn]
    return pa.table({
        "name": [x[0] for x in nn],
        "core": core,
        "suffix": [x[2] for x in nn],
        "squash": [c.replace(" ", "") for c in core],
        "first": [c.split(" ", 1)[0] for c in core],
        "addr": [x[0] for x in aa],
        "pc": [x[1] for x in aa],
        "hn": [x[2] for x in aa],
    })


print(norm_name("Maure Williams Colombier Inc"), norm_name("maurewilliamscolombier.com"))
print(norm_name("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"), norm_name("SCI Ptit Àmicale"))
print(norm_addr("85 Wanye Avenue, Ticonderoga Townshiip, New York"), norm_addr("6(29), C.I.T. Colony, Chennai, Tamil Nadu 600004"))

# %%
def read_source(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, engine="c")


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
            tbl = pa.concat_tables(parts).combine_chunks()
            pq.write_table(tbl, cache)
        self.split = split
        self.tbl = tbl
        self.n = tbl.num_rows
        self.src = tbl["src"].to_numpy()
        self.country_codes, self.countries = pd.factorize(tbl["country"].to_pandas())
        self.id_index = pd.Index(tbl["entity_id"].to_pandas())
        self.core_blank = self._is_blank("core")
        self.addr_blank = self._is_blank("addr")

    def _is_blank(self, name):
        return pac.equal(pac.utf8_length(self.tbl[name]), 0).to_numpy(zero_copy_only=False)

    def col(self, name):
        return self.tbl[name]

    def take(self, name, idx):
        return self.tbl[name].take(pa.array(idx)).to_pylist()


train = Records("train")
log(f"train records: {train.n:,}  countries={list(train.countries)}")
test = Records("test")
log(f"test records: {test.n:,}  countries={list(test.countries)}")

# %% [markdown]
# ## 2. Name / address embeddings (TF-IDF → SVD on the GPU)
#
# The TF-IDF vocabulary and SVD basis are fitted on a sample drawn from **both** splits (text only,
# no labels) so that French n-grams, which never occur in training, are represented.

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
        vec.stop_words_ = None  # large set of dropped n-grams; would be pickled to every worker
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
# ## 3. Blocking / candidate generation
#
# Within each country and each pool source (S2, S3): GPU top-k cosine search by name and by address,
# plus exact-key joins. The union is the candidate set that the model scores (= `candidate_pairs.tsv`).

# %%
@torch.no_grad()
def gpu_topk(q_emb, pool_emb, pool_idx, k):
    """For each query row, the k best pool rows by cosine. Returns (global idx, score)."""
    nq = len(q_emb)
    Q = torch.from_numpy(np.ascontiguousarray(q_emb)).to(DEVICE)
    best_s = torch.full((nq, k), -2.0, device=DEVICE, dtype=torch.float16)
    best_i = torch.full((nq, k), -1, device=DEVICE, dtype=torch.int64)
    for c0 in range(0, len(pool_idx), POOL_CHUNK):
        pidx = pool_idx[c0:c0 + POOL_CHUNK]
        P = torch.from_numpy(np.ascontiguousarray(pool_emb[pidx])).to(DEVICE)
        pidx_g = torch.from_numpy(pidx).to(DEVICE)
        kk = min(k, len(pidx))
        for q0 in range(0, nq, QUERY_BATCH):
            q1 = min(q0 + QUERY_BATCH, nq)
            s, i = (Q[q0:q1] @ P.T).topk(kk, dim=1)
            s = torch.cat([best_s[q0:q1], s], 1)
            i = torch.cat([best_i[q0:q1], pidx_g[i]], 1)
            top_s, pos = s.topk(k, dim=1)
            best_s[q0:q1] = top_s
            best_i[q0:q1] = torch.gather(i, 1, pos)
        del P, pidx_g
    out = best_i.cpu().numpy(), best_s.float().cpu().numpy()
    del Q, best_s, best_i
    torch.cuda.empty_cache()
    return out


def exact_key_pairs(rec, qidx, keys_fn):
    """Join S1 queries to S2/S3 pool records sharing an exact key (same country)."""
    full = keys_fn(rec).to_numpy(zero_copy_only=False)   # "" = no key
    pool = np.flatnonzero((rec.src != 1) & (full != ""))
    pool_df = pd.DataFrame({"k": full[pool], "c": pool})
    sizes = pool_df.groupby("k")["c"].transform("size")
    pool_df = pool_df[sizes <= EXACT_BLOCK_CAP]
    q_df = pd.DataFrame({"k": full[qidx], "q": qidx})
    q_df = q_df[q_df["k"] != ""]
    return q_df.merge(pool_df, on="k")[["q", "c"]]


def key_squash(rec):
    sq = rec.col("squash")
    key = pac.binary_join_element_wise(rec.col("country"), sq, "|")
    return pac.if_else(pac.greater_equal(pac.utf8_length(sq), 4), key, "")


def key_pc_first(rec):
    pc, first = rec.col("pc"), rec.col("first")
    key = pac.binary_join_element_wise(rec.col("country"), pc, first, "|")
    ok = pac.and_(pac.not_equal(pc, ""), pac.not_equal(first, ""))
    return pac.if_else(ok, key, "")


def generate_candidates(rec, qidx):
    qidx = np.sort(qidx)
    parts = []
    for ci in np.unique(rec.country_codes[qidx]):
        q_c = qidx[rec.country_codes[qidx] == ci]
        for s in (2, 3):
            in_pool = (rec.src == s) & (rec.country_codes == ci)
            for field, k, blank in (("core", K_NAME, rec.core_blank), ("addr", K_ADDR, rec.addr_blank)):
                pool = np.flatnonzero(in_pool & ~blank)
                q = q_c[~blank[q_c]]
                if len(pool) == 0 or len(q) == 0:
                    continue
                emb = EMB[(rec.split, field)]
                idx, _ = gpu_topk(emb[q], emb, pool, min(k, len(pool)))
                kk = idx.shape[1]
                df = pd.DataFrame({"q": np.repeat(q, kk), "c": idx.ravel(),
                                   f"rank_{field}": np.tile(np.arange(kk, dtype=np.float32), len(q))})
                parts.append(df[df["c"] >= 0])
            log(f"  {rec.split} country={rec.countries[ci]} S{s}: kNN done for {len(q_c):,} queries")
    ex1 = exact_key_pairs(rec, qidx, key_squash).assign(exact_squash=1.0)
    ex2 = exact_key_pairs(rec, qidx, key_pc_first).assign(exact_pc_first=1.0)
    parts += [ex1, ex2]
    cands = pd.concat(parts, ignore_index=True)
    agg = {"rank_core": "min", "rank_addr": "min", "exact_squash": "max", "exact_pc_first": "max"}
    cands = cands.groupby(["q", "c"], sort=True, as_index=False).agg(agg)
    cands["rank_core"] = cands["rank_core"].fillna(K_NAME)
    cands["rank_addr"] = cands["rank_addr"].fillna(K_ADDR)
    cands[["exact_squash", "exact_pc_first"]] = cands[["exact_squash", "exact_pc_first"]].fillna(0.0)
    log(f"{rec.split}: {len(cands):,} candidate pairs for {len(qidx):,} S1 queries "
        f"({len(cands) / len(qidx):.1f} per query)")
    return cands

# %% [markdown]
# ## 4. Ground truth, train / validation split

# %%
gt = pd.read_csv(DATA / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                 keep_default_na=False, quoting=csv.QUOTE_NONE)
gt_q_all = train.id_index.get_indexer(gt["source1_entity_id"])
m = gt["matched_entity_ids"].str.split(",").explode()
m = m[m != ""]
GT_Q = train.id_index.get_indexer(gt["source1_entity_id"].loc[m.index])
GT_C = train.id_index.get_indexer(m.values)
assert (GT_Q >= 0).all() and (GT_C >= 0).all()
GT_KEYS = np.sort(GT_Q.astype(np.int64) * train.n + GT_C)
log(f"ground truth: {len(gt_q_all):,} S1, {len(GT_KEYS):,} true pairs")

rng = np.random.default_rng(SEED)
s1_all = np.flatnonzero(train.src == 1)
perm = rng.permutation(s1_all)
TRAIN_Q = np.sort(perm[:N_TRAIN_S1])
VAL_Q = np.sort(perm[N_TRAIN_S1:N_TRAIN_S1 + N_VAL_S1])


def pair_labels(q, c):
    keys = q.astype(np.int64) * train.n + c
    pos = np.searchsorted(GT_KEYS, keys)
    pos = np.minimum(pos, len(GT_KEYS) - 1)
    return (GT_KEYS[pos] == keys).astype(np.int8)


def macro_f05(qset, pq, pc):
    """Official metric: per-S1 F0.5 averaged over every S1 in qset (singletons included)."""
    qset = np.sort(qset)
    in_q = np.isin(GT_Q, qset)
    tq, tc = GT_Q[in_q], GT_C[in_q]
    n_true = np.bincount(np.searchsorted(qset, tq), minlength=len(qset))
    n_pred = np.bincount(np.searchsorted(qset, pq), minlength=len(qset))
    hit = pair_labels(pq, pc).astype(bool)
    tp = np.bincount(np.searchsorted(qset, pq[hit]), minlength=len(qset))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
        r = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)
    return f.mean()

# %% [markdown]
# ## 5. Pair features

# %%
FEATURES = [
    "cos_core", "cos_addr", "rank_core", "rank_addr", "exact_squash", "exact_pc_first",
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw_squash", "n_full_tset", "squash_contains",
    "first_eq", "suffix_eq", "len_core_q", "len_core_c",
    "a_ratio", "a_tset", "a_tsort", "pc_eq", "hn_eq", "addr_blank_q", "addr_blank_c",
    "src_c", "n_cands", "rel_cos_core", "rel_cos_addr", "rel_n_tset", "rel_a_tset",
]


def _eq_or_missing(a, b):
    a = np.array(a, dtype=object)
    b = np.array(b, dtype=object)
    miss = (a == "") | (b == "")
    return np.where(miss, -1.0, (a == b).astype(np.float32))


def _row_cos(emb, q, c):
    out = np.empty(len(q), dtype=np.float32)
    for i in range(0, len(q), 1_000_000):
        a = torch.from_numpy(np.ascontiguousarray(emb[q[i:i + 1_000_000]])).to(DEVICE).float()
        b = torch.from_numpy(np.ascontiguousarray(emb[c[i:i + 1_000_000]])).to(DEVICE).float()
        out[i:i + 1_000_000] = (a * b).sum(1).cpu().numpy()
    return out


def pair_features(rec, cands):
    q, c = cands["q"].to_numpy(), cands["c"].to_numpy()
    F = pd.DataFrame({k: cands[k].to_numpy(np.float32)
                      for k in ("rank_core", "rank_addr", "exact_squash", "exact_pc_first")})
    F["cos_core"] = _row_cos(EMB[(rec.split, "core")], q, c)
    F["cos_addr"] = _row_cos(EMB[(rec.split, "addr")], q, c)
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
    F["first_eq"] = _eq_or_missing(rec.take("first", q), rec.take("first", c))
    F["suffix_eq"] = _eq_or_missing(rec.take("suffix", q), rec.take("suffix", c))
    F["pc_eq"] = _eq_or_missing(rec.take("pc", q), rec.take("pc", c))
    F["hn_eq"] = _eq_or_missing(rec.take("hn", q), rec.take("hn", c))
    F["len_core_q"] = np.fromiter(map(len, cq), dtype=np.float32, count=len(q))
    F["len_core_c"] = np.fromiter(map(len, cc), dtype=np.float32, count=len(q))
    F["addr_blank_q"] = rec.addr_blank[q].astype(np.float32)
    F["addr_blank_c"] = rec.addr_blank[c].astype(np.float32)
    F["src_c"] = rec.src[c].astype(np.float32)
    g = pd.Series(q)
    F["n_cands"] = g.map(g.value_counts()).to_numpy(np.float32)
    for col in ("cos_core", "cos_addr", "n_tset", "a_tset"):
        F[f"rel_{col}"] = F[col] - F.groupby(q)[col].transform("max")
    return F[FEATURES]


def chunked_features(rec, cands):
    """Features for a (q-sorted) candidate frame, in chunks of whole S1 queries."""
    uq = np.unique(cands["q"].to_numpy())
    bounds = np.searchsorted(cands["q"].to_numpy(), uq[::FEATURE_CHUNK])
    bounds = list(bounds) + [len(cands)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        out.append(pair_features(rec, cands.iloc[a:b]))
        log(f"  features {b:,}/{len(cands):,}")
    return pd.concat(out, ignore_index=True)

# %% [markdown]
# ## 6. Build training / validation pairs

# %%
def build_pairs(name, qidx):
    fcache, ccache = WORK / f"{name}_features.parquet", WORK / f"{name}_cands.parquet"
    if fcache.exists() and ccache.exists():
        return pd.read_parquet(ccache), pd.read_parquet(fcache)
    cands = generate_candidates(train, qidx)
    feats = chunked_features(train, cands)
    cands["label"] = pair_labels(cands["q"].to_numpy(), cands["c"].to_numpy())
    cands.to_parquet(ccache)
    feats.to_parquet(fcache)
    return cands, feats


tr_cands, tr_X = build_pairs("trainpairs", TRAIN_Q)
va_cands, va_X = build_pairs("valpairs", VAL_Q)

for name, qset, cands in (("train", TRAIN_Q, tr_cands), ("val", VAL_Q, va_cands)):
    n_true = np.isin(GT_Q, qset).sum()
    oracle = macro_f05(qset, cands["q"].to_numpy()[cands["label"] == 1], cands["c"].to_numpy()[cands["label"] == 1])
    log(f"{name}: blocking pair recall {cands['label'].sum() / n_true:.4f} "
        f"({cands['label'].sum():,}/{n_true:,}), {len(cands) / len(qset):.1f} cands/S1, "
        f"oracle macro-F0.5 {oracle:.4f}")

# %% [markdown]
# ## 7. XGBoost pair classifier (GPU)

# %%
XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric=["logloss", "aucpr"], device="cuda", tree_method="hist",
    max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    max_bin=256, seed=SEED,
)
dtrain = xgb.QuantileDMatrix(tr_X.to_numpy(np.float32), tr_cands["label"].to_numpy(), feature_names=FEATURES)
dval = xgb.QuantileDMatrix(va_X.to_numpy(np.float32), va_cands["label"].to_numpy(), ref=dtrain,
                           feature_names=FEATURES)
booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=2000, evals=[(dval, "val")],
                    early_stopping_rounds=50, verbose_eval=100)
booster.save_model(WORK / "xgb_baseline.json")
log(f"best iteration {booster.best_iteration}")
imp = pd.Series(booster.get_score(importance_type="gain"))
print(imp.sort_values(ascending=False).round(1).to_string())
del dtrain
gc.collect()

# %% [markdown]
# ## 8. Decision rule + threshold tuning on validation (macro F0.5)

# %%
def resolve(q, c, src_c, prob, threshold):
    """Keep pairs >= threshold, each S2/S3 record assigned to its best S1 only, per-source caps."""
    df = pd.DataFrame({"q": q, "c": c, "s": src_c, "p": prob})
    df = df[df["p"] >= threshold]
    df = df.sort_values("p", ascending=False).drop_duplicates("c")
    df["r"] = df.groupby(["q", "s"]).cumcount()
    cap = df["s"].map(MATCH_CAP).fillna(max(MATCH_CAP.values()))
    df = df[df["r"] < cap]
    return df["q"].to_numpy(), df["c"].to_numpy()


def predict(X):
    return booster.predict(xgb.DMatrix(X.to_numpy(np.float32), feature_names=FEATURES),
                           iteration_range=(0, booster.best_iteration + 1))


va_prob = predict(va_X)
va_src = train.src[va_cands["c"].to_numpy()]
scores = {}
for t in np.round(np.arange(0.20, 0.96, 0.025), 3):
    pq, pc = resolve(va_cands["q"].to_numpy(), va_cands["c"].to_numpy(), va_src, va_prob, t)
    scores[t] = macro_f05(VAL_Q, pq, pc)
BEST_T = max(scores, key=scores.get)
print(pd.Series(scores).round(4).to_string())
log(f"VALIDATION macro F0.5 = {scores[BEST_T]:.4f} at threshold {BEST_T}")
pd.Series(scores).to_csv(WORK / "val_threshold_scores.csv")

# %% [markdown]
# ## 9. Test inference → `output/candidate_pairs.tsv`, `output/matching_results.tsv`

# %%
del tr_cands, tr_X
gc.collect()

test_q = np.flatnonzero(test.src == 1)
te_cache = WORK / "test_cands.parquet"
if te_cache.exists():
    te_cands = pd.read_parquet(te_cache)
else:
    te_cands = generate_candidates(test, test_q)
    te_cands.to_parquet(te_cache)

# score in chunks of whole S1 queries, keep only pairs that could pass the threshold
q_arr = te_cands["q"].to_numpy()
uq = np.unique(q_arr)
bounds = list(np.searchsorted(q_arr, uq[::FEATURE_CHUNK])) + [len(te_cands)]
keep = []
for a, b in zip(bounds[:-1], bounds[1:]):
    chunk = te_cands.iloc[a:b]
    X = pair_features(test, chunk)
    p = predict(X)
    sel = p >= BEST_T
    keep.append(pd.DataFrame({"q": chunk["q"].to_numpy()[sel], "c": chunk["c"].to_numpy()[sel], "p": p[sel]}))
    log(f"  test scored {b:,}/{len(te_cands):,}")
kept = pd.concat(keep, ignore_index=True)
mq, mc = resolve(kept["q"].to_numpy(), kept["c"].to_numpy(), test.src[kept["c"].to_numpy()],
                 kept["p"].to_numpy(), BEST_T)
log(f"test: {len(mq):,} predicted matches, {len(np.unique(mq)):,} of {len(test_q):,} S1 with >=1 match")

# %%
def write_lists(path, header, q_all, pq, pc, rec):
    ids = rec.tbl["entity_id"].to_numpy(zero_copy_only=False)
    df = pd.DataFrame({"q": pq, "id": ids[pc]})
    lists = df.groupby("q")["id"].agg(",".join)
    col = pd.Series(q_all).map(lists).fillna("")
    out = pd.DataFrame({header[0]: ids[q_all], header[1]: col.to_numpy()})
    out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    log(f"wrote {path} ({len(out):,} rows)")


write_lists(OUT / "candidate_pairs.tsv", ("source1_entity_id", "candidate_entity_ids"),
            test_q, te_cands["q"].to_numpy(), te_cands["c"].to_numpy(), test)
write_lists(OUT / "matching_results.tsv", ("source1_entity_id", "matched_entity_ids"),
            test_q, mq, mc, test)

res = subprocess.run([sys.executable, str(ROOT / "utils" / "validate_submission.py"),
                      "--matching", str(OUT / "matching_results.tsv"),
                      "--candidate", str(OUT / "candidate_pairs.tsv"),
                      "--test-dir", str(DATA / "test")], capture_output=True, text=True)
print(res.stdout[-3000:], res.stderr[-3000:])
log("done")
