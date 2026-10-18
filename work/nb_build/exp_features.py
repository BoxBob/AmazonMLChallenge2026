# Experiment B + candidate-set rarity + D2 (cross-source consistency) on the cached v2 pairs.
import ast
import gc
import os
import re
import time
from collections import Counter

os.environ["JOBLIB_TEMP_FOLDER"] = r"D:\AmazonMLChallenge2026\work\tmp"
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

W = r"D:\AmazonMLChallenge2026\work\v2"
W3 = r"D:\AmazonMLChallenge2026\work\v3"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


tbl = pq.read_table(W + r"\train_records.parquet", columns=["core", "name", "addr", "hn", "src"]).combine_chunks()
srcs = tbl["src"].to_numpy()


def take(col, idx):
    return tbl[col].take(pa.array(idx)).to_pylist()


_LEAD = re.compile(r"\d+")
NUM_NAMES = ["hn_lev", "hn_contains", "hn_gap", "num_shared", "num_only_c", "num_only_q", "num_min_lev",
             "street_tset", "name_num_diff"]


def num_feats(hq, hc, aq, ac, nq, nc):
    """numeric-aware comparison: digit typos vs genuinely different numbers"""
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


CS_NAMES = ["cs_addr_rare_overlap", "cs_addr_unique_c", "cs_name_rare_overlap", "cs_name_unique_c", "cs_disc_match"]


def candset_feats(bounds, aq, ac, nq, nc):
    """Linacre-style token rarity within each entity's candidate set"""
    out = np.zeros((len(ac), 5), np.float32)
    for a, b in bounds:
        A = [set(x.split()) for x in ac[a:b]]
        Nn = [set(x.split()) for x in nc[a:b]]
        dfa = Counter(t for s in A for t in s)
        dfn = Counter(t for s in Nn for t in s)
        Q, QN, m = set(aq[a].split()), set(nq[a].split()), b - a
        disc = {t for t in Q if 0 < dfa.get(t, 0) < m}   # S1 tokens that separate the candidates
        for j in range(m):
            sa, sn = A[j], Nn[j]
            out[a + j, 0] = sum(1.0 / dfa[t] for t in Q & sa) / max(len(Q), 1)
            out[a + j, 1] = sum(1 for t in sa - Q if dfa[t] == 1)
            out[a + j, 2] = sum(1.0 / dfn[t] for t in QN & sn) / max(len(QN), 1)
            out[a + j, 3] = sum(1 for t in sn - QN if dfn[t] == 1)
            out[a + j, 4] = len(disc & sa) / len(disc) if disc else -1.0
    return out


def _chunks_by_entity(q, size):
    s0 = 0
    while s0 < len(q):
        s1 = min(s0 + size, len(q))
        while s1 < len(q) and q[s1] == q[s1 - 1]:
            s1 += 1
        yield s0, s1
        s0 = s1


def new_pair_features(q, c, tag):
    cache = f"{W3}\\{tag}_newfeat.npy"
    if os.path.exists(cache):
        return np.load(cache)
    outs = []
    for s0, s1 in _chunks_by_entity(q, 2_000_000):
        qq, cc = q[s0:s1], c[s0:s1]
        hq, hc, aq, ac = take("hn", qq), take("hn", cc), take("addr", qq), take("addr", cc)
        nq, nc = take("core", qq), take("core", cc)
        step = -(-len(qq) // 8)
        NF = np.vstack(Parallel(n_jobs=8)(
            delayed(num_feats)(hq[i:i + step], hc[i:i + step], aq[i:i + step], ac[i:i + step], nq[i:i + step],
                               nc[i:i + step]) for i in range(0, len(qq), step)))
        starts = np.flatnonzero(np.r_[True, qq[1:] != qq[:-1]])
        ends = np.r_[starts[1:], len(qq)]
        jobs = []
        for p in np.array_split(np.arange(len(starts)), 8):
            if len(p) == 0:
                continue
            lo, hi = starts[p[0]], ends[p[-1]]
            jobs.append(delayed(candset_feats)([(a - lo, b - lo) for a, b in zip(starts[p], ends[p])],
                                               aq[lo:hi], ac[lo:hi], nq[lo:hi], nc[lo:hi]))
        CS = np.vstack(Parallel(n_jobs=8)(jobs))
        outs.append(np.hstack([NF, CS]))
        log(f"  {tag}: new features {s1:,}/{len(q):,}")
    F = np.vstack(outs)
    np.save(cache, F)
    return F


XS_NAMES = ["xs_name", "xs_addr", "xs_hn_eq", "ss_name", "ss_addr", "ss_hn_eq"]


def cross_source_feats(q, c, p):
    """compare each candidate with the entity's best other-source candidate and its best rival in the same source"""
    s = srcs[c]
    df = pd.DataFrame({"q": q, "c": c, "s": s, "p": p}).sort_values(["q", "s", "p"], ascending=[True, True, False])
    df["rk"] = df.groupby(["q", "s"]).cumcount()
    best = df[df.rk == 0].set_index(["q", "s"])["c"]
    second = df[df.rk == 1].set_index(["q", "s"])["c"]
    other_s = np.where(s == 2, 3, 2)
    b_other = best.reindex(pd.MultiIndex.from_arrays([q, other_s])).to_numpy()
    b_best_same = best.reindex(pd.MultiIndex.from_arrays([q, s])).to_numpy()
    b_second_same = second.reindex(pd.MultiIndex.from_arrays([q, s])).to_numpy()
    b_same = np.where(b_best_same == c, b_second_same, b_best_same)
    out = np.full((len(q), 6), -1.0, np.float32)
    for j, bref in enumerate((b_other, b_same)):
        idx = np.flatnonzero(~np.isnan(bref))
        bi = bref[idx].astype(np.int64)
        for k, col in enumerate(("core", "addr")):
            out[idx, 3 * j + k] = process.cpdist(take(col, c[idx]), take(col, bi), scorer=fuzz.token_set_ratio,
                                                 workers=-1, dtype=np.float32)
        h1 = np.array(take("hn", c[idx]), dtype=object)
        h2 = np.array(take("hn", bi), dtype=object)
        out[idx, 3 * j + 2] = np.where((h1 == "") | (h2 == ""), -1.0, (h1 == h2).astype(np.float32))
    return out


src = open("improved_src.py", encoding="utf-8").read()
ns = {"np": np, "pd": pd}
exec(re.search(r"^def group_features\(.*?(?=^\S)", src, re.S | re.M).group(0), ns)
ns["G_FEATURES"] = ast.literal_eval(re.search(r"^G_FEATURES = (\[.*?\])", src, re.S | re.M).group(1))
_m1v2 = xgb.Booster()
_m1v2.load_model(W + r"\m1.json")
BASE = _m1v2.feature_names

tz, vz = np.load(W + r"\train_pairs.npz"), np.load(W + r"\val_pairs.npz")
tq, tc, ty = tz["q"].astype(np.int64), tz["c"].astype(np.int64), tz["y"]
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
vNF = new_pair_features(vq, vc, "val")
tNF = new_pair_features(tq, tc, "train")
F1 = BASE + NUM_NAMES + CS_NAMES
tX = np.hstack([tz["X"], tNF])
vX = np.hstack([vz["X"], vNF])
del tNF, vNF
gc.collect()

PAR = dict(objective="binary:logistic", eval_metric=["logloss", "aucpr"], device="cuda", tree_method="hist",
           max_depth=9, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, max_bin=256,
           seed=42)


def fit(X, y, Xv, yv, names):
    d = xgb.QuantileDMatrix(X, y, feature_names=names)
    dv = xgb.QuantileDMatrix(Xv, yv, ref=d, feature_names=names)
    b = xgb.train(PAR, d, 3000, evals=[(dv, "val")], early_stopping_rounds=60, verbose_eval=False)
    del d, dv
    gc.collect()
    return b


def pred(b, X, names):
    return b.predict(xgb.DMatrix(X, feature_names=names), iteration_range=(0, b.best_iteration + 1))


fold = np.random.default_rng(44).integers(0, 2, size=12_527_040)[tq].astype(np.int8)
oof = np.zeros(len(tq), np.float32)
for f in (0, 1):
    mk = fold != f
    b = fit(tX[mk], ty[mk], vX, vy, F1)
    oof[~mk] = pred(b, tX[~mk], F1)
    log(f"fold {f} best iteration {b.best_iteration}")
    del b
m1 = fit(tX, ty, vX, vy, F1)
m1.save_model(W3 + r"\exp_m1.json")
vp1 = pred(m1, vX, F1)
log(f"m1 best iteration {m1.best_iteration}")
print(pd.Series(m1.get_score(importance_type="gain")).reindex(NUM_NAMES + CS_NAMES).round(1).to_string())


def G(q, c, p):
    return np.hstack([ns["group_features"](q, srcs[c], p), cross_source_feats(q, c, p)])


F2 = F1 + ns["G_FEATURES"] + XS_NAMES
vG = np.hstack([vX, G(vq, vc, vp1)])
m2 = fit(np.hstack([tX, G(tq, tc, oof)]), ty, vG, vy, F2)
m2.save_model(W3 + r"\exp_m2.json")
vp2 = pred(m2, vG, F2)
log(f"m2 best iteration {m2.best_iteration}")
print(pd.Series(m2.get_score(importance_type="gain")).reindex(XS_NAMES).round(1).to_string())
np.savez(W3 + r"\val_p_exp.npz", p1=vp1, p2=vp2)

import valscore as V  # noqa: E402

_v2 = np.load(W3 + r"\val_p_v2.npz")
for name, p in (("v2 p1", _v2["p1"]), ("v2 p2", _v2["p2"]), ("exp p1", vp1), ("exp p2", vp2)):
    t, fa, fb, fall = V.ab_score(p)
    print(f"{name:8s} t={t:.3f}  A={fa:.4f}  B={fb:.4f}  all={fall:.4f}", flush=True)
