# Shared validation scoring for the v3 experiments: the same 100k validation entities as v1/v2,
# split into halves A (tune the threshold) and B (report), plus the full set.
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

W = r"D:\AmazonMLChallenge2026\work\v2"
D = r"D:\AmazonMLChallenge2026\dataset\train"
rec = pq.read_table(W + r"\train_records.parquet", columns=["entity_id", "src"]).to_pandas()
N = len(rec)
srcs = rec.src.to_numpy()
_z = np.load(W + r"\val_pairs.npz")
q = _z["q"].astype(np.int64)
c = _z["c"].astype(np.int64)

ids = pd.Index(rec.entity_id)
_gt = pd.read_csv(D + "/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
_m = _gt.matched_entity_ids.str.split(",").explode()
_m = _m[_m != ""]
GQ = ids.get_indexer(_gt.source1_entity_id.loc[_m.index]).astype(np.int64)
GC = ids.get_indexer(_m.values).astype(np.int64)
VAL = np.unique(q)
_s = np.isin(GQ, VAL)
GQ, GC = GQ[_s], GC[_s]
GK = np.sort(GQ * N + GC)


def f_ent(qset, pq_, pc_):
    k = pq_ * N + pc_
    pos = np.minimum(np.searchsorted(GK, k), len(GK) - 1)
    hit = GK[pos] == k
    inq = np.isin(GQ, qset)
    nt = np.bincount(np.searchsorted(qset, GQ[inq]), minlength=len(qset))
    npd = np.bincount(np.searchsorted(qset, pq_), minlength=len(qset))
    tp = np.bincount(np.searchsorted(qset, pq_[hit]), minlength=len(qset))
    with np.errstate(all="ignore"):
        P = np.where(npd > 0, tp / np.maximum(npd, 1), 0)
        R = np.where(nt > 0, tp / np.maximum(nt, 1), 0)
        f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    return np.where((nt == 0) & (npd == 0), 1.0, f)


def owner(q_, c_, p):
    """one owner per S2/S3 record: keep each c only for its highest-probability q"""
    o = np.lexsort((-p, c_))
    first = np.r_[True, c_[o][1:] != c_[o][:-1]]
    keep = np.zeros(len(c_), bool)
    keep[o[first]] = True
    return keep


def threshold_rule(q_, c_, p, t):
    k = owner(q_, c_, p) & (p >= t)
    d = pd.DataFrame({"q": q_[k], "c": c_[k], "s": srcs[c_[k]], "p": p[k]}).sort_values("p", ascending=False)
    d = d[d.groupby(["q", "s"]).cumcount() < np.where(d.s == 2, 5, 6)]
    return d.q.to_numpy(), d.c.to_numpy()


_half = np.random.default_rng(7).random(len(VAL)) < 0.5
A, B = VAL[_half], VAL[~_half]


def ab_score(p, qq=None, cc=None, grid=np.arange(0.5, 0.91, 0.025)):
    """tune the threshold on half A, report on half B and on all 100k"""
    qq = q if qq is None else qq
    cc = c if cc is None else cc
    inA = np.isin(qq, A)
    thr = {t: f_ent(A, *threshold_rule(qq[inA], cc[inA], p[inA], t)).mean() for t in grid}
    tb = max(thr, key=thr.get)
    return (tb, thr[tb], f_ent(B, *threshold_rule(qq[~inA], cc[~inA], p[~inA], tb)).mean(),
            f_ent(VAL, *threshold_rule(qq, cc, p, tb)).mean())
