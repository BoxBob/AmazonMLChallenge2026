# Where are the true pairs the v10 decision loses? (probability bucket of the blend, blank address, has an anchor sibling)
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
import valscore as V
W11 = r"D:\AmazonMLChallenge2026\work\v11"
vz = np.load(r"D:\AmazonMLChallenge2026\work\v10m80\val_pairs.npz")
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
p2 = np.load(W11 + r"\val_p2_v10.npy"); pce = np.load(W11 + r"\val_pce_v9_narrow.npy")
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
band = np.flatnonzero((p2 >= 0.01) & (p2 <= 0.99)); Xb = np.column_stack([lg(p2[band]), lg(pce)])
pb = p2.copy(); pb[band] = LogisticRegression(C=10.0).fit(Xb, vy[band]).predict_proba(Xb)[:, 1]
fq, fc = V.threshold_rule(vq, vc, pb, 0.70)
N = V.N; sel = np.isin(vq * N + vc, fq * N + fc)
addr = pq.read_table(V.W + r"\train_records.parquet", columns=["addr"])["addr"].to_numpy(zero_copy_only=False)
blank = np.array([not a for a in addr[vc]])
anch = pd.Series(pb >= 0.8).groupby(vq).transform("sum").to_numpy() - (pb >= 0.8)
lost = (vy == 1) & ~sel; fp = (vy == 0) & sel
print(f"true pairs in candidates {vy.sum():,}; lost {lost.sum():,}; false accepted {fp.sum():,}")
bk = pd.cut(pb, [-1, 1e-4, 1e-3, 1e-2, .1, .3, .5, .7, .9, 1.01])
print(pd.DataFrame({"bucket": bk[lost], "blank": blank[lost], "has_anchor": anch[lost] > 0}).groupby("bucket", observed=True)
      .agg(n=("blank", "size"), blank=("blank", "mean"), has_anchor=("has_anchor", "mean")).round(3))
print("lost true pairs with p >= threshold (lost to owner rule / caps):", (lost & (pb >= 0.7)).sum())
print("false accepted by bucket:", pd.Series(bk[fp]).value_counts().sort_index().to_dict())
tb = pq.read_table(V.W + r"\train_records.parquet", columns=["name", "addr", "src"]).to_pandas()
rng = np.random.default_rng(1)
zq = np.unique(vq[lost & blank & (pb > 0.1)])
for q in rng.choice(zq, 8, replace=False):
    m = np.flatnonzero((vq == q) & (pb > 0.02))
    m = m[np.argsort(-pb[m])][:9]
    print(f"\nS1 {tb.name[q][:40]:40s} | {tb.addr[q][:50]}")
    for i in m:
        tag = ("TRUE " if vy[i] else "false") + (" LOST" if lost[i] else "") + (" FP" if fp[i] else "")
        print(f"   {tag:10s} p={pb[i]:.2f} S{tb.src[vc[i]]} {tb.name[vc[i]][:40]:40s} | {tb.addr[vc[i]][:45]}")
