# quick error analysis on the cached validation pairs (no retraining)
import numpy as np, pandas as pd, pyarrow.parquet as pq, xgboost as xgb
W = r"D:\AmazonMLChallenge2026\work\baseline"
cands = pd.read_parquet(W + r"\valpairs_cands.parquet"); X = pd.read_parquet(W + r"\valpairs_features.parquet")
b = xgb.Booster(); b.load_model(W + r"\xgb_baseline.json"); b.set_param({"device": "cuda"})
p = b.predict(xgb.DMatrix(X.to_numpy(np.float32), feature_names=list(X.columns)))
tbl = pq.read_table(W + r"\train_records.parquet", columns=["entity_id", "core", "addr", "country"]).to_pandas()
tbl = tbl.set_index(np.arange(len(tbl)))
cands["p"] = p; T = 0.625
fp = cands[(cands.p >= T) & (cands.label == 0)]; fn = cands[(cands.p < T) & (cands.label == 1)]
print("val pairs", len(cands), "pos", cands.label.sum(), "FP", len(fp), "FN(in cands)", len(fn))
for name, df in (("FP", fp.sample(8, random_state=1)), ("FN", fn.sample(8, random_state=1))):
    print("==", name)
    for r in df.itertuples():
        a, c = tbl.loc[r.q], tbl.loc[r.c]
        print(f"  p={r.p:.2f} {a.country} | {a.core!r} / {a.addr!r}  <->  {c.core!r} / {c.addr!r}")
X["label"] = cands.label; X["pred"] = p >= T
print(pd.crosstab([X.addr_blank_c, X.label], X.pred))
