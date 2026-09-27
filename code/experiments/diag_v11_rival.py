# Blank-address zone pairs: does "another S1 in the whole pool has the same name" separate true from false?
import numpy as np, pandas as pd, pyarrow.parquet as pq, xgboost as xgb
from sklearn.linear_model import LogisticRegression
import valscore as V
W11 = r"D:\AmazonMLChallenge2026\work\v11"
vz = np.load(r"D:\AmazonMLChallenge2026\work\v10m80\val_pairs.npz")
vq, vc, vy, X = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"], vz["X"]
F = xgb.Booster(model_file=r"D:\AmazonMLChallenge2026\work\v4\m1.json").feature_names
p2 = np.load(W11 + r"\val_p2_v10.npy"); pce = np.load(W11 + r"\val_pce_v9_narrow.npy")
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
band = np.flatnonzero((p2 >= 0.01) & (p2 <= 0.99)); Xb = np.column_stack([lg(p2[band]), lg(pce)])
pb = p2.copy(); pb[band] = LogisticRegression(C=10.0).fit(Xb, vy[band]).predict_proba(Xb)[:, 1]
rec = pq.read_table(V.W + r"\train_records.parquet", columns=["core", "addr", "src"]).to_pandas()
s1 = rec.src.to_numpy() == 1
cnt = rec.core[s1].value_counts()                         # exact core-name frequency over ALL train S1
blank = rec.addr.to_numpy()[vc] == ""
z = (pb > 0.05) & (pb < 0.95) & blank
n_same = rec.core.to_numpy()[vc[z]]
k = pd.Series(n_same).map(cnt).fillna(0).to_numpy()
qn = pd.Series(rec.core.to_numpy()[vq[z]]).map(cnt).fillna(0).to_numpy()
d = pd.DataFrame({"y": vy[z], "S1 with c's exact name": np.minimum(k, 4), "S1 with q's name": np.minimum(qn, 4)})
print(d.groupby("S1 with c's exact name").y.agg(["mean", "size"]).round(3))
print(d.groupby("S1 with q's name").y.agg(["mean", "size"]).round(3))
for f in ("s1_name_cnt_q", "s1_name_cnt_c", "pool_name_cnt_q", "freq_name_c", "name_unique_pair"):
    print(f, pd.DataFrame({"y": vy[z], "v": np.minimum(X[z, F.index(f)], 5)}).groupby("v").y.agg(["mean", "size"]).round(3).T.to_dict())
