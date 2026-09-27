# v10 experiment: stack the (multilingual) cross-encoder WITH group context, replacing the 2-weight logistic blend.
# Papers: Ditto (cross-encoder reads the pair), TransClean / FAMER (a pair's credibility depends on its siblings).
# The GBDT already uses group context (T05/T09: +0.0055); the cross-encoder score has never seen it.
# Honest protocol: the stacker is trained on validation half A band pairs and scored on half B (the cross-encoder was
# trained on training entities, so its validation scores are out-of-sample).
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

import valscore as V

W4 = r"D:\AmazonMLChallenge2026\work\v4"
W9 = r"D:\AmazonMLChallenge2026\work\v9"
vz = np.load(W4 + r"\val_pairs.npz")
q, c, y = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
p2 = np.load(W9 + r"\val_p2_v4.npy")
band = np.flatnonzero((p2 >= 0.01) & (p2 <= 0.99))
pce = np.load(W9 + r"\val_pce_new.npy")             # multilingual cross-encoder, band pairs
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def ce_group_features(qb, sb, pc, p2b):
    """cross-encoder's view of each entity's band candidates (q-sorted not required)"""
    d = pd.DataFrame({"q": qb, "s": sb, "ce": pc, "p2": p2b})
    g = d.groupby("q")
    d["ce_rank"] = g.ce.rank(ascending=False, method="first")
    d["ce_gap"] = d.ce - g.ce.transform("max")
    d["ce_n05"] = g.ce.transform(lambda s: (s >= 0.5).sum())
    d["ce_n"] = g.ce.transform("size")
    per = d.groupby(["q", "s"]).ce.max().unstack(fill_value=0.0)
    other = np.where(sb == 2, 3, 2)
    d["ce_other_src_max"] = [per.at[a, o] if o in per.columns else 0.0 for a, o in zip(qb, other)]
    d["p2_gap"] = d.p2 - g.p2.transform("max")
    d["agree"] = np.abs(d.ce - d.p2)
    d["lp2"], d["lce"] = lg(d.p2.to_numpy()), lg(d.ce.to_numpy())
    cols = ["lp2", "lce", "ce_rank", "ce_gap", "ce_n05", "ce_n", "ce_other_src_max", "p2_gap", "agree"]
    return d[cols].to_numpy(np.float32), cols


srcs = V.srcs
qb, cb, yb = q[band], c[band], y[band]
X, cols = ce_group_features(qb, srcs[cb], pce, p2[band])
inA = np.isin(qb, V.A)

# baseline: current 2-weight logistic blend (fit on A)
lr = LogisticRegression(C=10.0).fit(X[inA][:, :2], yb[inA])
pb_lr = p2.copy()
pb_lr[band] = lr.predict_proba(X[:, :2])[:, 1]

# stacked: small GBDT on A (with a group split inside A for early stopping), CPU
rngA = np.random.default_rng(1).random(len(V.A)) < 0.8
A_fit, A_es = V.A[rngA], V.A[~rngA]
fit, es = np.isin(qb, A_fit), np.isin(qb, A_es)
params = dict(objective="binary:logistic", eval_metric="logloss", max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.9, min_child_weight=20, tree_method="hist", nthread=6, seed=0)
bst = xgb.train(params, xgb.DMatrix(X[fit], yb[fit], feature_names=cols), 2000,
                evals=[(xgb.DMatrix(X[es], yb[es], feature_names=cols), "es")], early_stopping_rounds=50, verbose_eval=False)
pb_st = p2.copy()
pb_st[band] = bst.predict(xgb.DMatrix(X, feature_names=cols), iteration_range=(0, bst.best_iteration + 1))
print(f"stacker: {bst.best_iteration} trees; gain: " +
      ", ".join(f"{k} {v:.0f}" for k, v in sorted(bst.get_score(importance_type='gain').items(), key=lambda kv: -kv[1])))
for name, p in (("group model only (no CE)", p2), ("logistic blend (current)", pb_lr), ("stacked CE + group ctx", pb_st)):
    t, fa, fb, fall = V.ab_score(p, qq=q, cc=c)
    print(f"{name:26s} t={t:.3f}  A={fa:.4f}  B={fb:.4f}  all={fall:.4f}")
np.save(W9 + r"\val_pb_stack.npy", pb_st)
bst.save_model(W9 + r"\ce_stacker.json")
