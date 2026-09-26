# Where is v4's remaining validation loss (0.0174)? Counterfactual decomposition, plus loss by pattern class.
import numpy as np, pandas as pd
import valscore as V
W4 = r"D:\AmazonMLChallenge2026\work\v4"
vz = np.load(W4 + r"\val_pairs.npz"); q, c, y = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
pb = np.load(W4 + r"\val_p_v4.npy")
mq, mc = V.threshold_rule(q, c, pb, 0.70)
base = V.f_ent(V.VAL, mq, mc)
k_pred = mq * V.N + mc
kt = q[y == 1] * V.N + c[y == 1]
tp = np.isin(k_pred, V.GK)
f_nofp = V.f_ent(V.VAL, mq[tp], mc[tp]).mean()
allk = np.union1d(k_pred, kt)
f_nofn = V.f_ent(V.VAL, allk // V.N, allk % V.N).mean()
f_or = V.f_ent(V.VAL, q[y == 1], c[y == 1]).mean()
print(f"v4 val {base.mean():.4f}; loss {1 - base.mean():.4f}")
print(f"  FP cost {f_nofp - base.mean():.4f} | classifier-FN cost {f_nofn - base.mean():.4f} | blocking cost {1 - f_or:.4f}")
# which S1 entities lose, by number of true matches
nt = np.bincount(np.searchsorted(V.VAL, V.GQ), minlength=len(V.VAL))
b = pd.cut(nt, [-1, 0, 1, 3, 5, 20], labels=["singleton", "1", "2-3", "4-5", "6+"])
d = pd.DataFrame({"b": b, "loss": 1 - base})
print((d.groupby("b", observed=True).loss.sum() / len(d)).round(4).to_string())
