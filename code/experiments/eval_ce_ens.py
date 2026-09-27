# Blend p2 with several cached cross-encoder scores on the v10 validation band (A/B protocol).  usage: eval_ce_ens.py tag1 tag2 ...
import sys, numpy as np
from sklearn.linear_model import LogisticRegression
import valscore as V
W11 = r"D:\AmazonMLChallenge2026\work\v11"
vz = np.load(r"D:\AmazonMLChallenge2026\work\v10m80\val_pairs.npz")
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
p2 = np.load(W11 + r"\val_p2_v10.npy"); band = np.flatnonzero((p2 >= 0.01) & (p2 <= 0.99))
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
P = {t: np.load(W11 + rf"\val_pce_{t}_narrow.npy") for t in sys.argv[1:]}
Xb = np.column_stack([lg(p2[band])] + [lg(P[t]) for t in sys.argv[1:]]); inA = np.isin(vq[band], V.A)
lr = LogisticRegression(C=10.0).fit(Xb[inA], vy[band][inA]); pb = p2.copy(); pb[band] = lr.predict_proba(Xb)[:, 1]
t, fa, fb, fall = V.ab_score(pb, qq=vq, cc=vc)
print(f"{'+'.join(sys.argv[1:]):24s}: blend {lr.coef_.round(3).tolist()} | t={t:.3f} A={fa:.4f} B={fb:.4f} all={fall:.4f}")
