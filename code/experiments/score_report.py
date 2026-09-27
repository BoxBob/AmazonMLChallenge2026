# Score report with the competition metric (macro F0.5 per S1 entity, singletons included) on the labelled
# validation set (US/India), plus a loss decomposition. Usage: score_report.py <val_probability_npy> [threshold]
import sys, numpy as np, pandas as pd
import valscore as V
W4 = r"D:\AmazonMLChallenge2026\work\v4"
vz = np.load(W4 + r"\val_pairs.npz"); q, c, y = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
p = np.load(sys.argv[1])
t, fa, fb, fall = V.ab_score(p, qq=q, cc=c)
if len(sys.argv) > 2: t = float(sys.argv[2])
mq, mc = V.threshold_rule(q, c, p, t)
f = V.f_ent(V.VAL, mq, mc)
print(f"macro F0.5  all={f.mean():.4f}  (threshold {t:.3f} tuned on half A; half A {fa:.4f} / half B {fb:.4f})")
ctry = pd.Series(V.rec.entity_id.to_numpy()[V.VAL]).str[:0]  # placeholder
import pyarrow.parquet as pq
cc = pq.read_table(r"D:\AmazonMLChallenge2026\work\v2\train_records.parquet", columns=["country"]).column(0).to_numpy(zero_copy_only=False)
nt = np.bincount(np.searchsorted(V.VAL, V.GQ), minlength=len(V.VAL))
d = pd.DataFrame({"f": f, "country": cc[V.VAL], "n_true": pd.cut(nt, [-1, 0, 1, 3, 5, 20], labels=["singleton", "1", "2-3", "4-5", "6+"])})
print("by country:", d.groupby("country").f.mean().round(4).to_dict())
print("by #true matches:", d.groupby("n_true", observed=True).f.mean().round(4).to_dict())
kp = mq * V.N + mc; tp = np.isin(kp, V.GK)
f_nofp = V.f_ent(V.VAL, mq[tp], mc[tp]).mean()
allk = np.union1d(kp, q[y == 1] * V.N + c[y == 1])
f_nofn = V.f_ent(V.VAL, allk // V.N, allk % V.N).mean()
f_or = V.f_ent(V.VAL, q[y == 1], c[y == 1]).mean()
print(f"loss {1 - f.mean():.4f} = false merges {f_nofp - f.mean():.4f} + rejected true pairs {f_nofn - f.mean():.4f} "
      f"+ blocking (never candidates) {1 - f_or:.4f}  [overlap-adjusted: {1 - f.mean() - (f_nofp - f.mean()) - (f_nofn - f.mean()) - (1 - f_or):+.4f}]")
