# v11 configuration choice on the v10 validation candidates (hold-out B decides; A tunes blend and threshold):
# cross-encoder(s) {v11 alone, v9 + v11} x band {0.01-0.99, 0.001-0.999} x {blend only, + zone re-stacker (T31)}.
# Uses the wide-band caches written by exp_v11_mce.py (val_pce_v9.npy / val_pce_v11.npy over p2 in [0.001, 0.999]).
import json
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import valscore as V

W11 = r"D:\AmazonMLChallenge2026\work\v11"
vz = np.load(r"D:\AmazonMLChallenge2026\work\v10m80\val_pairs.npz")
vq, vc, vy, X = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"], vz["X"]
FEAT = xgb.Booster(model_file=r"D:\AmazonMLChallenge2026\work\v4\m1.json").feature_names
ST_F = ["n_tset", "a_tset", "hn_eq", "pf_rank", "addr_blank_c", "rank_core"]
p2 = np.load(W11 + r"\val_p2_v10.npy")
wide = np.flatnonzero((p2 >= 0.001) & (p2 <= 0.999))
PCE = {t: np.load(W11 + rf"\val_pce_{t}.npy") for t in ("v9", "v11")}
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
PRM = dict(objective="binary:logistic", eval_metric="logloss", max_depth=6, eta=0.05, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=5, tree_method="hist")
res = {}
for ces in (("v11",), ("v9", "v11")):
    for lo, hi in ((0.01, 0.99), (0.001, 0.999)):
        m = (p2[wide] >= lo) & (p2[wide] <= hi)
        band = wide[m]
        Xb = np.column_stack([lg(p2[band])] + [lg(PCE[t][m]) for t in ces])
        inA = np.isin(vq[band], V.A)
        pb = p2.copy()
        pb[band] = LogisticRegression(C=10.0).fit(Xb[inA], vy[band][inA]).predict_proba(Xb)[:, 1]
        name = f"{'+'.join(ces)} band {lo}-{hi}"
        t, fa, fb, fall = V.ab_score(pb, qq=vq, cc=vc)
        res[name] = fb
        print(f"{name:32s} blend    : t={t:.3f} A={fa:.4f} B={fb:.4f} all={fall:.4f}", flush=True)
        # zone re-stacker, cross-fitted A<->B
        full = {t: np.full(len(p2), 0.5, np.float32) for t in ces}
        for t in ces:
            full[t][band] = PCE[t][m]
        zone = np.flatnonzero((pb >= 0.01) & (pb <= 0.99))
        F = np.column_stack([lg(pb[zone]), lg(p2[zone])] + [lg(full[t][zone]) for t in ces]
                            + [X[zone][:, [FEAT.index(f) for f in ST_F]]]).astype(np.float32)
        inAz = np.isin(vq[zone], V.A)
        oof = np.zeros(len(zone), np.float32)
        for tr in (inAz, ~inAz):
            b = xgb.train(PRM, xgb.DMatrix(F[tr], label=vy[zone][tr]), 400)
            oof[~tr] = b.predict(xgb.DMatrix(F[~tr]))
        pn = pb.copy()
        pn[zone] = oof
        t, fa, fb, fall = V.ab_score(pn, qq=vq, cc=vc)
        res[name + " +stacker"] = fb
        print(f"{name:32s} +stacker: t={t:.3f} A={fa:.4f} B={fb:.4f} all={fall:.4f}", flush=True)
json.dump(res, open(W11 + r"\select_results.json", "w"), indent=1)
print("best:", max(res, key=res.get), max(res.values()))
