# v5 experiment: domain adaptation to an unseen country, evaluated with a leave-one-country-out protocol (DADER-style).
# Source = US (labelled), target = India (labels hidden during adaptation, used only for scoring).
# Methods compared on India validation entities (macro F0.5, threshold chosen on the SOURCE validation only):
#   M0  source only                      - train on US, apply to India
#   M1  self-training (pseudo-labels)    - DADER family: add confident target predictions as labels, retrain
#   M2  EM prior correction              - Saerens et al. 2002: re-estimate the positive prior on the target, adjust
#   M3  per-country quantile alignment   - distribution alignment (CORAL family): rank-normalise features per country
#   UB  target oracle                    - train on India with labels (upper bound, not usable for France)
import gc
import os
import time

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

import valscore as V

W3 = r"D:\AmazonMLChallenge2026\work\v3"
SH = r"D:\AmazonMLChallenge2026\work\v2"
T0 = time.time()
N_ENT = int(os.environ.get("LOCO_ENT", 80_000))      # entities per domain for training (CPU budget)


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


ctry = pq.read_table(SH + r"\train_records.parquet", columns=["country"]).column(0).to_numpy(zero_copy_only=False)
tz, vz = np.load(W3 + r"\train_pairs.npz"), np.load(W3 + r"\val_pairs.npz")
tq, ty = tz["q"].astype(np.int64), tz["y"]
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
names = xgb.Booster(model_file=W3 + r"\m1.json").feature_names
rng = np.random.default_rng(0)


def pick(country):
    ents = np.unique(tq[ctry[tq] == country])
    ents = rng.choice(ents, min(N_ENT, len(ents)), replace=False)
    return np.flatnonzero(np.isin(tq, ents))


src_i, tgt_i = pick("US"), pick("India")
Xs, ys = tz["X"][src_i], ty[src_i]
Xt, yt_hidden = tz["X"][tgt_i], ty[tgt_i]
vsrc = ctry[vq] == "US"
vtgt = ctry[vq] == "India"
Xvs, yvs, Xvt = vz["X"][vsrc], vy[vsrc], vz["X"][vtgt]
del tz
gc.collect()
log(f"source pairs {len(ys):,}  target pairs {len(yt_hidden):,}  val US {vsrc.sum():,}  val India {vtgt.sum():,}")

PAR = dict(objective="binary:logistic", eval_metric="logloss", device="cpu", nthread=4, tree_method="hist",
           max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, seed=42)


def fit(X, y, Xv, yv, w=None):
    d = xgb.DMatrix(X, y, weight=w, feature_names=names)
    dv = xgb.DMatrix(Xv, yv, feature_names=names)
    return xgb.train(PAR, d, 1500, evals=[(dv, "v")], early_stopping_rounds=40, verbose_eval=False)


def pr(b, X):
    return b.predict(xgb.DMatrix(X, feature_names=names), iteration_range=(0, b.best_iteration + 1))


VAL_US = V.VAL[ctry[V.VAL] == "US"]
VAL_IN = V.VAL[ctry[V.VAL] == "India"]


def score(p_src_val, p_tgt_val):
    """threshold chosen on the SOURCE (US) validation only, applied to the target (India) validation"""
    grid = np.arange(0.3, 0.95, 0.025)
    fs = [V.f_ent(VAL_US, *V.threshold_rule(vq[vsrc], vc[vsrc], p_src_val, t)).mean() for t in grid]
    t = grid[int(np.argmax(fs))]
    f_t = V.f_ent(VAL_IN, *V.threshold_rule(vq[vtgt], vc[vtgt], p_tgt_val, t)).mean()
    f_best = max(V.f_ent(VAL_IN, *V.threshold_rule(vq[vtgt], vc[vtgt], p_tgt_val, u)).mean() for u in grid)
    return t, f_t, f_best


res = {}
m0 = fit(Xs, ys, Xvs, yvs)
p0_s, p0_t = pr(m0, Xvs), pr(m0, Xvt)
res["M0 source only"] = score(p0_s, p0_t)
log(f"M0 done: {res['M0 source only']}")

# M1 self-training: confident target predictions become labels (two rounds, class-balanced confidence cuts)
Xa, ya = Xs, ys
b = m0
for rnd in (1, 2):
    pt = pr(b, Xt)
    keep = (pt >= 0.97) | (pt <= 0.03)
    log(f"  self-training round {rnd}: pseudo-labelled {keep.mean():.1%} of target pairs, "
        f"pseudo-label accuracy {(((pt >= 0.5) == (yt_hidden == 1))[keep]).mean():.4f} (hidden labels, diagnostic only)")
    Xa = np.vstack([Xs, Xt[keep]])
    ya = np.concatenate([ys, (pt[keep] >= 0.5).astype(np.int8)])
    b = fit(Xa, ya, Xvs, yvs)
res["M1 self-training"] = score(pr(b, Xvs), pr(b, Xvt))
log(f"M1 done: {res['M1 self-training']}")


# M2 Saerens EM prior correction on the target validation pairs
def saerens(p, prior_src, iters=100):
    pi = prior_src
    for _ in range(iters):
        a = (pi / prior_src) * p
        b_ = ((1 - pi) / (1 - prior_src)) * (1 - p)
        post = a / (a + b_)
        new = post.mean()
        if abs(new - pi) < 1e-7:
            break
        pi = new
    return post, pi


prior_src = ys.mean()
p2_t, pi_t = saerens(p0_t, prior_src)
res["M2 EM prior correction"] = score(p0_s, p2_t)
log(f"M2 done: source prior {prior_src:.4f}, estimated target prior {pi_t:.4f} (true {vy[vtgt].mean():.4f}); "
    f"{res['M2 EM prior correction']}")


# M3 per-domain quantile alignment: each feature rank-normalised within its own domain
def qnorm(X, ref):
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        r = np.sort(ref[:, j])
        out[:, j] = np.searchsorted(r, X[:, j], side="right") / len(r)
    return out


m3 = fit(qnorm(Xs, Xs), ys, qnorm(Xvs, Xs), yvs)
res["M3 quantile alignment"] = score(pr(m3, qnorm(Xvs, Xs)), pr(m3, qnorm(Xvt, Xt)))
log(f"M3 done: {res['M3 quantile alignment']}")

# UB: target oracle
ub = fit(Xt, yt_hidden, Xvt, vy[vtgt])
pu = pr(ub, Xvt)
res["UB target oracle"] = (None, None, max(V.f_ent(VAL_IN, *V.threshold_rule(vq[vtgt], vc[vtgt], pu, u)).mean()
                                         for u in np.arange(0.3, 0.95, 0.025)))
print("\nmethod                     | threshold from source | India F0.5 at that threshold | India F0.5 at best threshold (oracle)")
for k, (t, f, fb) in res.items():
    print(f"{k:26s} | {'' if t is None else f'{t:.3f}':>21s} | {'' if f is None else f'{f:.4f}':>28s} | {fb:.4f}")
