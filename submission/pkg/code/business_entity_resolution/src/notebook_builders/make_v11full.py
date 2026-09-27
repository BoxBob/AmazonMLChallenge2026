"""Build v11full_src.py (-> code/src/v11_full.ipynb) from v10full_src.py:
  * cross-encoder: the v11 data-rich multilingual model (T35, exp_v11_mce.py), band from env V11_BAND ("0.01,0.99")
  * T31: the undecided zone (blend p 0.01-0.99) is re-stacked by a small XGBoost, cross-fitted on validation halves
    A<->B (out-of-fold on validation; the two fold models are averaged on test)
  * validation pairs reused from v10full (same top-80 candidates); per-chunk test arrays cached for later re-scoring
"""
import os

s = open("v10full_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


rep("# # Business Entity Resolution — v10: prefilter top-80 + multilingual cross-encoder (full test inference)",
    "# # Business Entity Resolution — v11: data-rich cross-encoder + zone re-stacker (full test inference)")
rep('WORK = ROOT / "work" / "v10full"', 'WORK = ROOT / "work" / os.environ.get("V11_WORK", "v11full")')
rep('OUT_MATCH = OUT / "lb_probe" / "matching_results_v10_raw.tsv"',
    'OUT_MATCH = OUT / "lb_probe" / "matching_results_v11_raw.tsv"')
rep('OUT_CAND = OUT / "lb_probe" / "candidate_pairs_v10.tsv"', 'OUT_CAND = OUT / "lb_probe" / "candidate_pairs_v11.tsv"')
rep("PREV = WORK                                   # v10: rebuild VALIDATION pairs with the new cut",
    'PREV = ROOT / "work" / "v10full"             # v11: the v10 top-80 validation pairs (identical candidates)')
rep('CE_DIR = ROOT / "work" / "v9" / "mce_model"   # v9: multilingual cross-encoder',
    'CE_DIR = Path(os.environ.get("V11_CE", str(ROOT / "work" / "v11" / "mce_model")))   # v11: data-rich multilingual CE')
band = os.environ.get("V11_BAND", "0.01,0.99")
rep("CE_BAND = (0.01, 0.99)      # the cross-encoder re-scores pairs whose GBDT probability is in this band",
    f"CE_BAND = ({band})      # v11: band chosen on validation (exp_v11_mce.py)")

# ---- T31 zone re-stacker, fit right after the validation blend
rep("""pb_va = p2_va.copy()
pb_va[va_band] = blend.predict_proba(_Xb)[:, 1]
""", """pb_va = p2_va.copy()
pb_va[va_band] = blend.predict_proba(_Xb)[:, 1]

# v11 (T31): re-stack the undecided zone with a small XGBoost, cross-fitted on validation halves A<->B, so every
# validation probability stays out-of-fold; on test the two fold models are averaged.
ZONE = (0.01, 0.99)
ST_F = ["n_tset", "a_tset", "hn_eq", "pf_rank", "addr_blank_c", "rank_core"]
ST_PRM = dict(objective="binary:logistic", eval_metric="logloss", max_depth=6, eta=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, tree_method="hist")


def zone_feats(pb, p2, pce_full, X, idx):
    return np.column_stack([_logit(pb[idx]), _logit(p2[idx]), _logit(np.nan_to_num(pce_full[idx], nan=0.5)),
                            X[idx][:, [FEATURES.index(f) for f in ST_F]]]).astype(np.float32)


_pce_full = np.full(len(pb_va), np.nan, np.float32)
_pce_full[va_band] = pce_va
va_zone = np.flatnonzero((pb_va >= ZONE[0]) & (pb_va <= ZONE[1]))
_Fz = zone_feats(pb_va, p2_va, _pce_full, va_X, va_zone)
_inAz = np.isin(va_q[va_zone], VAL_A)
_oof = np.zeros(len(va_zone), np.float32)
STACKERS = []
for _tr in (_inAz, ~_inAz):
    _b = xgb.train(ST_PRM, xgb.DMatrix(_Fz[_tr], label=va_y[va_zone][_tr]), 400)
    _oof[~_tr] = _b.predict(xgb.DMatrix(_Fz[~_tr]))
    STACKERS.append(_b)
pb_va[va_zone] = _oof
pb_A[va_zone] = _oof
log(f"zone re-stacker: {len(va_zone):,} validation pairs re-scored out-of-fold")
""")

# ---- test: keep the group-model probability, apply the stacker, cache chunk arrays
rep("""            pce = ce_score(test, q[band], c[band])
            p = p.copy()
            p[band] = blend.predict_proba(np.column_stack([_logit(p[band]), _logit(pce)]))[:, 1]
""", """            pce = ce_score(test, q[band], c[band])
            p2t = p
            p = p.copy()
            p[band] = blend.predict_proba(np.column_stack([_logit(p[band]), _logit(pce)]))[:, 1]
            pce_full = np.full(len(p), np.nan, np.float32)
            pce_full[band] = pce
            zone = np.flatnonzero((p >= ZONE[0]) & (p <= ZONE[1]))
            _Fzt = zone_feats(p, p2t, pce_full, X, zone)
            p[zone] = np.mean([b.predict(xgb.DMatrix(_Fzt)) for b in STACKERS], axis=0)
            np.savez(CKPT / f"arr_{a:08d}.npz", q=q.astype(np.int32), c=c.astype(np.int32), p2=p2t.astype(np.float32),
                     pce=pce_full.astype(np.float16), st=X[:, [FEATURES.index(f) for f in ST_F]].astype(np.float16))
""")
rep('log("SILVER REPORT', 'log("SILVER REPORT (templated French names collide: not a France accuracy measure, T34) ')
open("v11full_src.py", "w", encoding="utf-8").write(s)
print("v11full_src.py written", len(s.splitlines()), "lines; CE band", band)
