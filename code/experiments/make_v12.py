"""Build v12_src.py (-> code/src/v12.ipynb) from v11full_src.py + v4's training code:
  * learned blocking (T37): a 4th kNN channel "bi" from the multilingual bi-encoder "b" (trained without validation and
    GBDT-training entities; exp_v12_biencoder.py / exp_v12_embed.py) + features rank_bi, cos_bi, rel_cos_bi
  * the prefilter, stage-1 matcher (2-fold OOF) and group-context model are RETRAINED with the new features
    (v4 code); training pairs at the prefilter top-40 (RAM), validation / test at top-80 (T29)
  * the v11 cross-encoder (wide band) + zone re-stacker (+ cos_bi) are kept
"""
s = open("v11full_src.py", encoding="utf-8").read()
v4 = open("v4_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


def cut(text, start, end):
    i = text.index(start)
    j = text.index(end, i) + len(end)
    return text[i:j]


rep("# # Business Entity Resolution — v11: data-rich cross-encoder + zone re-stacker (full test inference)",
    "# # Business Entity Resolution — v12: learned blocking channel (bi-encoder) + retrained GBDTs\n#\n"
    "# v11 (leaderboard 0.977) + a 4th retrieval channel from a contrastively fine-tuned multilingual bi-encoder:\n"
    "# on validation it lifts candidate recall 98.0% -> 99.4% with +4 candidates per S1 (oracle ceiling 0.9939 -> 0.998).")
rep('WORK = ROOT / "work" / os.environ.get("V11_WORK", "v11full")', 'WORK = ROOT / "work" / "v12full"')
rep('OUT_MATCH = OUT / "lb_probe" / "matching_results_v11_raw.tsv"', 'OUT_MATCH = OUT / "lb_probe" / "matching_results_v12_raw.tsv"')
rep('OUT_CAND = OUT / "lb_probe" / "candidate_pairs_v11.tsv"', 'OUT_CAND = OUT / "lb_probe" / "candidate_pairs_v12.tsv"')
rep('K = {"core": 15, "addr": 20, "comb": 30}   # kNN neighbours per (S1, source) per channel',
    'K = {"core": 15, "addr": 20, "comb": 30}   # kNN neighbours per (S1, source) per channel\n'
    'K_BI = 10                   # v12: bi-encoder neighbours per (S1, source)\n'
    'BI_DIR = ROOT / "work" / "v12"   # bi-encoder embeddings + kNN (exp_v12_embed.py)')

# ---- bi-encoder embeddings join the EMB dict (fp16 memmaps, 384-d, L2-normalised)
rep("""    del vec, V
    gc.collect(); torch.cuda.empty_cache()
""", """    del vec, V
    gc.collect(); torch.cuda.empty_cache()
for rec in (train, test):                                  # v12: learned embedding (bi-encoder "b")
    EMB[(rec.split, "bi")] = np.load(BI_DIR / f"{rec.split}_emb_bi.npy", mmap_mode="r")
    assert EMB[(rec.split, "bi")].shape[0] == rec.n
""")

# ---- the new channel in stage 1 + prefilter features
rep('STAGE_CHANNELS = CHANNELS + ("blank",)\nK["blank"] = K_BLANK',
    'STAGE_CHANNELS = CHANNELS + ("blank", "bi")\nK["blank"] = K_BLANK\nK["bi"] = K_BI')
rep('PF_FEATURES = ["rank_core", "rank_addr", "rank_comb", "rank_blank", "cos_core", "cos_addr", "cos_comb", *EXACT_KEYS,\n'
    '               "src_c", "addr_blank_q", "addr_blank_c", "n1_cands", "rel_cos_core", "rel_cos_addr", "rel_cos_comb"]',
    'PF_FEATURES = ["rank_core", "rank_addr", "rank_comb", "rank_blank", "cos_core", "cos_addr", "cos_comb", *EXACT_KEYS,\n'
    '               "src_c", "addr_blank_q", "addr_blank_c", "n1_cands", "rel_cos_core", "rel_cos_addr", "rel_cos_comb",\n'
    '               "rank_bi", "cos_bi", "rel_cos_bi"]                     # v12: learned blocking channel')
rep('    F["cos_comb"] = (F["cos_core"] + F["cos_addr"]) / 2\n',
    '    F["cos_comb"] = (F["cos_core"] + F["cos_addr"]) / 2\n    F["cos_bi"] = _row_cos(EMB[(rec.split, "bi")], q, c)\n')
rep('''    for ch in CHANNELS:
        F[f"rel_cos_{ch}"] = F[f"cos_{ch}"] - _group_max(q, F[f"cos_{ch}"])
''', '''    for ch in CHANNELS + ("bi",):
        F[f"rel_cos_{ch}"] = F[f"cos_{ch}"] - _group_max(q, F[f"cos_{ch}"])
''')

# ---- kNN caches: add the bi channel (train Q_TV order / test S1 order, written by exp_v12_embed.py)
rep('''KNN_TV["blank"] = np.load(_bc) if _bc.exists() else gpu_knn_blank(train, Q_TV)
np.save(_bc, KNN_TV["blank"])
''', '''KNN_TV["blank"] = np.load(_bc) if _bc.exists() else gpu_knn_blank(train, Q_TV)
np.save(_bc, KNN_TV["blank"])
KNN_TV["bi"] = np.load(BI_DIR / "train_knn_bi.npy")                  # v12
assert KNN_TV["bi"].shape == (len(Q_TV), 2 * K_BI)
''')
rep('''KNN_TE["blank"] = np.load(_bc) if _bc.exists() else gpu_knn_blank(test, test_q)
np.save(_bc, KNN_TE["blank"])
''', '''KNN_TE["blank"] = np.load(_bc) if _bc.exists() else gpu_knn_blank(test, test_q)
np.save(_bc, KNN_TE["blank"])
KNN_TE["bi"] = np.load(BI_DIR / "test_knn_bi.npy")                   # v12
assert KNN_TE["bi"].shape == (len(test_q), 2 * K_BI)
''')

# ---- prefilter: retrain (v4 code), then use top-80 for validation/test (T29) and top-40 for training pairs
pf_train = cut(v4, "# stage-1 recall on validation (before prefilter)", "del Xp, yp, Xe, ye, qe, pe\ngc.collect()")
rep('''# v5: reuse the v4 prefilter model and its cut M (read from the v4 run log)
pf_model = xgb.Booster(model_file=str(PREV / "prefilter.json"))
PF_M = int(os.environ.get("PF_M_OVERRIDE", "80"))   # v10: larger prefilter cut (v4 used 40)
log(f"prefilter keeps top {PF_M} candidates per S1")
''', pf_train + '''
PF_M_SEL = PF_M
PF_M = 80                                   # T29: validation / test use the top-80
log(f"retention rule picked top {PF_M_SEL}; validation/test use top {PF_M}, training pairs top 40")
''')

# ---- training + validation pairs rebuilt with the new channel
rep('''PREV = ROOT / "work" / "v10full"             # v11: the v10 top-80 validation pairs (identical candidates)
va_q, va_c, va_X, va_y = build_pairs("val", VAL_Q)
PREV = _prev_cache                            # training pairs: v4 cache (models are not retrained)''',
    '''PREV = WORK                                   # v12: new candidates + features -> rebuild both
va_q, va_c, va_X, va_y = build_pairs("val", VAL_Q)
PF_M = 40                                     # training pairs at top-40 (RAM; v4-v11 models were trained this way)
tr_q, tr_c, tr_X, tr_y = build_pairs("train", TRAIN_Q)
PF_M = 80
PREV = _prev_cache
log(f"pairs: train {len(tr_q):,} ({tr_y.mean():.2%} true) | val {len(va_q):,}; "
    f"val candidate recall {va_y.sum() / n_true(VAL_Q):.4f} (v10/v11: 0.9801)")''')

# ---- stage-1 matcher: retrain with out-of-fold predictions (v4 code)
m1_train = cut(v4, "fold = (np.random.default_rng(SEED + 2)", 'm1.save_model(WORK / "m1.json")')
rep('m1 = xgb.Booster(model_file=str(PREV / "m1.json"))      # v5: the v4 stage-1 matcher\n', m1_train + "\n")

# ---- group-context model: retrain
rep('''Gva = group_ctx(train, va_q, va_c, p1_va)
m2 = xgb.Booster(model_file=str(PREV / "m2.json"))      # v5: the v4 group-context model
''', '''Gtr = group_ctx(train, tr_q, tr_c, p_oof)
Gva = group_ctx(train, va_q, va_c, p1_va)
m2 = fit(np.hstack([tr_X, Gtr]), tr_y, np.hstack([va_X, Gva]), va_y, ALL_FEATURES)
m2.save_model(WORK / "m2.json")
del Gtr
''')

# ---- re-stacker also sees the learned similarity
rep('ST_F = ["n_tset", "a_tset", "hn_eq", "pf_rank", "addr_blank_c", "rank_core"]',
    'ST_F = ["n_tset", "a_tset", "hn_eq", "pf_rank", "addr_blank_c", "rank_core", "cos_bi", "rank_bi"]')
open("v12_src.py", "w", encoding="utf-8").write(s)
print("v12_src.py written", len(s.splitlines()), "lines")
