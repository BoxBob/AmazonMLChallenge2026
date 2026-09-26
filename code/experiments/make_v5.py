"""Build v5_src.py (-> code/src/v5.ipynb) from v4_src.py.

v5 = v4 + self-training on France (T15: leave-one-country-out showed an unseen country costs ~0.039 and that only
self-training on confident target pseudo-labels transfers). v4's cached pairs, prefilter and models are reused; the
test section re-scores France only, then merges with v4's US/India predictions.
"""
s = open("v4_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


head_end = s.index("# %%\nimport os")
s = '''# %% [markdown]
# # Business Entity Resolution — v5: self-training for the unlabelled country (France)
#
# **Why:** a leave-one-country-out experiment (train on the US, test on India, the DADER evaluation protocol) showed an
# unseen country costs ~0.039 macro F0.5 even with correct blocking, because what similarity patterns *mean* differs by
# country. Of four adaptation methods, only **self-training on confident pseudo-labels** transferred
# (0.9293 → 0.9390; Saerens prior correction +0; per-country quantile alignment −0.045).
#
# **What:** v4 (teacher: stage-1 → group model → cross-encoder blend) scores every French candidate pair. Pairs with
# p ≥ 0.97 become positives and p ≤ 0.03 negatives. The stage-1 matcher is retrained on the training pairs plus these French
# pseudo-labels and France is re-scored (2 rounds). US and India keep v4's predictions: their test behaviour already
# matches validation (silver-label acceptance).
#
# Reuses `work/v4` (pairs, prefilter, models) and `work/v3/ce_model`. Outputs `output/*_v5.tsv`.

''' + s[head_end:]
rep('WORK = ROOT / "work" / "v4"', 'WORK = ROOT / "work" / "v5"\nPREV = ROOT / "work" / "v4"      # v4 artifacts reused by v5')
rep('OUT_CAND = OUT / "candidate_pairs_v4.tsv"', 'OUT_CAND = OUT / "candidate_pairs_v5.tsv"')
rep('OUT_MATCH = OUT / "matching_results_v4.tsv"', 'OUT_MATCH = OUT / "matching_results_v5.tsv"\n'
    'V4_MATCH = OUT / "matching_results_v4.tsv"\nV4_CAND = OUT / "candidate_pairs_v4.tsv"\n'
    'ST_POS, ST_NEG = 0.97, 0.03      # self-training confidence cuts\nST_ROUNDS = 2\nST_NEG_KEEP = 0.25   # share of confident French negatives kept (memory)\nST_TRAIN_KEEP = 0.6  # share of training rows kept next to the pseudo-labels (memory)')
rep('_bc = WORK / "train_knn_blank.npy"', '_bc = PREV / "train_knn_blank.npy"')
rep('_bc = WORK / "test_knn_blank.npy"', '_bc = PREV / "test_knn_blank.npy"')
rep('    cache = WORK / f"{name}_pairs.npz"', '    cache = PREV / f"{name}_pairs.npz"')

# ---- section 7: reuse the v4 prefilter (skip the recall loop and the prefilter fit)
start = s.index("# stage-1 recall on validation (before prefilter)")
end = s.index('log(f"prefilter keeps top {PF_M} candidates per S1")')
s = s[:start] + '''# v5: reuse the v4 prefilter model and its cut M (read from the v4 run log)
pf_model = xgb.Booster(model_file=str(PREV / "prefilter.json"))
PF_M = int(re.search(r"prefilter keeps top (\\d+)", (PREV / "run.log").read_text(encoding="utf-8")).group(1))
''' + s[end:]
rep('''del Xp, yp, Xe, ye, qe, pe
gc.collect()''', 'gc.collect()')

# ---- sections 10/11: load v4 m1 / m2 instead of training
start = s.index("fold = (np.random.default_rng(SEED + 2)")
end = s.index('log(f"stage-1 matcher best iteration {m1.best_iteration}")')
s = s[:start] + '''m1 = xgb.Booster(model_file=str(PREV / "m1.json"))      # v5: the v4 stage-1 matcher
p1_va = predict(m1, va_X, FEATURES)
''' + s[end:]
rep('''Gtr = group_ctx(train, tr_q, tr_c, p_oof)
Gva = group_ctx(train, va_q, va_c, p1_va)
m2 = fit(np.hstack([tr_X, Gtr]), tr_y, np.hstack([va_X, Gva]), va_y, ALL_FEATURES)
m2.save_model(WORK / "m2.json")''', '''Gva = group_ctx(train, va_q, va_c, p1_va)
m2 = xgb.Booster(model_file=str(PREV / "m2.json"))      # v5: the v4 group-context model''')
rep('''del Gtr
gc.collect()''', 'gc.collect()')

# ---- section 13: France-only self-training, merged with v4 for US / India
start = s.index("ids_te = test.tbl[\"entity_id\"].to_numpy(zero_copy_only=False)")
end = s.index("# %%\nres = subprocess.run(")
s = s[:start] + '''ids_te = test.tbl["entity_id"].to_numpy(zero_copy_only=False)
te_ctry = np.asarray(test.countries)
q_ctry = te_ctry[test.country_codes[test_q]]
fr_q = test_q[q_ctry == "France"]
assert V4_MATCH.exists(), "v5 needs the v4 test output (run v4.ipynb first)"
del train          # not needed for France self-training; frees RAM for the feature matrices
gc.collect()

# ---- pass 1: features + teacher (v4 stack) probabilities for every French candidate pair; features go to a disk memmap
Qf, Cf, n_rows = [], [], 0
xfile = WORK / "fr_X.npy"
chunks = []
for a in range(0, len(fr_q), STAGE_CHUNK):
    q, c, X = candidates_and_features(test, fr_q[a:a + STAGE_CHUNK], test_q, KNN_TE, EX_TE)
    np.save(WORK / f"fr_X_{a}.npy", X)
    chunks.append(WORK / f"fr_X_{a}.npy")
    Qf.append(q); Cf.append(c)
    log(f"  France features {min(a + STAGE_CHUNK, len(fr_q)):,}/{len(fr_q):,} S1")
Qf, Cf = np.concatenate(Qf), np.concatenate(Cf)
Xf = np.lib.format.open_memmap(xfile, mode="w+", dtype=np.float32, shape=(len(Qf), len(FEATURES)))
o = 0
for f_ in chunks:
    x = np.load(f_)
    Xf[o:o + len(x)] = x
    o += len(x)
    f_.unlink()
Xf.flush()
log(f"France candidate pairs: {len(Qf):,} ({len(Qf) / len(fr_q):.1f} per S1)")

PCE = np.full(len(Qf), np.nan, dtype=np.float32)   # cross-encoder scores, computed lazily for band pairs


def full_stack(m1_model):
    """stage-1 -> group-context model -> cross-encoder blend, over all French pairs (chunked by entity)"""
    out = np.empty(len(Qf), np.float32)
    bounds = list(np.searchsorted(Qf, fr_q[::STAGE_CHUNK])) + [len(Qf)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        X = np.asarray(Xf[a:b])
        q, c = Qf[a:b], Cf[a:b]
        p = predict(m1_model, X, FEATURES)
        p = predict(m2, np.hstack([X, group_ctx(test, q, c, p)]), ALL_FEATURES)
        band = np.flatnonzero((p >= CE_BAND[0]) & (p <= CE_BAND[1]))
        todo = band[np.isnan(PCE[a + band])]
        if len(todo):
            PCE[a + todo] = ce_score(test, q[todo], c[todo])
        p = p.copy()
        p[band] = blend.predict_proba(np.column_stack([_logit(p[band]), _logit(PCE[a + band])]))[:, 1]
        out[a:b] = p
    return out


p_teacher = full_stack(m1)
p_start = p_teacher.copy()

# training rows (subsampled for memory) next to the pseudo-labels
_tz = np.load(PREV / "train_pairs.npz")
_keep_tr = np.random.default_rng(SEED + 5).random(len(_tz["y"])) < ST_TRAIN_KEEP
Xtr_st, ytr_st = _tz["X"][_keep_tr], _tz["y"][_keep_tr]
del _tz
_vz = np.load(PREV / "val_pairs.npz")
Xva_st, yva_st = _vz["X"], _vz["y"]
del _vz
gc.collect()

m1_st = m1
for rnd in range(1, ST_ROUNDS + 1):
    pos = p_teacher >= ST_POS
    neg = (p_teacher <= ST_NEG) & (np.random.default_rng(SEED + rnd).random(len(p_teacher)) < ST_NEG_KEEP)
    sel = np.flatnonzero(pos | neg)
    log(f"self-training round {rnd}: pseudo-positives {pos.sum():,}, pseudo-negatives kept {neg.sum():,} "
        f"(confident share {((p_teacher >= ST_POS) | (p_teacher <= ST_NEG)).mean():.1%})")
    X_aug = np.vstack([Xtr_st, np.asarray(Xf[sel])])
    y_aug = np.concatenate([ytr_st, pos[sel].astype(np.int8)])
    m1_st = fit(X_aug, y_aug, Xva_st, yva_st, FEATURES)
    del X_aug, y_aug
    gc.collect()
    m1_st.save_model(WORK / f"m1_selftrain_r{rnd}.json")
    p_new = full_stack(m1_st)
    log(f"  round {rnd}: student best iteration {m1_st.best_iteration}; France pairs above threshold "
        f"{(p_teacher >= T_DEFAULT).sum():,} -> {(p_new >= T_DEFAULT).sum():,}; "
        f"decision flips {((p_teacher >= T_DEFAULT) != (p_new >= T_DEFAULT)).sum():,}")
    p_teacher = p_new
Xf._mmap.close()
del Xf, Xtr_st, Xva_st
gc.collect()
xfile.unlink()

# ---- France decisions (one owner per record, caps, v4 threshold)
thr_fr = np.full(len(Qf), T_DEFAULT, dtype=np.float32)
mq_fr, mc_fr = resolve(Qf, Cf, test.src[Cf], p_teacher, thr_fr)
mq_v4, mc_v4 = resolve(Qf, Cf, test.src[Cf], p_start, thr_fr)
log(f"France matches: v4 {len(mq_v4):,} ({len(np.unique(mq_v4)):,} S1) -> v5 {len(mq_fr):,} ({len(np.unique(mq_fr)):,} S1)")

# ---- merge: v4 rows for US/India, v5 rows for France
v4 = pd.read_csv(V4_MATCH, sep="\\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
fr_lists = pd.Series(ids_te[mc_fr]).groupby(ids_te[mq_fr]).agg(",".join)
is_fr = v4.source1_entity_id.isin(set(ids_te[fr_q]))
v4.loc[is_fr, "matched_entity_ids"] = v4.loc[is_fr, "source1_entity_id"].map(fr_lists).fillna("").to_numpy()
v4.to_csv(OUT_MATCH, sep="\\t", index=False, quoting=csv.QUOTE_NONE)
import shutil  # noqa: E402
shutil.copyfile(V4_CAND, OUT_CAND)                  # identical blocking -> identical candidate sets
mq = test.id_index.get_indexer(np.repeat(v4.source1_entity_id.to_numpy(),
                                         v4.matched_entity_ids.map(lambda x: len(x.split(",")) if x else 0).to_numpy()))
mc = test.id_index.get_indexer(np.concatenate([x.split(",") for x in v4.matched_entity_ids if x]))
log(f"wrote {OUT_MATCH}: {len(mq):,} matches; candidates copied from v4 ({OUT_CAND.name})")

# label-free monitor (as in v4), now after self-training
_sk = _key(test, "ex_name_hn").to_numpy(zero_copy_only=False)
_s1k = pd.Series(_sk[test_q])
_uniq = _s1k[_s1k != ""].value_counts()
_sil = pd.DataFrame({"k": _sk[test_q], "q": test_q}).merge(
    pd.DataFrame({"k": _sk[test.src != 1], "c": np.flatnonzero(test.src != 1)}), on="k")
_sil = _sil[(_sil.k != "") & (_sil.k.map(_uniq) == 1)]
_pred = set((mq.astype(np.int64) * test.n + mc).tolist())
_sil["acc"] = [k in _pred for k in (_sil.q.to_numpy(np.int64) * test.n + _sil.c.to_numpy()).tolist()]
_sil["country"] = te_ctry[test.country_codes[_sil.q.to_numpy()]]
log("SILVER REPORT v5 (test, unique name+number pairs predicted as matches): "
    + str(_sil.groupby("country").acc.mean().round(4).to_dict()))

''' + s[end:]
open("v5_src.py", "w", encoding="utf-8").write(s)
print("v5_src.py written", len(s.splitlines()), "lines")
