"""Build v8abl_src.py (-> code/src/v8_france_ablation.ipynb) from v6probe_src.py.

Component ablation on France (the unlabelled domain): save each stage's French probabilities separately:
  p1 = stage-1 matcher, p2 = + group-context model, pb = + cross-encoder blend (= v4).
Variants A (p1), B (p2), C (pb), each at its own validation-tuned global threshold, then the descriptor-swap filter.
US/India rows stay exactly v4. If one stage transfers badly to France, removing it is a large, measurable change.
"""
s = open("v6probe_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


rep("# # Business Entity Resolution — v6 probe: France threshold (controlled leaderboard experiment)",
    "# # Business Entity Resolution — v8: France component ablation (controlled leaderboard experiment)\n#\n"
    "# Saves stage-1 / group-model / blend probabilities for every French pair and writes France variants that stop at\n"
    "# each stage (A: stage-1 only, B: no cross-encoder, C: full = v4), each at its validation-tuned threshold.\n#")
rep('WORK = ROOT / "work" / "v6probe"', 'WORK = ROOT / "work" / "v8abl"')
rep('OUT_MATCH = OUT / "lb_probe" / "matching_results_fr_t0.700.tsv"',
    'OUT_MATCH = OUT / "lb_probe" / "matching_results_v8abl_C_raw.tsv"')
rep("PROBE_T = [0.55, 0.625, 0.70, 0.775, 0.85]", "PROBE_T = []   # replaced by per-stage variants below")
# keep per-stage probabilities
rep('''    out = np.empty(len(Qf), np.float32)''', '''    out = np.empty(len(Qf), np.float32)
    global P1_ALL, P2_ALL
    P1_ALL = np.empty(len(Qf), np.float32)
    P2_ALL = np.empty(len(Qf), np.float32)''')
rep('''        p = predict(m1_model, X, FEATURES)
        p = predict(m2, np.hstack([X, group_ctx(test, q, c, p)]), ALL_FEATURES)''',
    '''        p = predict(m1_model, X, FEATURES)
        P1_ALL[a:b] = p
        p = predict(m2, np.hstack([X, group_ctx(test, q, c, p)]), ALL_FEATURES)
        P2_ALL[a:b] = p''')
rep('''np.savez(WORK / "france_v4_probs.npz", q=Qf, c=Cf, p=p_start, ids=ids_te)''',
    '''np.savez(WORK / "france_stage_probs.npz", q=Qf, c=Cf, p1=P1_ALL, p2=P2_ALL, pb=p_start, ids=ids_te)
STAGE_T = {"A": results["stage-1 matcher"][0], "B": results["group-context model"][0], "C": results["blend"][0]}
log(f"validation-tuned global thresholds per stage: {STAGE_T}")''')
# after the (now empty) threshold loop, write the three stage variants
rep('''v4 = pd.read_csv(OUT_MATCH, sep="\\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)   # the t=0.70 file''',
    '''for _stage, _p in (("A", P1_ALL), ("B", P2_ALL), ("C", p_start)):
    mq_fr, mc_fr = resolve(Qf, Cf, test.src[Cf], _p, np.full(len(Qf), STAGE_T[_stage], dtype=np.float32))
    fr_lists = pd.Series(ids_te[mc_fr]).groupby(ids_te[mq_fr]).agg(",".join)
    out = v4.copy()
    out.loc[is_fr, "matched_entity_ids"] = out.loc[is_fr, "source1_entity_id"].map(fr_lists).fillna("").to_numpy()
    path = OUT / "lb_probe" / f"matching_results_v8abl_{_stage}_raw.tsv"
    out.to_csv(path, sep="\\t", index=False, quoting=csv.QUOTE_NONE)
    log(f"France stage {_stage} (t={STAGE_T[_stage]:.3f}): {len(mq_fr):,} French matches, "
        f"{len(np.unique(mq_fr)):,} French S1 with a match -> {path.name}")
v4 = pd.read_csv(OUT_MATCH, sep="\\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)   # stage C file''')
open("v8abl_src.py", "w", encoding="utf-8").write(s)
print("v8abl_src.py written", len(s.splitlines()), "lines")
