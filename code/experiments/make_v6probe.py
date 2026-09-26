"""Build v6probe_src.py (-> code/src/v6_france_probe.ipynb) from v5_src.py.

A controlled leaderboard probe for the unlabelled country: US/India are fixed at v4; only France's decision threshold
changes. The v4 French probabilities are saved so any further France variant is instant.
"""
s = open("v5_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


head_end = s.index("# %%\nimport os")
s = '''# %% [markdown]
# # Business Entity Resolution — v6 probe: France threshold (controlled leaderboard experiment)
#
# France has no labels, and v5 showed that an unverified France change can hurt (self-training: 0.972 vs v4 0.973, P19).
# The one trustworthy signal for France is the leaderboard. This notebook fixes US/India at v4 and writes France variants
# that differ **only** in France's decision threshold. Comparing their scores tells whether v4's French probabilities are
# too cautious (a lower threshold wins) or too eager (a higher threshold wins).
#
# Also saves `work/v6probe/france_v4_probs.npz` (every French candidate pair with its v4 probability) for fast variants.

''' + s[head_end:]
rep('WORK = ROOT / "work" / "v5"', 'WORK = ROOT / "work" / "v6probe"')
rep('OUT_MATCH = OUT / "matching_results_v5.tsv"', 'OUT_MATCH = OUT / "lb_probe" / "matching_results_fr_t0.700.tsv"')
rep('OUT_CAND = OUT / "candidate_pairs_v5.tsv"', 'OUT_CAND = OUT / "lb_probe" / "candidate_pairs_v4.tsv"')
rep("ST_ROUNDS = 2", "ST_ROUNDS = 0                      # probe: no self-training\nPROBE_T = [0.55, 0.625, 0.70, 0.775, 0.85]")
rep('for p in (WORK, SHARED, TMP, OUT):', 'for p in (WORK, SHARED, TMP, OUT, OUT / "lb_probe"):')
rep('''p_start = p_teacher.copy()
''', '''p_start = p_teacher.copy()
np.savez(WORK / "france_v4_probs.npz", q=Qf, c=Cf, p=p_start, ids=ids_te)
log(f"saved France v4 probabilities for {len(Qf):,} pairs")
''')
rep('''_tz = np.load(PREV / "train_pairs.npz")
_keep_tr = np.random.default_rng(SEED + 5).random(len(_tz["y"])) < ST_TRAIN_KEEP
Xtr_st, ytr_st = _tz["X"][_keep_tr], _tz["y"][_keep_tr]
del _tz
_vz = np.load(PREV / "val_pairs.npz")
Xva_st, yva_st = _vz["X"], _vz["y"]
del _vz
gc.collect()
''', '''Xtr_st = Xva_st = None                              # probe: no retraining, nothing to load
''')
start = s.index("# ---- France decisions (one owner per record, caps, v4 threshold)")
end = s.index("# label-free monitor (as in v4), now after self-training")
s = s[:start] + '''# ---- France variants: only France's threshold changes; US/India rows stay exactly v4
import shutil  # noqa: E402
v4 = pd.read_csv(V4_MATCH, sep="\\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
is_fr = v4.source1_entity_id.isin(set(ids_te[fr_q]))
shutil.copyfile(V4_CAND, OUT_CAND)
for t in PROBE_T:
    mq_fr, mc_fr = resolve(Qf, Cf, test.src[Cf], p_start, np.full(len(Qf), t, dtype=np.float32))
    fr_lists = pd.Series(ids_te[mc_fr]).groupby(ids_te[mq_fr]).agg(",".join)
    out = v4.copy()
    out.loc[is_fr, "matched_entity_ids"] = out.loc[is_fr, "source1_entity_id"].map(fr_lists).fillna("").to_numpy()
    path = OUT / "lb_probe" / f"matching_results_fr_t{t:.3f}.tsv"
    out.to_csv(path, sep="\\t", index=False, quoting=csv.QUOTE_NONE)
    same = (out.matched_entity_ids == v4.matched_entity_ids).mean()
    log(f"France threshold {t:.3f}: {len(mq_fr):,} French matches, {len(np.unique(mq_fr)):,} French S1 with a match; "
        f"rows identical to v4 {same:.4f}  -> {path.name}")
v4 = pd.read_csv(OUT_MATCH, sep="\\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)   # the t=0.70 file
mq = test.id_index.get_indexer(np.repeat(v4.source1_entity_id.to_numpy(),
                                         v4.matched_entity_ids.map(lambda x: len(x.split(",")) if x else 0).to_numpy()))
mc = test.id_index.get_indexer(np.concatenate([x.split(",") for x in v4.matched_entity_ids if x]))

''' + s[end:]
s = s.replace('log("SILVER REPORT v5 (test', 'log("SILVER REPORT probe t=0.70 (test')
open("v6probe_src.py", "w", encoding="utf-8").write(s)
print("v6probe_src.py written", len(s.splitlines()), "lines")
