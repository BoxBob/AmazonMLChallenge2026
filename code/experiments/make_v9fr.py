"""Build v9fr_src.py (-> code/src/v9_france_mce.ipynb) from v6probe_src.py.

France-only variant with the multilingual cross-encoder (T26: validation band AUC 0.9400 -> 0.9481, blend hold-out
0.9826 -> 0.9828). The blend is refit on validation with the new cross-encoder and the validation-tuned threshold is used.
US/India rows stay exactly v4; the descriptor-swap filter is applied afterwards (run script post-step).
"""
s = open("v6probe_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


rep("# # Business Entity Resolution — v6 probe: France threshold (controlled leaderboard experiment)",
    "# # Business Entity Resolution — v9: multilingual cross-encoder for France (controlled leaderboard experiment)\n#\n"
    "# The cross-encoder carries France (T25) but was English-only. Here French pairs are re-scored with the multilingual\n"
    "# `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0), fine-tuned with the identical recipe (work/v9/mce_model).\n#")
rep('WORK = ROOT / "work" / "v6probe"', 'WORK = ROOT / "work" / "v9fr"')
rep('CE_DIR = ROOT / "work" / "v3" / "ce_model"   # the v3 cross-encoder is reused',
    'CE_DIR = ROOT / "work" / "v9" / "mce_model"   # v9: multilingual cross-encoder')
rep('OUT_MATCH = OUT / "lb_probe" / "matching_results_fr_t0.700.tsv"',
    'OUT_MATCH = OUT / "lb_probe" / "matching_results_v9fr_raw.tsv"')
rep("for t in PROBE_T:", "for t in [T_DEFAULT]:          # v9: the validation-tuned threshold of the refit blend")
rep('    path = OUT / "lb_probe" / f"matching_results_fr_t{t:.3f}.tsv"',
    '    path = OUT_MATCH')
rep('np.savez(WORK / "france_v4_probs.npz", q=Qf, c=Cf, p=p_start, ids=ids_te)',
    'np.savez(WORK / "france_v9_probs.npz", q=Qf, c=Cf, p=p_start, ids=ids_te)')
open("v9fr_src.py", "w", encoding="utf-8").write(s)
print("v9fr_src.py written", len(s.splitlines()), "lines")
