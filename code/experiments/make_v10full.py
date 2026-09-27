"""Build v10full_src.py (-> code/src/v10_full.ipynb): v10m validation setup (prefilter top-80, multilingual cross-encoder
blend refit on validation) + v4's streaming test inference (scores chunk by chunk; no feature store on disk).
T29: validation hold-out 0.9828 -> 0.9836 with top-80.
"""
s = open("v10m_src.py", encoding="utf-8").read()
v4 = open("v4_src.py", encoding="utf-8").read()


def rep(old, new, count=1, where="s"):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


rep("# # Business Entity Resolution — v10: larger prefilter cut, validation-only experiment",
    "# # Business Entity Resolution — v10: prefilter top-80 + multilingual cross-encoder (full test inference)\n#\n"
    "# Validation hold-out 0.9836 (v9: 0.9828). France is re-scored too, but its rows are only adopted after a leaderboard check.")
rep('WORK = ROOT / "work" / f"v10m{os.environ.get(\'PF_M_OVERRIDE\', \'80\')}"', 'WORK = ROOT / "work" / "v10full"')
rep('OUT_MATCH = OUT / "lb_probe" / "matching_results_v9usin_raw.tsv"', 'OUT_MATCH = OUT / "lb_probe" / "matching_results_v10_raw.tsv"')
rep('OUT_CAND = OUT / "lb_probe" / "candidate_pairs_v4.tsv"', 'OUT_CAND = OUT / "lb_probe" / "candidate_pairs_v10.tsv"')
tail = v4[v4.index("# %% [markdown]\n# ## 13. Test inference"):]
tail = tail.replace("output/candidate_pairs_v4.tsv", "candidate_pairs_v10.tsv").replace("output/matching_results_v4.tsv",
                                                                                          "matching_results_v10_raw.tsv")
end = s.index('# %%\nlog("v10 validation-only experiment done (no test inference)")')
s = s[:end] + tail
open("v10full_src.py", "w", encoding="utf-8").write(s)
print("v10full_src.py written", len(s.splitlines()), "lines")
