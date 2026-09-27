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
# P24: with top-80 a 50k-S1 chunk holds 4M pairs; pickling their strings to the joblib workers exhausted RAM at 1.0M S1.
# Test chunks of 25k S1 (~2M pairs, like v4) + a disk checkpoint per chunk so a crash/restart resumes instead of restarting.
old_loop = '''kept, n_cands = [], 0
with open(OUT_CAND, "w", encoding="utf-8", newline="") as fcand:
    fcand.write("source1_entity_id\\tcandidate_entity_ids\\n")
    for a in range(0, len(test_q), STAGE_CHUNK):
        qch = test_q[a:a + STAGE_CHUNK]
        q, c, X = candidates_and_features(test, qch, test_q, KNN_TE, EX_TE)'''
new_loop = '''kept, n_cands = [], 0
TEST_CHUNK = 25_000                                   # P24: ~2M candidate pairs per chunk at top-80
CKPT = WORK / "test_chunks"
CKPT.mkdir(exist_ok=True)
for a in range(0, len(test_q), TEST_CHUNK):
    qch = test_q[a:a + TEST_CHUNK]
    ck_k, ck_c = CKPT / f"kept_{a:08d}.parquet", CKPT / f"cand_{a:08d}.tsv"
    if ck_k.exists() and ck_c.exists():               # resume: chunk already scored
        kept.append(pd.read_parquet(ck_k))
        continue
    if True:
        q, c, X = candidates_and_features(test, qch, test_q, KNN_TE, EX_TE)'''
assert s.count(old_loop) == 1
s = s.replace(old_loop, new_loop)
old_tail = '''        kept.append(pd.DataFrame({"q": q[sel], "c": c[sel], "p": p[sel], "t": thr[sel]}))
        # candidate list = the exact set scored by the matcher
        lists = pd.Series(ids_te[c]).groupby(q).agg(",".join)
        col = pd.Series(qch).map(lists).fillna("")
        pd.DataFrame({"a": ids_te[qch], "b": col.to_numpy()}).to_csv(
            fcand, sep="\\t", index=False, header=False, quoting=csv.QUOTE_NONE)
        n_cands += len(q)
        log(f"  test scored {min(a + STAGE_CHUNK, len(test_q)):,}/{len(test_q):,} S1 ({n_cands:,} candidate pairs)")
log(f"wrote {OUT_CAND} ({len(test_q):,} rows, {n_cands / len(test_q):.1f} candidates per S1)")'''
new_tail = '''        kc = pd.DataFrame({"q": q[sel], "c": c[sel], "p": p[sel], "t": thr[sel]})
        lists = pd.Series(ids_te[c]).groupby(q).agg(",".join)   # candidate list = the exact set scored
        col = pd.Series(qch).map(lists).fillna("")
        pd.DataFrame({"a": ids_te[qch], "b": col.to_numpy()}).to_csv(ck_c, sep="\\t", index=False, header=False,
                                                                     quoting=csv.QUOTE_NONE)
        kc.to_parquet(ck_k)                          # written last: a chunk counts as done only when both exist
        kept.append(kc)
        del q, c, X, p
        gc.collect()
        log(f"  test scored {min(a + TEST_CHUNK, len(test_q)):,}/{len(test_q):,} S1")
with open(OUT_CAND, "w", encoding="utf-8", newline="") as fcand:   # assemble the candidate file from the chunks
    fcand.write("source1_entity_id\\tcandidate_entity_ids\\n")
    for a in range(0, len(test_q), TEST_CHUNK):
        with open(CKPT / f"cand_{a:08d}.tsv", encoding="utf-8") as fc:
            n_cands += sum(len(l.rstrip("\\n").split("\\t")[1].split(",")) for l in fc if l.rstrip("\\n").split("\\t")[1])
            fc.seek(0)
            fcand.write(fc.read())
log(f"wrote {OUT_CAND} ({len(test_q):,} rows, {n_cands / len(test_q):.1f} candidates per S1)")'''
assert s.count(old_tail) == 1
s = s.replace(old_tail, new_tail)
open("v10full_src.py", "w", encoding="utf-8").write(s)
print("v10full_src.py written", len(s.splitlines()), "lines")
