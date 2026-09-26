"""Build v9usin_src.py (-> code/src/v9_usin_mce.ipynb) from v9fr_src.py.

Same as v9fr (multilingual cross-encoder, blend refit on validation, validation-tuned threshold) but for the US and
India entities instead of France: these countries are measured by validation, where the multilingual cross-encoder is
better (blend hold-out 0.9826 -> 0.9828). France rows are left as in the base file.
"""
s = open("v9fr_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


rep("# # Business Entity Resolution — v9: multilingual cross-encoder for France (controlled leaderboard experiment)",
    "# # Business Entity Resolution — v9: multilingual cross-encoder for the US and India (validated)")
rep('WORK = ROOT / "work" / "v9fr"', 'WORK = ROOT / "work" / "v9usin"')
rep('OUT_MATCH = OUT / "lb_probe" / "matching_results_v9fr_raw.tsv"',
    'OUT_MATCH = OUT / "lb_probe" / "matching_results_v9usin_raw.tsv"')
rep('fr_q = test_q[q_ctry == "France"]', 'fr_q = test_q[q_ctry != "France"]          # v9usin: US + India entities')
rep('np.savez(WORK / "france_v9_probs.npz", q=Qf, c=Cf, p=p_start, ids=ids_te)',
    'np.savez(WORK / "usin_v9_probs.npz", q=Qf, c=Cf, p=p_start, ids=ids_te)')
# the per-country thresholds (US/India) apply here, as in v4
rep("    mq_fr, mc_fr = resolve(Qf, Cf, test.src[Cf], p_start, np.full(len(Qf), t, dtype=np.float32))",
    "    _thr = np.array([THRESH.get(x, T_DEFAULT) for x in te_ctry[test.country_codes[Qf]]], dtype=np.float32)\n"
    "    mq_fr, mc_fr = resolve(Qf, Cf, test.src[Cf], p_start, _thr)")
# disk: write features straight into one preallocated memmap (US+India ~59M pairs ~22 GB; the chunk-then-merge path
# of the France notebooks would need ~44 GB at peak, more than is free on D:)
rep('''Qf, Cf, n_rows = [], [], 0
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
Xf.flush()''', '''Qf, Cf = [], []
xfile = WORK / "fr_X.npy"
_Xfull = np.lib.format.open_memmap(xfile, mode="w+", dtype=np.float32, shape=(len(fr_q) * PF_M, len(FEATURES)))
o = 0
for a in range(0, len(fr_q), STAGE_CHUNK):
    q, c, X = candidates_and_features(test, fr_q[a:a + STAGE_CHUNK], test_q, KNN_TE, EX_TE)
    _Xfull[o:o + len(X)] = X
    o += len(X)
    Qf.append(q); Cf.append(c)
    log(f"  US/India features {min(a + STAGE_CHUNK, len(fr_q)):,}/{len(fr_q):,} S1")
Qf, Cf = np.concatenate(Qf), np.concatenate(Cf)
_Xfull.flush()
Xf = _Xfull[:o]''')
rep("Xf._mmap.close()", "_Xfull._mmap.close()")
rep("del Xf, Xtr_st, Xva_st", "del Xf, _Xfull, Xtr_st, Xva_st")
open("v9usin_src.py", "w", encoding="utf-8").write(s)
print("v9usin_src.py written", len(s.splitlines()), "lines")
