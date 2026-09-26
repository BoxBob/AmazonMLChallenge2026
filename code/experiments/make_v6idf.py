"""Build v6idf_src.py (-> code/src/v6_france_idf.ipynb) from v6probe_src.py.

Controlled France-only variant: French pair features use token IDF computed **within France** (Splink-style term
frequency within the population being linked) instead of over the whole test split, where templated French words look
~1.9 nats rarer than they are among French records. US/India rows stay exactly v4.
"""
s = open("v6probe_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


rep("# # Business Entity Resolution — v6 probe: France threshold (controlled leaderboard experiment)",
    "# # Business Entity Resolution — v6: per-country IDF for France (controlled leaderboard experiment)\n#\n"
    "# Variant of the France probe: French features use IDF computed among French records only (term frequency within the\n"
    "# population being linked, as in Splink). Everything else is v4. Output: `output/lb_probe/matching_results_fr_idf_t*.tsv`.\n#")
rep('WORK = ROOT / "work" / "v6probe"', 'WORK = ROOT / "work" / "v6idf"')
rep('OUT_MATCH = OUT / "lb_probe" / "matching_results_fr_t0.700.tsv"',
    'OUT_MATCH = OUT / "lb_probe" / "matching_results_fr_idf_t0.700.tsv"')
rep("PROBE_T = [0.55, 0.625, 0.70, 0.775, 0.85]", "PROBE_T = [0.625, 0.70, 0.775]")
rep('    path = OUT / "lb_probe" / f"matching_results_fr_t{t:.3f}.tsv"',
    '    path = OUT / "lb_probe" / f"matching_results_fr_idf_t{t:.3f}.tsv"')
rep('np.savez(WORK / "france_v4_probs.npz", q=Qf, c=Cf, p=p_start, ids=ids_te)',
    'np.savez(WORK / "france_idf_probs.npz", q=Qf, c=Cf, p=p_start, ids=ids_te)')
rep('''fr_q = test_q[q_ctry == "France"]
''', '''fr_q = test_q[q_ctry == "France"]
# v6: IDF within France for French features (Splink-style term frequency within the linked population)
_fr_rows = pa.array(te_ctry[test.country_codes] == "France")
for _col in ("core", "addr"):
    _toks = pac.list_flatten(pac.utf8_split_whitespace(pac.filter(test.tbl[_col], _fr_rows)))
    _vc = pac.value_counts(_toks)
    _n_fr = int(pac.sum(pac.cast(_fr_rows, pa.int64())).as_py())
    _cnts = _vc.field("counts").to_numpy()
    _w = np.log(_n_fr / _cnts)
    _before = IDF[("test", _col)]
    IDF[("test", _col)] = ({v: float(x) for v, x, c in zip(_vc.field("values").to_pylist(), _w, _cnts) if c >= 2},
                           float(math.log(_n_fr)))
    _ex = [t for t in ("club", "lille", "sarl", "gestion", "roubaix", "rue") if t in IDF[("test", _col)][0]][:4]
    log(f"France-only IDF for {_col}: {len(IDF[('test', _col)][0]):,} tokens; e.g. "
        + ", ".join(f"{t}: {_before[0].get(t, _before[1]):.2f} -> {IDF[('test', _col)][0][t]:.2f}" for t in _ex))
''')
open("v6idf_src.py", "w", encoding="utf-8").write(s)
print("v6idf_src.py written", len(s.splitlines()), "lines")
