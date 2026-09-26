"""Build v7fr_src.py (-> code/src/v7_france_address.ipynb) from v6probe_src.py.

France-only address normalization for pair features (P21): French anchor pairs (same name + house number + street) have
address similarity 77 vs 88-90 elsewhere, because one source writes the REGION (Hauts-de-France) and the other the
DEPARTEMENT (Nord), plus N° markers and St/Saint variants. The model learned "true pairs have ~98% address similarity", so
true French pairs look weaker than they are. Blocking is unchanged; only French pair features / cross-encoder inputs use
the cleaned addresses. US/India rows stay exactly as in v4.
"""
s = open("v6probe_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


rep("# # Business Entity Resolution — v6 probe: France threshold (controlled leaderboard experiment)",
    "# # Business Entity Resolution — v7: France address normalization (controlled leaderboard experiment)\n#\n"
    "# French pair features use addresses without administrative areas (region / departement: one source writes\n"
    "# `Hauts-de-France`, the other `Nord`), without `N°` markers, and with `St`/`Ste` read as `Saint`/`Sainte`.\n#")
rep('WORK = ROOT / "work" / "v6probe"', 'WORK = ROOT / "work" / "v7fr"')
rep('OUT_MATCH = OUT / "lb_probe" / "matching_results_fr_t0.700.tsv"',
    'OUT_MATCH = OUT / "lb_probe" / "matching_results_v7fr_raw_t0.700.tsv"')
rep("PROBE_T = [0.55, 0.625, 0.70, 0.775, 0.85]", "PROBE_T = [0.70]")
rep('    path = OUT / "lb_probe" / f"matching_results_fr_t{t:.3f}.tsv"',
    '    path = OUT / "lb_probe" / f"matching_results_v7fr_raw_t{t:.3f}.tsv"')
rep('np.savez(WORK / "france_v4_probs.npz", q=Qf, c=Cf, p=p_start, ids=ids_te)',
    'np.savez(WORK / "france_v7fr_probs.npz", q=Qf, c=Cf, p=p_start, ids=ids_te)')
rep('''fr_q = test_q[q_ctry == "France"]
''', '''fr_q = test_q[q_ctry == "France"]
# v7: French address normalization for pair features (administrative areas, N° markers, St/Saint)
_FR_ADMIN = re.compile(r"\\b(hauts de france|nouvelle aquitaine|pays de la loire|loire atlantique|pas de calais|gironde)\\b"
                       r"|(?<!\\bdu )(?<!\\bde )\\bnord\\b")
_FR_TOK = {"st": "saint", "ste": "sainte", "no": "", "ndeg": "", "n": ""}


def _fr_clean(a):
    a = _FR_ADMIN.sub(" ", a)
    return " ".join(t2 for t2 in (_FR_TOK.get(t, t) for t in a.split()) if t2)


_is_fr = te_ctry[test.country_codes] == "France"
_addr = test.tbl["addr"].to_pylist()
_n_changed = 0
for _i in np.flatnonzero(_is_fr):
    _new = _fr_clean(_addr[_i])
    _n_changed += _new != _addr[_i]
    _addr[_i] = _new
test.tbl = test.tbl.set_column(test.tbl.column_names.index("addr"), "addr", pa.array(_addr))
log(f"French address normalization: {_n_changed:,} of {int(_is_fr.sum()):,} French addresses changed; "
    f"e.g. {_fr_clean('53 r de cartigny roubaix hauts de france')!r}, {_fr_clean('no 15 blvd alexandre iii dunkerque nord')!r}")
del _addr
''')
open("v7fr_src.py", "w", encoding="utf-8").write(s)
print("v7fr_src.py written", len(s.splitlines()), "lines")
