"""Build v10m_src.py (-> code/src/v10_prefilter_m.ipynb) from v9usin_src.py: VALIDATION-ONLY test of a larger prefilter
cut (Sparkly: top-k size is the main recall lever). 14% of blocking misses are retrieved by stage 1 but cut by the top-40
prefilter. Validation candidates are rebuilt with PF_M from the env var PF_M_OVERRIDE and scored with the existing models
(no retraining); the notebook stops before test inference.
"""
s = open("v9usin_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


rep("# # Business Entity Resolution — v9: multilingual cross-encoder for the US and India (validated)",
    "# # Business Entity Resolution — v10: larger prefilter cut, validation-only experiment")
rep('WORK = ROOT / "work" / "v9usin"', 'WORK = ROOT / "work" / f"v10m{os.environ.get(\'PF_M_OVERRIDE\', \'80\')}"')
rep('PF_M = int(re.search(r"prefilter keeps top (\\d+)", (PREV / "run.log").read_text(encoding="utf-8", errors="replace")).group(1))',
    'PF_M = int(os.environ.get("PF_M_OVERRIDE", "80"))   # v10: larger prefilter cut (v4 used 40)')
rep('''va_q, va_c, va_X, va_y = build_pairs("val", VAL_Q)''',
    '''_prev_cache = PREV
PREV = WORK                                   # v10: rebuild VALIDATION pairs with the new cut
va_q, va_c, va_X, va_y = build_pairs("val", VAL_Q)
PREV = _prev_cache                            # training pairs: v4 cache (models are not retrained)''')
end = s.index("# %% [markdown]\n# ## 13. Test inference")
s = s[:end] + '''# %%
log("v10 validation-only experiment done (no test inference)")
'''
open("v10m_src.py", "w", encoding="utf-8").write(s)
print("v10m_src.py written", len(s.splitlines()), "lines")
