"""Build v4_src.py (-> code/src/v4.ipynb) from v3_src.py with explicit, checked patches.

v4 fixes pitfall P14 (France blocking: templated names start with the city, so first-token keys and the kNN name channel
are swamped) and adds the E1 S1-side ambiguity features (P11). It also prints a label-free silver-label report per
country so that a failure in an unlabelled country cannot stay hidden again.
"""
s = open("v3_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


# ---------------------------------------------------------------- header / paths
head_end = s.index("# %%\nimport os")
s = '''# %% [markdown]
# # Business Entity Resolution — v4 (GPU)
#
# v4 = v3 + fixes for the gap between validation (0.981) and the hidden test (0.965).
#
# **Diagnosis (see `VERSION_FINDINGS.md`, P14):** the test is 15% France, which has no labels. French names are
# templated `{City} {Word} {LegalForm}` (`Roubaix Gestion SARL`), so (a) the name kNN is swamped by dozens of identical
# names, (b) the "first token + house number" key becomes `roubaix|53` and overflows its block cap, and (c) the
# "house number + next token" key becomes `53 r` (r = rue). 85% of French same-name-same-number pairs that v3 missed were
# never even candidates. Validation (US/India) cannot show this because there the first token is distinctive.
#
# | Change | Why (first principles) |
# |---|---|
# | key `ex_name_hn` = country + full name + house number | what identifies a business is the *whole* name at a *number*, in any language |
# | key `ex_hn_street` = house number + first distinctive street word (skips street types/articles in all languages) | `53 r de cartigny` → `53 cartigny` |
# | S1-side ambiguity features (E1, +0.0011 on validation) | a name shared by many S1 entities is weak evidence |
# | label-free silver report per country | unlabelled countries must be monitored, not assumed |
#
# Shared caches: `work/v2` (normalization, embeddings, kNN), `work/v3/ce_model` (cross-encoder).
# Outputs: `output/candidate_pairs_v4.tsv`, `output/matching_results_v4.tsv`.

''' + s[head_end:]
rep('WORK = ROOT / "work" / "v3"', 'WORK = ROOT / "work" / "v4"')
rep('OUT_CAND = OUT / "candidate_pairs_v3.tsv"', 'OUT_CAND = OUT / "candidate_pairs_v4.tsv"')
rep('OUT_MATCH = OUT / "matching_results_v3.tsv"', 'OUT_MATCH = OUT / "matching_results_v4.tsv"')
rep('CE_DIR = WORK / "ce_model"', 'CE_DIR = ROOT / "work" / "v3" / "ce_model"   # the v3 cross-encoder is reused')
rep('## 13. Test inference → `output/candidate_pairs_v3.tsv`, `output/matching_results_v3.tsv`',
    '## 13. Test inference → `output/candidate_pairs_v4.tsv`, `output/matching_results_v4.tsv`')

# ---------------------------------------------------------------- Records: hn_street column + S1-side ambiguity counts
rep('''        self.native = tbl["native"].to_numpy(zero_copy_only=False)
        self._counts()''', '''        self.native = tbl["native"].to_numpy(zero_copy_only=False)
        self._counts()
        self._v4_columns()''')
rep('''    def col(self, name):
        return self.tbl[name]''', '''    def _v4_columns(self):
        """v4: house number + first distinctive street word, and S1-side ambiguity counts"""
        hs = Parallel(n_jobs=N_JOBS)(delayed(_hn_street_chunk)(a, p) for a, p in zip(
            _chunks(self.tbl["addr"].to_pylist(), 500_000), _chunks(self.tbl["pc"].to_pylist(), 500_000)))
        self.tbl = self.tbl.append_column("hn_street", pa.array([x for h in hs for x in h]))
        s1 = self.src == 1
        for col, attr, minlen in (("squash", "s1_name", 3), ("addr", "s1_addr", 1)):
            key = pac.binary_join_element_wise(self.tbl["country"], self.tbl[col], "|")
            codes = pac.dictionary_encode(key).combine_chunks().indices.to_numpy()
            blank = pac.less(pac.utf8_length(self.tbl[col]), minlen).to_numpy(zero_copy_only=False)
            n_s1 = np.bincount(codes[s1], minlength=codes.max() + 1)
            n_pool = np.bincount(codes[~s1], minlength=codes.max() + 1)
            setattr(self, attr + "_cnt", np.where(blank, 0, n_s1[codes]).astype(np.float32))
            setattr(self, attr.replace("s1_", "pool_") + "_cnt", np.where(blank, 0, n_pool[codes]).astype(np.float32))

    def col(self, name):
        return self.tbl[name]''')
rep('''class Records:''', '''STREET_STOP = {"r", "rue", "st", "str", "ave", "blvd", "rd", "dr", "ln", "pl", "ct", "hwy", "ste", "apt", "fl", "bldg",
               "ch", "all", "imp", "rte", "fg", "qu", "res", "bat", "no", "n", "s", "e", "w", "ne", "nw", "se", "sw",
               "de", "du", "des", "la", "le", "les", "l", "d", "of", "the", "and", "sq", "pkwy", "cir", "ter", "ctr",
               "jn", "nr", "opp", "sec", "twp", "unit", "road", "street", "avenue", "lane", "cross", "main", "floor",
               "plot", "flat", "house", "door", "shop", "gali", "marg", "nagar", "chemin", "allee", "impasse",
               "boulevard", "place", "quai", "route", "cours", "lieu", "dit", "po", "box", "suite"}


def _hn_street_chunk(addrs, pcs):
    """house number + first distinctive street word; street types and articles of any language are skipped"""
    out = []
    for addr, pc in zip(addrs, pcs):
        key = ""
        toks = addr.split()
        for i, t in enumerate(toks):
            if t[0].isdigit() and t != pc:
                for u in toks[i + 1:i + 6]:
                    if u.isalpha() and len(u) >= 3 and u not in STREET_STOP:
                        key = t + " " + u
                        break
                break
        out.append(key)
    return out


def _chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


class Records:''')

# ---------------------------------------------------------------- new exact keys
rep('''EXACT_KEYS = ("ex_squash", "ex_pc_first", "ex_first_hn", "ex_first2", "ex_cnum", "ex_hn_next", "ex_blank_squash")''',
    '''EXACT_KEYS = ("ex_squash", "ex_pc_first", "ex_first_hn", "ex_first2", "ex_cnum", "ex_hn_next", "ex_blank_squash",
              "ex_name_hn", "ex_hn_street")''')
rep('''    pairs = {"ex_pc_first": ("pc", "first"), "ex_first_hn": ("first", "hn"), "ex_first2": ("first2",),
             "ex_cnum": ("cnum",), "ex_hn_next": ("hn_next",)}[name]''',
    '''    if name == "ex_name_hn":                      # v4: full name + house number (language independent)
        ok = pac.and_(pac.greater_equal(pac.utf8_length(c("squash")), 4), pac.not_equal(c("hn"), ""))
        return pac.if_else(ok, pac.binary_join_element_wise(c("country"), c("squash"), c("hn"), "|"), "")
    pairs = {"ex_pc_first": ("pc", "first"), "ex_first_hn": ("first", "hn"), "ex_first2": ("first2",),
             "ex_cnum": ("cnum",), "ex_hn_next": ("hn_next",), "ex_hn_street": ("hn_street",)}[name]''')

# ---------------------------------------------------------------- E1 ambiguity features
rep('''    "cs_addr_rare_overlap", "cs_addr_unique_c", "cs_name_rare_overlap", "cs_name_unique_c", "cs_disc_match",
]''', '''    "cs_addr_rare_overlap", "cs_addr_unique_c", "cs_name_rare_overlap", "cs_name_unique_c", "cs_disc_match",
    # v4 (E1): S1-side ambiguity: how many S1 entities share this name / address (over the full S1 table)
    "s1_name_cnt_q", "s1_name_cnt_c", "pool_name_cnt_q", "s1_addr_cnt_q", "s1_addr_cnt_c", "pool_addr_cnt_q",
    "name_unique_pair",
]''')
rep('''    for j, name in enumerate(FEATURES[-14:]):
        F[name] = (NF if j < 9 else CS)[:, j if j < 9 else j - 9]''',
    '''    for j, name in enumerate(FEATURES[-21:-7]):
        F[name] = (NF if j < 9 else CS)[:, j if j < 9 else j - 9]
    F["s1_name_cnt_q"], F["s1_name_cnt_c"] = np.log1p(rec.s1_name_cnt[q]), np.log1p(rec.s1_name_cnt[c])
    F["pool_name_cnt_q"] = np.log1p(rec.pool_name_cnt[q])
    F["s1_addr_cnt_q"], F["s1_addr_cnt_c"] = np.log1p(rec.s1_addr_cnt[q]), np.log1p(rec.s1_addr_cnt[c])
    F["pool_addr_cnt_q"] = np.log1p(rec.pool_addr_cnt[q])
    F["name_unique_pair"] = ((rec.s1_name_cnt[q] == 1) & (rec.s1_name_cnt[c] <= 1)).astype(np.float32)''')

# ---------------------------------------------------------------- label-free silver report
rep('''print(pd.DataFrame({"S1": by_c, "with_match": has, "share": (has / by_c).round(3)}))''',
    '''print(pd.DataFrame({"S1": by_c, "with_match": has, "share": (has / by_c).round(3)}))

# label-free monitor: share of "silver" pairs (same country + full name + house number, unique among S1) that were
# predicted. On labelled validation this share is ~0.99-1.00 for true silver pairs; a much lower value in any country
# (v3: France 0.952) signals a blocking or transfer failure there.
_sk = _key(test, "ex_name_hn").to_numpy(zero_copy_only=False)
_s1k = pd.Series(_sk[test_q])
_uniq = _s1k[_s1k != ""].value_counts()
_sil = pd.DataFrame({"k": _sk[test_q], "q": test_q}).merge(
    pd.DataFrame({"k": _sk[test.src != 1], "c": np.flatnonzero(test.src != 1)}), on="k")
_sil = _sil[(_sil.k != "") & (_sil.k.map(_uniq) == 1)]
_pred = set((mq.astype(np.int64) * test.n + mc).tolist())
_sil["acc"] = [k in _pred for k in (_sil.q.to_numpy(np.int64) * test.n + _sil.c.to_numpy()).tolist()]
_sil["country"] = te_ctry[test.country_codes[_sil.q.to_numpy()]]
log("SILVER REPORT (test, unique name+number pairs predicted as matches): "
    + str(_sil.groupby("country").acc.mean().round(4).to_dict()) + "  (v3: France 0.952)")''')

open("v4_src.py", "w", encoding="utf-8").write(s)
print("v4_src.py written", len(s.splitlines()), "lines")
