"""Build v3_src.py (→ code/src/v3.ipynb) from the v2 source with explicit, checked patches."""
import re

s = open("improved_src.py", encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:80], n)
    s = s.replace(old, new)


# ---------------------------------------------------------------- header
head_end = s.index("# %%\nimport os")
s = '''# %% [markdown]
# # Business Entity Resolution — v3 (towards SOTA, GPU)
#
# v3 = v2 + the fixes that won the v3 experiments (see `VERSION_FINDINGS.md`):
#
# | v2 loss bucket | v3 fix | Source |
# |---|---|---|
# | Blocking misses (34% of loss, 41% blank-address) | extra **name kNN over blank-address pool records** + blank-address squashed-name key | Sparkly (top-k per sub-population) |
# | False positives: same name, neighbouring house number | **numeric-aware features** (digit edit distance, dropped digit, numeric gap, unmatched numbers) | Ditto span typing, Linacre |
# | Distractor siblings | **candidate-set token rarity** (tokens unique to a candidate among the entity's candidates) | Linacre |
# | Wrong merges in multi-source clusters | **cross-source consistency** features in the group model | Saeedi et al., TransClean |
# | Uncertain pairs | **Ditto-style cross-encoder re-ranker** (Apache-2.0 MiniLM-L12, `[COL]/[VAL]/[NUM]` serialization) blended with the GBDT on the uncertain band | Ditto, Peeters & Bizer |
#
# Normalization, lexicon, embeddings and the 3-channel kNN are identical to v2 and are read from the shared
# cache `work/v2` (built by `improved.ipynb`, or rebuilt here if missing).
# Outputs: `output/candidate_pairs_v3.tsv`, `output/matching_results_v3.tsv`.

''' + s[head_end:]

# ---------------------------------------------------------------- paths / config
rep('WORK = ROOT / "work" / "v2"', 'WORK = ROOT / "work" / "v3"\nSHARED = ROOT / "work" / "v2"   # caches shared with v2 (same normalization/embeddings/kNN)')
rep("for p in (WORK, TMP, OUT):", "for p in (WORK, SHARED, TMP, OUT):")
rep('OUT_CAND = OUT / "candidate_pairs_v2.tsv"', 'OUT_CAND = OUT / "candidate_pairs_v3.tsv"')
rep('OUT_MATCH = OUT / "matching_results_v2.tsv"', 'OUT_MATCH = OUT / "matching_results_v3.tsv"')
rep('os.environ["TEMP"] = os.environ["TMP"] = os.environ["JOBLIB_TEMP_FOLDER"] = str(TMP)',
    'os.environ["TEMP"] = os.environ["TMP"] = os.environ["JOBLIB_TEMP_FOLDER"] = str(TMP)\n'
    'os.environ["HF_HOME"] = str(ROOT / "work" / "hf")   # Hugging Face model cache on D:')
rep('    cache = WORK / "lexicons.parquet"', '    cache = SHARED / "lexicons.parquet"')
rep('        cache = WORK / f"{split}_records.parquet"', '        cache = SHARED / f"{split}_records.parquet"')
rep('    path = WORK / f"{rec.split}_emb_{field}.npy"', '    path = SHARED / f"{rec.split}_emb_{field}.npy"')
rep('    if all((WORK / f"{r.split}_emb_{field}.npy").exists() for r in (train, test)):',
    '    if all((SHARED / f"{r.split}_emb_{field}.npy").exists() for r in (train, test)):')
rep('knn_cache = WORK / "train_knn.npz"', 'knn_cache = SHARED / "train_knn.npz"')
rep('knn_cache = WORK / "test_knn.npz"', 'knn_cache = SHARED / "test_knn.npz"')
rep('T0 = time.time()\n\n\ndef log',
    '''K_BLANK = 5                 # v3: name kNN neighbours per source among blank-address pool records
BLANK_KEY_CAP = 1000        # v3: block cap of the blank-address squashed-name key
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"   # Apache-2.0, 33M params
CE_TRAIN_S1 = 150_000       # training entities for the cross-encoder (top-8 candidates each)
CE_MAX_LEN = 96
CE_BAND = (0.01, 0.99)      # the cross-encoder re-scores pairs whose GBDT probability is in this band
T0 = time.time()


def log''')

# ---------------------------------------------------------------- blank-address kNN channel
rep('''EXACT_KEYS = ("ex_squash", "ex_pc_first", "ex_first_hn", "ex_first2", "ex_cnum", "ex_hn_next")''',
    '''@torch.no_grad()
def gpu_knn_blank(rec, qidx):
    """v3: name kNN restricted to blank-address S2/S3 records -> int32 (len(qidx), 2*K_BLANK)"""
    EN = EMB[(rec.split, "core")]
    out = np.full((len(qidx), 2 * K_BLANK), -1, dtype=np.int32)
    for ci in np.unique(rec.country_codes[qidx]):
        qpos = np.flatnonzero(rec.country_codes[qidx] == ci)
        Q = torch.from_numpy(np.ascontiguousarray(EN[qidx[qpos]])).to(DEVICE)
        for si, s in enumerate((2, 3)):
            pool = np.flatnonzero((rec.src == s) & (rec.country_codes == ci) & rec.addr_blank & ~rec.core_blank)
            if len(pool) == 0:
                continue
            P = torch.from_numpy(np.ascontiguousarray(EN[pool])).to(DEVICE)
            k = min(K_BLANK, len(pool))
            for q0 in range(0, len(qpos), 2048):
                idx = (Q[q0:q0 + 2048] @ P.T).topk(k, dim=1).indices.cpu().numpy()
                out[qpos[q0:q0 + 2048], si * K_BLANK:si * K_BLANK + k] = pool[idx]
            del P
        del Q
        torch.cuda.empty_cache()
    out[rec.core_blank[qidx]] = -1
    log(f"  {rec.split}: blank-address name kNN done for {len(qidx):,} queries")
    return out


EXACT_KEYS = ("ex_squash", "ex_pc_first", "ex_first_hn", "ex_first2", "ex_cnum", "ex_hn_next", "ex_blank_squash")''')
rep('''    if name == "ex_squash":''', '''    if name in ("ex_squash", "ex_blank_squash"):''')
rep('''        pool = np.flatnonzero((rec.src != 1) & (idx != empty))
        sizes = np.bincount(idx[pool], minlength=len(codes.dictionary))
        pool = pool[sizes[idx[pool]] <= EXACT_BLOCK_CAP]''',
    '''        pool_mask = (rec.src != 1) & (idx != empty)
        if name == "ex_blank_squash":             # v3: blank-address records get their own key block
            pool_mask &= rec.addr_blank
        pool = np.flatnonzero(pool_mask)
        sizes = np.bincount(idx[pool], minlength=len(codes.dictionary))
        pool = pool[sizes[idx[pool]] <= (BLANK_KEY_CAP if name == "ex_blank_squash" else EXACT_BLOCK_CAP)]''')

# ---------------------------------------------------------------- stage-1 with the extra channel
rep('''PF_FEATURES = ["rank_core", "rank_addr", "rank_comb",''',
    '''STAGE_CHANNELS = CHANNELS + ("blank",)
K["blank"] = K_BLANK
PF_FEATURES = ["rank_core", "rank_addr", "rank_comb", "rank_blank",''')
rep('''    for chi, ch in enumerate(CHANNELS):
        sub = knn[ch][pos]''', '''    for chi, ch in enumerate(STAGE_CHANNELS):
        sub = knn[ch][pos]''')
rep('''    a, b = np.searchsorted(ex[0], qchunk[0]), np.searchsorted(ex[0], qchunk[-1], side="right")
    mine = a + np.flatnonzero(np.isin(ex[0][a:b], qchunk))   # the id range also holds other splits' queries
    qs.append(ex[0][mine]); cs.append(ex[1][mine])
    rk.append(np.zeros(len(mine), dtype=np.int8)); chs.append((ex[2][mine] + 3).astype(np.int8))''',
    '''    lo, hi = np.searchsorted(ex[0], qchunk), np.searchsorted(ex[0], qchunk, side="right")   # rows of exactly these queries
    n_ex = hi - lo
    mine = np.repeat(lo, n_ex) + (np.arange(n_ex.sum()) - np.repeat(np.cumsum(n_ex) - n_ex, n_ex))
    qs.append(ex[0][mine]); cs.append(ex[1][mine])
    rk.append(np.zeros(len(mine), dtype=np.int8)); chs.append((ex[2][mine] + len(STAGE_CHANNELS)).astype(np.int8))''')
rep('''    for chi, ch in enumerate(CHANNELS):
        r = np.full(len(uk), K[ch], dtype=np.int8)''', '''    for chi, ch in enumerate(STAGE_CHANNELS):
        r = np.full(len(uk), K[ch], dtype=np.int8)''')
rep('''        f[inv[chs == ti + 3]] = 1.0''', '''        f[inv[chs == ti + len(STAGE_CHANNELS)]] = 1.0''')
rep('''    KNN_TV = gpu_knn_multi(train, Q_TV)
    np.savez(knn_cache, **KNN_TV)''', '''    KNN_TV = gpu_knn_multi(train, Q_TV)
    np.savez(knn_cache, **KNN_TV)
_bc = WORK / "train_knn_blank.npy"
KNN_TV["blank"] = np.load(_bc) if _bc.exists() else gpu_knn_blank(train, Q_TV)
np.save(_bc, KNN_TV["blank"])''')
rep('''    KNN_TE = gpu_knn_multi(test, test_q)
    np.savez(knn_cache, **KNN_TE)''', '''    KNN_TE = gpu_knn_multi(test, test_q)
    np.savez(knn_cache, **KNN_TE)
_bc = WORK / "test_knn_blank.npy"
KNN_TE["blank"] = np.load(_bc) if _bc.exists() else gpu_knn_blank(test, test_q)
np.save(_bc, KNN_TE["blank"])''')
rep('''log(f"VAL stage-1 blocking recall {_found / n_true(VAL_Q):.4f} (baseline 0.8364)''',
    '''log(f"VAL stage-1 blocking recall {_found / n_true(VAL_Q):.4f} (v2 0.9750, baseline 0.8364)''')

# ---------------------------------------------------------------- new pair features
rep('''    "rel_n_tset", "rel_a_tset", "rel_pf",
]''', '''    "rel_n_tset", "rel_a_tset", "rel_pf",
    # v3: numeric-aware address features (Ditto span typing / Linacre: numbers are not typos)
    "hn_lev", "hn_contains", "hn_gap", "num_shared", "num_only_c", "num_only_q", "num_min_lev",
    "street_tset", "name_num_diff",
    # v3: token rarity within the entity's candidate set (Linacre)
    "cs_addr_rare_overlap", "cs_addr_unique_c", "cs_name_rare_overlap", "cs_name_unique_c", "cs_disc_match",
]
_LEAD = re.compile(r"\\d+")


def _num_feats(hq, hc, aq, ac, nq, nc):
    """numeric-aware comparison: a digit typo (1604 vs 604) vs a different building (1315 vs 1336)"""
    from rapidfuzz.distance import Levenshtein
    n = len(hq)
    out = np.full((n, 9), -1.0, np.float32)
    for i in range(n):
        a, b = hq[i], hc[i]
        if a and b:
            out[i, 0] = Levenshtein.distance(a, b)
            out[i, 1] = 1.0 if (a in b or b in a) else 0.0
            ma, mb = _LEAD.match(a), _LEAD.match(b)
            if ma and mb:
                ia, ib = int(ma.group(0)[:9]), int(mb.group(0)[:9])
                out[i, 2] = abs(ia - ib) / max(ia, ib, 1)
        X, Y = aq[i].split(), ac[i].split()
        SX = {t for t in X if t[0].isdigit()}
        SY = {t for t in Y if t[0].isdigit()}
        if SX and SY:
            out[i, 3] = len(SX & SY)
            out[i, 4] = len(SY - SX)
            out[i, 5] = len(SX - SY)
            out[i, 6] = max((min(Levenshtein.distance(t, u) for u in SX) for t in SY - SX), default=0)
        tx = " ".join(t for t in X if not t[0].isdigit())
        ty = " ".join(t for t in Y if not t[0].isdigit())
        if tx and ty:
            out[i, 7] = fuzz.token_set_ratio(tx, ty)
        dq = {t for t in nq[i].split() if any(ch.isdigit() for ch in t)}
        dc = {t for t in nc[i].split() if any(ch.isdigit() for ch in t)}
        if dq or dc:
            out[i, 8] = len(dq ^ dc)
    return out


def _candset_feats(bounds, aq, ac, nq, nc):
    """token rarity inside each entity's candidate set: what makes this candidate different from its siblings"""
    out = np.zeros((len(ac), 5), np.float32)
    for a, b in bounds:
        A = [set(x.split()) for x in ac[a:b]]
        Nn = [set(x.split()) for x in nc[a:b]]
        dfa = Counter(t for s in A for t in s)
        dfn = Counter(t for s in Nn for t in s)
        Q, QN, m = set(aq[a].split()), set(nq[a].split()), b - a
        disc = {t for t in Q if 0 < dfa.get(t, 0) < m}
        for j in range(m):
            sa, sn = A[j], Nn[j]
            out[a + j, 0] = sum(1.0 / dfa[t] for t in Q & sa) / max(len(Q), 1)
            out[a + j, 1] = sum(1 for t in sa - Q if dfa[t] == 1)
            out[a + j, 2] = sum(1.0 / dfn[t] for t in QN & sn) / max(len(QN), 1)
            out[a + j, 3] = sum(1 for t in sn - QN if dfn[t] == 1)
            out[a + j, 4] = len(disc & sa) / len(disc) if disc else -1.0
    return out''')
rep('''    F["rel_pf"] = F["pf"] - _group_max(q, F["pf"])
    return np.column_stack([F[f] for f in FEATURES]).astype(np.float32)''',
    '''    F["rel_pf"] = F["pf"] - _group_max(q, F["pf"])
    hq, hc = rec.take("hn", q), rec.take("hn", c)
    NF = np.vstack(Parallel(n_jobs=N_JOBS)(
        delayed(_num_feats)(hq[i:i + step], hc[i:i + step], aq[i:i + step], ac[i:i + step], cq[i:i + step],
                            cc[i:i + step]) for i in range(0, len(q), step)))
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    jobs = []
    for part in np.array_split(np.arange(len(starts)), N_JOBS):
        if len(part):
            lo, hi = starts[part[0]], ends[part[-1]]
            jobs.append(delayed(_candset_feats)([(x - lo, y - lo) for x, y in zip(starts[part], ends[part])],
                                                aq[lo:hi], ac[lo:hi], cq[lo:hi], cc[lo:hi]))
    CS = np.vstack(Parallel(n_jobs=N_JOBS)(jobs))
    for j, name in enumerate(FEATURES[-14:]):
        F[name] = (NF if j < 9 else CS)[:, j if j < 9 else j - 9]
    return np.column_stack([F[f] for f in FEATURES]).astype(np.float32)''')

# ---------------------------------------------------------------- cross-source group features
rep('''ALL_FEATURES = FEATURES + G_FEATURES
Gtr = group_features(tr_q, train.src[tr_c], p_oof)
Gva = group_features(va_q, train.src[va_c], p1_va)''',
    '''XS_FEATURES = ["xs_name", "xs_addr", "xs_hn_eq", "ss_name", "ss_addr", "ss_hn_eq"]


def cross_source_feats(rec, q, c, p):
    """v3 (Saeedi et al. / TransClean): agreement of each candidate with the entity's best candidate from the
    other source, and with its best rival from the same source"""
    s = rec.src[c]
    df = pd.DataFrame({"q": q, "c": c, "s": s, "p": p}).sort_values(["q", "s", "p"], ascending=[True, True, False])
    df["rk"] = df.groupby(["q", "s"]).cumcount()
    best = df[df.rk == 0].set_index(["q", "s"])["c"]
    second = df[df.rk == 1].set_index(["q", "s"])["c"]
    b_other = best.reindex(pd.MultiIndex.from_arrays([q, np.where(s == 2, 3, 2)])).to_numpy()
    b_best_same = best.reindex(pd.MultiIndex.from_arrays([q, s])).to_numpy()
    b_second_same = second.reindex(pd.MultiIndex.from_arrays([q, s])).to_numpy()
    b_same = np.where(b_best_same == c, b_second_same, b_best_same)
    out = np.full((len(q), 6), -1.0, np.float32)
    for j, bref in enumerate((b_other, b_same)):
        idx = np.flatnonzero(~np.isnan(bref))
        bi = bref[idx].astype(np.int64)
        for k, col in enumerate(("core", "addr")):
            out[idx, 3 * j + k] = process.cpdist(rec.take(col, c[idx]), rec.take(col, bi),
                                                 scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        h1 = np.array(rec.take("hn", c[idx]), dtype=object)
        h2 = np.array(rec.take("hn", bi), dtype=object)
        out[idx, 3 * j + 2] = np.where((h1 == "") | (h2 == ""), -1.0, (h1 == h2).astype(np.float32))
    return out


def group_ctx(rec, q, c, p):
    return np.hstack([group_features(q, rec.src[c], p), cross_source_feats(rec, q, c, p)])


ALL_FEATURES = FEATURES + G_FEATURES + XS_FEATURES
Gtr = group_ctx(train, tr_q, tr_c, p_oof)
Gva = group_ctx(train, va_q, va_c, p1_va)''')
rep('''log(f"group-context model best iteration {m2.best_iteration}")
del Gtr, tr_X
gc.collect()''', '''log(f"group-context model best iteration {m2.best_iteration}")
del Gtr
gc.collect()

# %% [markdown]
# ## 11b. Ditto-style cross-encoder re-ranker (v3)
# Records are serialized as `[COL] name [VAL] … [COL] addr [VAL] …` with the house number wrapped in
# `[NUM] … [/NUM]` (Ditto's span typing), and an Apache-2.0 MiniLM cross-encoder is fine-tuned on the top-8
# candidates of 150k training entities. It re-scores only the uncertain band of GBDT probabilities, and a
# logistic blend of the two logits (fit on validation half A, reported on half B) gives the final probability.

# %%
import math  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

CE_DIR = WORK / "ce_model"


def serialize(rec, idx):
    names, addrs, hns = rec.take("name", idx), rec.take("addr", idx), rec.take("hn", idx)
    out = []
    for n, a, h in zip(names, addrs, hns):
        if h:
            a = (" " + a + " ").replace(f" {h} ", f" [NUM] {h} [/NUM] ", 1).strip()
        out.append(f"[COL] name [VAL] {n} [COL] addr [VAL] {a if a else 'none'}")
    return out


def ce_batches(rec, qi, ci, bs):
    for a in range(0, len(qi), bs):
        enc = ce_tok(serialize(rec, qi[a:a + bs]), serialize(rec, ci[a:a + bs]), truncation=True,
                     max_length=CE_MAX_LEN, padding=True, return_tensors="pt")
        yield a, {k: v.to(DEVICE) for k, v in enc.items()}


if (CE_DIR / "config.json").exists():
    ce_tok = AutoTokenizer.from_pretrained(CE_DIR)
    ce_model = AutoModelForSequenceClassification.from_pretrained(CE_DIR).to(DEVICE)
    log(f"cross-encoder loaded from {CE_DIR}")
else:
    ce_tok = AutoTokenizer.from_pretrained(CE_MODEL)
    ce_tok.add_special_tokens({"additional_special_tokens": ["[COL]", "[VAL]", "[NUM]", "[/NUM]"]})
    ce_model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL, num_labels=1)
    ce_model.resize_token_embeddings(len(ce_tok))
    ce_model.to(DEVICE).train()
    ents = np.random.default_rng(3).choice(np.unique(tr_q), CE_TRAIN_S1, replace=False)
    sel = np.flatnonzero(np.isin(tr_q, ents) & (tr_X[:, FEATURES.index("pf_rank")] < 8))
    sel = np.random.default_rng(4).permutation(sel)
    yt = torch.from_numpy(tr_y[sel].astype(np.float32))
    opt = torch.optim.AdamW(ce_model.parameters(), lr=3e-5, weight_decay=0.01)
    steps = math.ceil(len(sel) / 64)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    for step, (a, enc) in enumerate(ce_batches(train, tr_q[sel].astype(np.int64), tr_c[sel].astype(np.int64), 64)):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = ce_model(**enc).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), yt[a:a + 64].to(DEVICE))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(ce_model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 2000 == 0:
            log(f"  cross-encoder step {step:,}/{steps:,} loss {loss.item():.4f}")
    ce_model.save_pretrained(CE_DIR)
    ce_tok.save_pretrained(CE_DIR)
ce_model.eval()
del tr_X
gc.collect()


@torch.no_grad()
def ce_score(rec, q, c):
    out = np.zeros(len(q), np.float32)
    for a, enc in ce_batches(rec, q.astype(np.int64), c.astype(np.int64), 256):
        with torch.autocast("cuda", dtype=torch.float16):
            out[a:a + 256] = torch.sigmoid(ce_model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


def _logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


va_band = np.flatnonzero((p2_va >= CE_BAND[0]) & (p2_va <= CE_BAND[1]))
pce_va = ce_score(train, va_q[va_band], va_c[va_band])
log(f"cross-encoder scored {len(va_band):,} validation band pairs ({len(va_band) / len(p2_va):.1%})")
_half = np.random.default_rng(7).random(len(VAL_Q)) < 0.5            # validation halves A / B
VAL_A, VAL_B = VAL_Q[_half], VAL_Q[~_half]
_inA = np.isin(va_q[va_band], VAL_A)
_Xb = np.column_stack([_logit(p2_va[va_band]), _logit(pce_va)])
blend_A = LogisticRegression(C=10.0).fit(_Xb[_inA], va_y[va_band][_inA])
pb_A = p2_va.copy()
pb_A[va_band] = blend_A.predict_proba(_Xb)[:, 1]
blend = LogisticRegression(C=10.0).fit(_Xb, va_y[va_band])            # refit on all validation for test time
log(f"blend weights (GBDT, cross-encoder) {blend.coef_.round(3).tolist()} bias {blend.intercept_.round(3).tolist()}")
pb_va = p2_va.copy()
pb_va[va_band] = blend.predict_proba(_Xb)[:, 1]''')

# ---------------------------------------------------------------- decision uses the blend too
rep('''for nm, prob in (("stage-1 matcher", p1_va), ("group-context model", p2_va)):''',
    '''for nm, prob in (("stage-1 matcher", p1_va), ("group-context model", p2_va), ("blend", pb_va)):''')
rep('''best_prob = p2_va if BEST_MODEL == "group-context model" else p1_va''',
    '''best_prob = {"stage-1 matcher": p1_va, "group-context model": p2_va, "blend": pb_va}[BEST_MODEL]''')
rep('''(baseline 0.8871)")''', '''(v2 0.9721, baseline 0.8871)")''')


rep('''BEST_MODEL = max(results, key=lambda k: max(results[k][1], results[k][3]))''',
    '''BEST_MODEL = max(results, key=lambda k: max(results[k][1], results[k][3]))
# honest estimate: blend + threshold tuned on half A, scored on half B
_mA = np.isin(va_q, VAL_A)
_tA = {t: macro_f05(VAL_A, *resolve(va_q[_mA], va_c[_mA], train.src[va_c[_mA]], pb_A[_mA], np.full(_mA.sum(), t)))
       for t in np.round(np.arange(0.5, 0.91, 0.025), 3)}
_tb = max(_tA, key=_tA.get)
_mB = ~_mA
log(f"HOLD-OUT CHECK: blend tuned on half A (t={_tb}) -> half B macro F0.5 = "
    f"{macro_f05(VAL_B, *resolve(va_q[_mB], va_c[_mB], train.src[va_c[_mB]], pb_A[_mB], np.full(_mB.sum(), _tb))):.4f}")''')

# ---------------------------------------------------------------- test inference
rep('''## 13. Test inference → `output/candidate_pairs_v2.tsv`, `output/matching_results_v2.tsv`''',
    '''## 13. Test inference → `output/candidate_pairs_v3.tsv`, `output/matching_results_v3.tsv`''')
rep('''        if BEST_MODEL == "group-context model":
            p = predict(m2, np.hstack([X, group_features(q, test.src[c], p)]), ALL_FEATURES)''',
    '''        if BEST_MODEL != "stage-1 matcher":
            p = predict(m2, np.hstack([X, group_ctx(test, q, c, p)]), ALL_FEATURES)
        if BEST_MODEL == "blend":
            band = np.flatnonzero((p >= CE_BAND[0]) & (p <= CE_BAND[1]))
            pce = ce_score(test, q[band], c[band])
            p = p.copy()
            p[band] = blend.predict_proba(np.column_stack([_logit(p[band]), _logit(pce)]))[:, 1]''')

open("v3_src.py", "w", encoding="utf-8").write(s)
print("v3_src.py written", len(s.splitlines()), "lines")
