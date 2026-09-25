import numpy as np, pandas as pd, pyarrow.parquet as pq, torch, time
W = r"D:\AmazonMLChallenge2026\work\baseline"; D = r"D:\AmazonMLChallenge2026\dataset\train"
rec = pq.read_table(W + r"\train_records.parquet", columns=["entity_id","country","src","core","addr"]).to_pandas()
N = len(rec); src = rec.src.to_numpy(); ctry = pd.factorize(rec.country)[0]
cb = (rec.core == "").to_numpy(); ab = (rec.addr == "").to_numpy()
EN = np.load(W + r"\train_emb_core.npy", mmap_mode="r"); EA = np.load(W + r"\train_emb_addr.npy", mmap_mode="r")
rng = np.random.default_rng(42); perm = rng.permutation(np.flatnonzero(src == 1)); VAL = np.sort(perm[300_000:340_000])  # 40k subset
ids = pd.Index(rec.entity_id)
gt = pd.read_csv(D + "/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
m = gt.matched_entity_ids.str.split(",").explode(); m = m[m != ""]
gq = ids.get_indexer(gt.source1_entity_id.loc[m.index]); gc_ = ids.get_indexer(m.values)
sel = np.isin(gq, VAL); GK = set((gq[sel].astype(np.int64) * N + gc_[sel]).tolist()); print("true pairs", len(GK))
@torch.no_grad()
def topk(qe, pool_emb_fn, pidx, k):
    Q = torch.from_numpy(np.ascontiguousarray(qe)).cuda()
    bs = torch.full((len(Q), k), -9.0, device="cuda", dtype=torch.float16); bi = torch.full((len(Q), k), -1, device="cuda")
    for c0 in range(0, len(pidx), 300_000):
        p = pidx[c0:c0+300_000]; P = torch.from_numpy(pool_emb_fn(p)).cuda(); pg = torch.from_numpy(p).cuda()
        for q0 in range(0, len(Q), 1024):
            s, i = (Q[q0:q0+1024] @ P.T).topk(min(k, len(p)), 1)
            s = torch.cat([bs[q0:q0+1024], s], 1); i = torch.cat([bi[q0:q0+1024], pg[i]], 1)
            ts, pos = s.topk(k, 1); bs[q0:q0+1024] = ts; bi[q0:q0+1024] = torch.gather(i, 1, pos)
    return bi.cpu().numpy()
K = 50
res = {}
for ch in ("name", "addr", "comb"):
    rows = []
    t = time.time()
    for ci in np.unique(ctry[VAL]):
        q = VAL[ctry[VAL] == ci]
        for s in (2, 3):
            pool = np.flatnonzero((src == s) & (ctry == ci))
            if ch == "name": qq = q[~cb[q]]; pool = pool[~cb[pool]]; qe = EN[qq]; f = lambda p: np.ascontiguousarray(EN[p])
            elif ch == "addr": qq = q[~ab[q]]; pool = pool[~ab[pool]]; qe = EA[qq]; f = lambda p: np.ascontiguousarray(EA[p])
            else: qq = q; qe = np.hstack([EN[qq], EA[qq]]); f = lambda p: np.hstack([EN[p], EA[p]])
            idx = topk(qe, f, pool, K)
            rows.append((np.repeat(qq, K), idx.ravel(), np.tile(np.arange(K), len(qq))))
    res[ch] = [np.concatenate(x) for x in zip(*rows)]
    print(ch, f"{time.time()-t:.0f}s")
def recall(cfg):
    keys = set()
    for ch, k in cfg.items():
        q, c, r = res[ch]; s = r < k; keys.update((q[s].astype(np.int64) * N + c[s]).tolist())
    return len(keys & GK) / len(GK), len(keys) / len(VAL)
for cfg in ({"name":10,"addr":10}, {"name":20,"addr":20}, {"name":10,"addr":30}, {"name":20,"addr":40}, {"name":50,"addr":50},
            {"comb":10}, {"comb":20}, {"comb":50}, {"name":10,"addr":10,"comb":20}, {"name":15,"addr":25,"comb":25},
            {"name":20,"addr":30,"comb":40}, {"name":30,"addr":50,"comb":50}):
    r, n = recall(cfg); print(f"{str(cfg):45} recall {r:.4f}  cands/q {n:.1f}")

keys = set()
for ch, k in {"name":30,"addr":50,"comb":50}.items():
    q, c, r = res[ch]; s = r < k; keys.update((q[s].astype(np.int64) * N + c[s]).tolist())
miss = np.array(sorted(GK - keys), dtype=np.int64); mq, mc = miss // N, miss % N
raw = {s: pd.read_csv(f"{D}/train_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3).set_index("entity_id") for s in (1,2,3)}
eid = rec.entity_id.to_numpy()
def raw_of(i):
    e = eid[i]; t = raw[int(e[1])].loc[e]; return f"{t.business_name[:38]:38} | {t.business_address[:60]}"
print("\nmissed", len(miss), "native-script cand:", np.mean([not raw[int(eid[i][1])].loc[eid[i]].business_name.isascii() for i in mc[:3000]]))
print("India share:", (rec.country.to_numpy()[mq] == "India").mean())
for j in np.random.default_rng(0).choice(len(miss), 22, replace=False):
    print(raw_of(mq[j])); print("   ", raw_of(mc[j]))
