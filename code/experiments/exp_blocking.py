# Experiment C: recover v2 blocking misses with (a) name kNN over blank-address pool records, (b) squashed-name key
# restricted to blank-address pool records with a larger cap, (c) a larger prefilter cut M.
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pac, torch, time
W = r"D:\AmazonMLChallenge2026\work\v2"; D = r"D:\AmazonMLChallenge2026\dataset\train"; T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
tbl = pq.read_table(W + r"\train_records.parquet", columns=["entity_id", "country", "src", "squash", "addr", "core"]).combine_chunks()
N = tbl.num_rows; src = tbl["src"].to_numpy(); ctry = pd.factorize(tbl["country"].to_pandas())[0]
ablank = pac.equal(pac.utf8_length(tbl["addr"]), 0).to_numpy(zero_copy_only=False)
EN = np.load(W + r"\train_emb_core.npy", mmap_mode="r")
z = np.load(W + r"\val_pairs.npz"); vq, vc = z["q"].astype(np.int64), z["c"].astype(np.int64)
VAL = np.unique(vq)
ids = pd.Index(tbl["entity_id"].to_pandas())
gt = pd.read_csv(D + "/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
m = gt.matched_entity_ids.str.split(",").explode(); m = m[m != ""]
GQ = ids.get_indexer(gt.source1_entity_id.loc[m.index]).astype(np.int64); GC = ids.get_indexer(m.values).astype(np.int64)
s_ = np.isin(GQ, VAL); GQ, GC = GQ[s_], GC[s_]
gk = GQ * N + GC; have = np.isin(gk, vq * N + vc); miss = gk[~have]
log(f"val true pairs {len(gk):,}; missing from v2 final candidates {len(miss):,} ({len(miss)/len(gk):.2%}); blank-addr among misses {ablank[miss % N].mean():.1%}")

@torch.no_grad()
def knn(qidx, pool, k):
    Q = torch.from_numpy(np.ascontiguousarray(EN[qidx])).cuda(); P = torch.from_numpy(np.ascontiguousarray(EN[pool])).cuda()
    out = []
    for a in range(0, len(qidx), 2048):
        out.append(pool[(Q[a:a+2048] @ P.T).topk(min(k, len(pool)), 1).indices.cpu().numpy()])
    return np.vstack(out)
rows = {k: [] for k in (5, 10, 20)}
for ci in np.unique(ctry[VAL]):
    qc = VAL[ctry[VAL] == ci]
    for s in (2, 3):
        pool = np.flatnonzero((src == s) & (ctry == ci) & ablank)
        idx = knn(qc, pool, 20)
        for k in rows: rows[k].append(qc[:, None] * N + idx[:, :k])
    log(f"country {ci}: blank-address pools searched")
for k in rows:
    keys = np.unique(np.concatenate([r.ravel() for r in rows[k]]))
    rec_ = np.isin(miss, keys).sum()
    new = np.setdiff1d(keys, vq * N + vc)
    log(f"(a) blank-address name kNN k={k:2d}/source: recovers {rec_:,}/{len(miss):,} misses ({rec_/len(gk):.2%} of true pairs); adds {len(new)/len(VAL):.1f} new cands/S1")
# (b) squashed-name key on blank-address pool records, larger caps
sq = tbl["squash"].to_numpy(zero_copy_only=False)
key = pd.Series(np.char.add(ctry.astype(str), "|").astype(object) + sq)
for cap in (50, 200, 1000):
    pool = np.flatnonzero((src != 1) & ablank & (pd.Series(sq).str.len().to_numpy() >= 4))
    pdf = pd.DataFrame({"k": key.to_numpy()[pool], "c": pool}); pdf = pdf[pdf.groupby("k").c.transform("size") <= cap]
    j = pd.DataFrame({"k": key.to_numpy()[VAL], "q": VAL}).merge(pdf, on="k")
    keys = j.q.to_numpy() * N + j.c.to_numpy()
    rec_ = np.isin(miss, keys).sum(); new = np.setdiff1d(keys, vq * N + vc)
    log(f"(b) blank-address squashed-name key cap={cap:4d}: recovers {rec_:,} misses ({rec_/len(gk):.2%}); adds {len(new)/len(VAL):.2f} new cands/S1")
