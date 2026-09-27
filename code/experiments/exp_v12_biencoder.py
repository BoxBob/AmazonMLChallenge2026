# v12 (NEXT_PLAN phase 3, "C"): learned blocking channel (SC-Block, Brinkmann et al. ESWC 2024; DeepBlocker).
# 2% of true pairs are never candidates (ceiling 0.9939 at top-80) and 63% of those rank >= 100 in the TF-IDF/SVD channels:
# more k can't reach them, a better representation can. A multilingual bi-encoder (Apache-2.0
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, plain transformers + mean pooling) is fine-tuned with a
# symmetric in-batch contrastive loss (MultipleNegativesRanking) on true (S1, S2/S3) pairs of NON-validation entities.
# Measured on validation: share of the current blocking misses it retrieves at k, and the new oracle ceiling.
# Resumable: model, then corpus embeddings (fp16 memmap on D:), then search results are cached.
import math
import os
import time

os.environ["HF_HOME"] = r"D:\AmazonMLChallenge2026\work\hf"
os.environ["TEMP"] = os.environ["TMP"] = r"D:\AmazonMLChallenge2026\work\tmp"
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as Fn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import valscore as V

BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
W12 = r"D:\AmazonMLChallenge2026\work\v12"
MODEL_DIR = W12 + r"\bienc_model"
os.makedirs(W12, exist_ok=True)
N_PAIRS, BATCH, LR, MAX_LEN, SCALE, KMAX = 1_200_000, 128, 5e-5, 48, 20.0, 100
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


tb = pq.read_table(V.W + r"\train_records.parquet", columns=["name", "addr", "country", "src"]).combine_chunks()
N = tb.num_rows
SRC = tb["src"].to_numpy()
CTRY = pd.factorize(tb["country"].to_pandas())[0].astype(np.int8)


def texts(idx):
    return [f"{n} | {a}" if a else n for n, a in zip(tb["name"].take(pa.array(idx)).to_pylist(),
                                                     tb["addr"].take(pa.array(idx)).to_pylist())]


tok = AutoTokenizer.from_pretrained(MODEL_DIR if os.path.exists(MODEL_DIR + r"\config.json") else BASE)


def encode(model, idx):
    enc = tok(texts(idx), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
    out = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
    return Fn.normalize((out * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1)


# ---------------- 1. training pairs: one random true match per non-validation S1 (no duplicate S1 inside a batch)
ids = pd.Index(pq.read_table(V.W + r"\train_records.parquet", columns=["entity_id"])["entity_id"].to_pandas())
gt = pd.read_csv(V.D + "/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
gm = gt.assign(c=gt.matched_entity_ids.str.split(",")).explode("c")
gm = gm[gm.c != ""]
GQ_all = ids.get_indexer(gm.source1_entity_id).astype(np.int64)
GC_all = ids.get_indexer(gm.c).astype(np.int64)
if not os.path.exists(MODEL_DIR + r"\config.json"):
    keep = ~np.isin(GQ_all, V.VAL)
    d = pd.DataFrame({"q": GQ_all[keep], "c": GC_all[keep]}).sample(frac=1.0, random_state=11)
    d = d.drop_duplicates("q").head(N_PAIRS)
    tq, tc = d.q.to_numpy(), d.c.to_numpy()
    log(f"training pairs {len(tq):,} (one per S1, validation entities excluded)")
    model = AutoModel.from_pretrained(BASE).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(tq) // BATCH
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    lab = torch.arange(BATCH, device="cuda")
    run = 0.0
    for s in range(steps):
        a = s * BATCH
        with torch.autocast("cuda", dtype=torch.float16):
            eq, ec = encode(model, tq[a:a + BATCH]), encode(model, tc[a:a + BATCH])
        sim = SCALE * eq.float() @ ec.float().T
        loss = (Fn.cross_entropy(sim, lab) + Fn.cross_entropy(sim.T, lab)) / 2
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if s % 1000 == 0:
            log(f"  step {s:,}/{steps:,} loss {run:.4f}")
    model.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    del model, opt
    torch.cuda.empty_cache()
    log("bi-encoder fine-tuned")

# ---------------- 2. embed the S2/S3 corpus and the validation S1 (fp16 memmaps, resumable in 1M-record blocks)
model = AutoModel.from_pretrained(MODEL_DIR).cuda().eval().half()
corpus = np.flatnonzero(SRC != 1)
emb_path = W12 + r"\train_pool_emb.f16"
E = np.lib.format.open_memmap(emb_path + ".npy", mode="r+" if os.path.exists(emb_path + ".npy") else "w+",
                              dtype=np.float16, shape=(len(corpus), 384))
done_path = W12 + r"\train_pool_emb.done"
done = int(open(done_path).read()) if os.path.exists(done_path) else 0
BLOCK, EB, QB = 500_000, 1024, 512
with torch.no_grad():
    for b0 in range(done, len(corpus), BLOCK):
        for a in range(b0, min(b0 + BLOCK, len(corpus)), EB):
            E[a:a + EB] = encode(model, corpus[a:a + EB]).cpu().numpy()
        E.flush()
        open(done_path, "w").write(str(min(b0 + BLOCK, len(corpus))))
        log(f"  embedded {min(b0 + BLOCK, len(corpus)):,}/{len(corpus):,} S2/S3 records")
    Q = np.concatenate([encode(model, V.VAL[a:a + EB]).cpu().numpy() for a in range(0, len(V.VAL), EB)])
log("embeddings ready")

# ---------------- 3. exact top-KMAX search per validation S1 (same country only), corpus streamed through the GPU
res_path = W12 + r"\val_bienc_top100.npz"
if os.path.exists(res_path):
    z = np.load(res_path)
    top_c, top_s = z["c"], z["s"]
else:
    qt = torch.from_numpy(Q).cuda()
    qctry = torch.from_numpy(CTRY[V.VAL].astype(np.int64)).cuda()
    best_s = torch.full((len(Q), KMAX), -2.0, device="cuda")
    best_c = torch.full((len(Q), KMAX), -1, dtype=torch.long, device="cuda")
    for b0 in range(0, len(corpus), BLOCK):
        eb = torch.from_numpy(np.asarray(E[b0:b0 + BLOCK])).cuda()
        cb = torch.from_numpy(corpus[b0:b0 + BLOCK]).cuda()
        cctry = torch.from_numpy(CTRY[corpus[b0:b0 + BLOCK]].astype(np.int64)).cuda()
        for a in range(0, len(Q), QB):                      # QB x BLOCK fp16 scores = 512 MB
            s = qt[a:a + QB] @ eb.T
            s[qctry[a:a + QB, None] != cctry[None, :]] = -2.0
            v, i = s.topk(KMAX, dim=1)
            allv = torch.cat([best_s[a:a + QB], v.float()], 1)
            allc = torch.cat([best_c[a:a + QB], cb[i]], 1)
            vv, ii = allv.topk(KMAX, dim=1)
            best_s[a:a + QB], best_c[a:a + QB] = vv, allc.gather(1, ii)
        del eb, cb, cctry
        log(f"  searched {min(b0 + BLOCK, len(corpus)):,}/{len(corpus):,}")
    top_c, top_s = best_c.cpu().numpy(), best_s.cpu().numpy()
    np.savez(res_path, c=top_c, s=top_s)

# ---------------- 4. recall of the current blocking misses, and the oracle ceiling with the union
vz = np.load(r"D:\AmazonMLChallenge2026\work\v10m80\val_pairs.npz")
cur = set((vz["q"].astype(np.int64) * N + vz["c"].astype(np.int64)).tolist())
gk = V.GQ * N + V.GC
in_cur = np.array([k in cur for k in gk.tolist()])
log(f"validation true pairs {len(gk):,}; in current top-80 candidates {in_cur.mean():.2%}; misses {np.sum(~in_cur):,}")
qpos = np.searchsorted(V.VAL, V.GQ)
for k in (10, 20, 30, 50, 100):
    bk = set((V.VAL[:, None] * N + top_c[:, :k]).ravel().tolist())
    hit = np.array([x in bk for x in gk.tolist()])
    new = len(bk - cur) / len(V.VAL)
    u = in_cur | hit
    ceil = V.f_ent(V.VAL, V.GQ[u], V.GC[u]).mean()
    log(f"k={k:3d}: bi-encoder recall alone {hit.mean():.2%} | recovers {np.mean(hit[~in_cur]):.1%} of current misses | "
        f"union recall {u.mean():.2%} | new candidates per S1 {new:.1f} | oracle ceiling {ceil:.4f} (now "
        f"{V.f_ent(V.VAL, V.GQ[in_cur], V.GC[in_cur]).mean():.4f})")
log("done")
