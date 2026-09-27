# v12 precompute: bi-encoder "b" (trained without validation AND GBDT-training entities, so cos_bi is out-of-sample for
# every pair the GBDTs see) -> embeddings of ALL train/test records + the "bi" kNN channel in the notebook's layout:
# int32 (n_queries, 2*K_BI), S2 neighbours in [:K_BI], S3 in [K_BI:], same country, -1 = none; query order =
# Q_TV (train: TRAIN_Q u VAL_Q, sorted) / test S1 (sorted record index). Resumable per 1M-record block.
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
from transformers import AutoModel, AutoTokenizer

W2 = r"D:\AmazonMLChallenge2026\work\v2"
W12 = r"D:\AmazonMLChallenge2026\work\v12"
MODEL_DIR = W12 + r"\bienc_modelb"
K_BI, MAX_LEN, EB, BLOCK, QB, PB = 10, 48, 1024, 1_000_000, 1024, 400_000
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR).cuda().eval().half()


@torch.no_grad()
def encode(tb, idx):
    t = [f"{n} | {a}" if a else n for n, a in zip(tb["name"].take(pa.array(idx)).to_pylist(),
                                                  tb["addr"].take(pa.array(idx)).to_pylist())]
    enc = tok(t, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
    out = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
    return Fn.normalize((out * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1)


for split in ("train", "test"):
    tb = pq.read_table(W2 + rf"\{split}_records.parquet", columns=["name", "addr", "country", "src"]).combine_chunks()
    n = tb.num_rows
    src = tb["src"].to_numpy()
    ctry = pd.factorize(tb["country"].to_pandas())[0]
    # ---- embeddings (all records, S1 included)
    path = W12 + rf"\{split}_emb_bi.npy"
    E = np.lib.format.open_memmap(path, mode="r+" if os.path.exists(path) else "w+", dtype=np.float16, shape=(n, 384))
    dpath = W12 + rf"\{split}_emb_bi.done"
    done = int(open(dpath).read()) if os.path.exists(dpath) else 0
    for b0 in range(done, n, BLOCK):
        for a in range(b0, min(b0 + BLOCK, n), EB):
            E[a:min(a + EB, b0 + BLOCK, n)] = encode(tb, np.arange(a, min(a + EB, b0 + BLOCK, n))).cpu().numpy()
        E.flush()
        open(dpath, "w").write(str(min(b0 + BLOCK, n)))
        log(f"{split}: embedded {min(b0 + BLOCK, n):,}/{n:,}")
    # ---- kNN channel
    kpath = W12 + rf"\{split}_knn_bi.npy"
    if os.path.exists(kpath):
        continue
    if split == "train":
        n1 = int((src == 1).sum())
        perm = np.random.default_rng(42).permutation(n1)
        qall = np.union1d(np.concatenate([perm[:300_000], perm[400_000:500_000]]), perm[300_000:400_000])
    else:
        qall = np.flatnonzero(src == 1)
    out = np.full((len(qall), 2 * K_BI), -1, dtype=np.int32)
    with torch.no_grad():
        for ci in np.unique(ctry[qall]):
            qpos = np.flatnonzero(ctry[qall] == ci)
            Q = torch.from_numpy(np.asarray(E[qall[qpos]])).cuda()
            for si, s in enumerate((2, 3)):
                pool = np.flatnonzero((src == s) & (ctry == ci))
                bs = torch.full((len(qpos), K_BI), -9.0, device="cuda")
                bi = torch.full((len(qpos), K_BI), -1, dtype=torch.long, device="cuda")
                for p0 in range(0, len(pool), PB):
                    P = torch.from_numpy(np.asarray(E[pool[p0:p0 + PB]])).cuda()
                    pg = torch.from_numpy(pool[p0:p0 + PB]).cuda()
                    for q0 in range(0, len(qpos), QB):
                        v, i = (Q[q0:q0 + QB] @ P.T).topk(min(K_BI, len(pg)), dim=1)
                        av = torch.cat([bs[q0:q0 + QB], v.float()], 1)
                        ai = torch.cat([bi[q0:q0 + QB], pg[i]], 1)
                        vv, ii = av.topk(K_BI, dim=1)
                        bs[q0:q0 + QB], bi[q0:q0 + QB] = vv, ai.gather(1, ii)
                    del P, pg
                out[qpos, si * K_BI:(si + 1) * K_BI] = bi.cpu().numpy().astype(np.int32)
                del bs, bi
                torch.cuda.empty_cache()
            del Q
            log(f"{split}: bi kNN done for country code {ci} ({len(qpos):,} queries)")
    np.save(kpath, out)
    log(f"{split}: wrote {kpath}")
log("done")
