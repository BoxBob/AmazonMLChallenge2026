# Where do the true pairs that blocking misses rank in each kNN channel? (Sparkly: top-k size is the main recall lever.)
# For every validation true pair absent from v4's final candidates: rank of c among same-country, same-source pool
# records by name cosine, address cosine and combined (mean) cosine, computed exactly on the GPU.
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import valscore as V

SH = r"D:\AmazonMLChallenge2026\work\v2"
W4 = r"D:\AmazonMLChallenge2026\work\v4"
tb = pq.read_table(SH + r"\train_records.parquet", columns=["country", "src", "addr", "core"]).to_pandas()
ctry = pd.factorize(tb.country)[0]
src = tb.src.to_numpy()
EN = np.load(SH + r"\train_emb_core.npy", mmap_mode="r")
EA = np.load(SH + r"\train_emb_addr.npy", mmap_mode="r")
vz = np.load(W4 + r"\val_pairs.npz")
have = set((vz["q"].astype(np.int64) * V.N + vz["c"].astype(np.int64)).tolist())
miss = np.array([k for k in V.GK.tolist() if k not in have], dtype=np.int64)
mq, mc = miss // V.N, miss % V.N
print(f"true validation pairs never candidates: {len(miss):,} of {len(V.GK):,} ({len(miss) / len(V.GK):.2%})")

ranks = np.zeros((len(miss), 3), np.int64)
with torch.no_grad():
    for (ci, s), grp in pd.DataFrame({"i": np.arange(len(miss)), "ct": ctry[mq], "s": src[mc]}).groupby(["ct", "s"]):
        pool = np.flatnonzero((ctry == ci) & (src == s))
        Pn = torch.from_numpy(np.ascontiguousarray(EN[pool])).cuda()
        Pa = torch.from_numpy(np.ascontiguousarray(EA[pool])).cuda()
        idx = grp.i.to_numpy()
        for a in range(0, len(idx), 256):
            ii = idx[a:a + 256]
            qn = torch.from_numpy(np.ascontiguousarray(EN[mq[ii]])).cuda()
            qa = torch.from_numpy(np.ascontiguousarray(EA[mq[ii]])).cuda()
            cn = torch.from_numpy(np.ascontiguousarray(EN[mc[ii]])).cuda()
            ca = torch.from_numpy(np.ascontiguousarray(EA[mc[ii]])).cuda()
            sn, sa = qn @ Pn.T, qa @ Pa.T
            tn = (qn * cn).sum(1, keepdim=True)
            ta = (qa * ca).sum(1, keepdim=True)
            ranks[ii, 0] = (sn > tn).sum(1).cpu().numpy()
            ranks[ii, 1] = (sa > ta).sum(1).cpu().numpy()
            ranks[ii, 2] = ((sn + sa) > (tn + ta)).sum(1).cpu().numpy()
        del Pn, Pa
        torch.cuda.empty_cache()
best = ranks.min(1)
blank = tb.addr.to_numpy()[mc] == ""
bins = [-1, 14, 29, 59, 99, 299, 999, 10 ** 9]
labels = ["<15 (already in k)", "15-29", "30-59", "60-99", "100-299", "300-999", ">=1000"]
d = pd.DataFrame({"best_rank": pd.cut(best, bins, labels=labels), "name": pd.cut(ranks[:, 0], bins, labels=labels),
                  "addr": pd.cut(ranks[:, 1], bins, labels=labels), "comb": pd.cut(ranks[:, 2], bins, labels=labels),
                  "blank": blank})
print("\nrank of the missed true record in its best channel (0 = top):")
print(d.best_rank.value_counts(sort=False).to_string())
print("\nper channel:")
print(pd.DataFrame({ch: d[ch].value_counts(sort=False) for ch in ("name", "addr", "comb")}).to_string())
print("\nblank-address share by best-rank bucket:")
print(d.groupby("best_rank", observed=True).blank.mean().round(3).to_string())
np.save(r"D:\AmazonMLChallenge2026\work\v9\blocking_miss_ranks.npy", ranks)
