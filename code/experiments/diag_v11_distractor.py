# Blank-address zone candidates: are the false ones members of ANOTHER cluster, or orphans (matched to no S1)?
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
import valscore as V
W11 = r"D:\AmazonMLChallenge2026\work\v11"
vz = np.load(r"D:\AmazonMLChallenge2026\work\v10m80\val_pairs.npz")
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
p2 = np.load(W11 + r"\val_p2_v10.npy"); pce = np.load(W11 + r"\val_pce_v9_narrow.npy")
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
band = np.flatnonzero((p2 >= 0.01) & (p2 <= 0.99)); Xb = np.column_stack([lg(p2[band]), lg(pce)])
pb = p2.copy(); pb[band] = LogisticRegression(C=10.0).fit(Xb, vy[band]).predict_proba(Xb)[:, 1]
rec = pq.read_table(V.W + r"\train_records.parquet", columns=["addr", "src", "name"]).to_pandas()
blank = (rec.addr.to_numpy() == "")
# full train ground truth: record -> owning S1
import csv
gt = pd.read_csv(V.D + "/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
m = gt.assign(c=gt.matched_entity_ids.str.split(",")).explode("c"); m = m[m.c != ""]
ids = pd.Index(rec.entity_id if "entity_id" in rec else pq.read_table(V.W + r"\train_records.parquet", columns=["entity_id"]).to_pandas().entity_id)
owner = np.full(len(rec), -1, np.int64); owner[ids.get_indexer(m.c)] = ids.get_indexer(m.source1_entity_id)
s = rec.src.to_numpy()
for k in (2, 3):
    print(f"S{k}: records {np.sum(s == k):,}; orphans (no S1 owner) {np.mean(owner[s == k] < 0):.1%}; "
          f"blank-address {np.mean(blank[s == k]):.1%}; orphans among blank {np.mean(owner[(s == k) & blank] < 0):.1%}")
z = (pb > 0.05) & (pb < 0.95) & blank[vc]
f = z & (vy == 0)
o = owner[vc[f]]
print(f"blank-address zone pairs {z.sum():,} ({vy[z].mean():.1%} true); false ones: orphan {np.mean(o < 0):.1%}, "
      f"owned by another S1 {np.mean(o >= 0):.1%}")
oo = o[o >= 0]
print(f"   ...of those, the other owner is in the S1 pool (validation or train S1s): all are S1 records by construction; "
      f"other owner is a validation entity {np.isin(oo, V.VAL).mean():.1%}")
# name identity of false candidates vs the S1
nm = rec.name.to_numpy()
same = np.array([nm[a] == nm[b] for a, b in zip(vq[z], vc[z])])
print(f"exact-name blank zone pairs: {same.sum():,}, true rate {vy[z][same].mean():.1%}; non-exact true rate {vy[z][~same].mean():.1%}")
from rapidfuzz import fuzz, process
core = pq.read_table(V.W + r"\train_records.parquet", columns=["core"])["core"].to_numpy(zero_copy_only=False)
fz = np.flatnonzero(f); fz = fz[owner[vc[fz]] >= 0]
sq_ = process.cpdist(list(core[vq[fz]]), list(core[vc[fz]]), scorer=fuzz.token_set_ratio, workers=-1)
so_ = process.cpdist(list(core[owner[vc[fz]]]), list(core[vc[fz]]), scorer=fuzz.token_set_ratio, workers=-1)
sr_ = process.cpdist(list(core[vq[fz]]), list(core[vc[fz]]), scorer=fuzz.ratio, workers=-1)
sro = process.cpdist(list(core[owner[vc[fz]]]), list(core[vc[fz]]), scorer=fuzz.ratio, workers=-1)
print(f"false blank zone candidates with a known owner: {len(fz):,}")
print(f"  token_set: owner more similar {np.mean(so_ > sq_):.1%}, tie {np.mean(so_ == sq_):.1%}, query more similar {np.mean(so_ < sq_):.1%}")
print(f"  ratio    : owner more similar {np.mean(sro > sr_):.1%}, tie {np.mean(sro == sr_):.1%}, query more similar {np.mean(sro < sr_):.1%}")
for i in fz[:10]:
    print(f"   c={core[vc[i]][:35]:35s} | q={core[vq[i]][:35]:35s} | owner={core[owner[vc[i]]][:35]}")
