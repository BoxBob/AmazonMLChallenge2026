# v6 experiment S1: "name-sibling" evidence for blank-address candidates (pitfall P11, the largest v4 loss bucket).
# First principles: records form clusters (one real business, records in S2 and S3). If blank-address record c truly
# belongs to S1 entity q, the other pool records carrying c's name (its siblings) that HAVE an address should sit at
# q's address; if c is a distractor, its siblings sit elsewhere (another business with the same name).
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

import valscore as V

SH = r"D:\AmazonMLChallenge2026\work\v2"
W4 = r"D:\AmazonMLChallenge2026\work\v4"
SIB_CAP = 30

tb = pq.read_table(SH + r"\train_records.parquet", columns=["country", "src", "squash", "addr"]).combine_chunks()
df = tb.to_pandas()
key = (df.country + "|" + df.squash).where(df.squash.str.len() >= 3, "")
addr = df.addr.to_numpy()
pool_addr = (df.src != 1) & (df.addr != "") & (key != "")
# name key -> sibling record indices that have an address (capped)
_idx = np.flatnonzero(pool_addr.to_numpy())
_k = key.to_numpy()[_idx]
_o = np.argsort(_k, kind="stable")
_k, _idx = _k[_o], _idx[_o]
_starts = np.flatnonzero(np.r_[True, _k[1:] != _k[:-1]])
_ends = np.r_[_starts[1:], len(_k)]
sib = pd.Series([_idx[a:min(b, a + SIB_CAP)] for a, b in zip(_starts, _ends)], index=_k[_starts])


def sibling_feats(q, c):
    """for each pair: #siblings of c with an address, #at q's address (addr sim>=80), #clearly elsewhere (<50), max sim"""
    kc = key.to_numpy()[c]
    out = np.zeros((len(q), 4), np.float32)
    lists = sib.reindex(kc).to_numpy()
    rows_q, rows_s, owner = [], [], []
    for i, (lst, ci) in enumerate(zip(lists, c)):
        if isinstance(lst, np.ndarray):
            lst = lst[lst != ci]
            rows_q.extend([q[i]] * len(lst)); rows_s.extend(lst.tolist()); owner.extend([i] * len(lst))
    if not rows_q:
        return out
    sims = process.cpdist(list(addr[rows_q]), list(addr[rows_s]), scorer=fuzz.token_set_ratio, workers=-1)
    d = pd.DataFrame({"i": owner, "s": sims})
    g = d.groupby("i").s
    out[:, 0] = g.size().reindex(range(len(q)), fill_value=0).to_numpy()
    out[:, 1] = (d.s >= 80).groupby(d.i).sum().reindex(range(len(q)), fill_value=0).to_numpy()
    out[:, 2] = (d.s < 50).groupby(d.i).sum().reindex(range(len(q)), fill_value=0).to_numpy()
    out[:, 3] = g.max().reindex(range(len(q)), fill_value=-1).to_numpy()
    return out


vz = np.load(W4 + r"\val_pairs.npz")
q, c, y = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
pb = np.load(W4 + r"\val_p_v4.npy")
blank_c = addr[c] == ""
sel = np.flatnonzero(blank_c & (pb >= 0.01))                 # blank-address candidates the model is not sure are wrong
F = sibling_feats(q[sel], c[sel])
d = pd.DataFrame({"true": y[sel] == 1, "p": pb[sel], "n_sib": F[:, 0], "at_q": F[:, 1], "elsewhere": F[:, 2], "max_sim": F[:, 3]})
d["bucket"] = np.select([d.n_sib == 0, (d.at_q > 0) & (d.elsewhere == 0), (d.at_q > 0) & (d.elsewhere > 0), d.at_q == 0],
                        ["no siblings with address", "siblings only at q's address", "siblings at q AND elsewhere",
                         "siblings only elsewhere"], "other")
print(f"blank-address candidate pairs with p>=0.01: {len(d):,}; true rate {d.true.mean():.3f}")
print(d.groupby("bucket").agg(pairs=("true", "size"), true_rate=("true", "mean"), mean_p=("p", "mean"),
                              accepted=("p", lambda s: (s >= 0.70).mean())).round(3).to_string())
