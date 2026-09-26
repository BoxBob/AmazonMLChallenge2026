# v7 analysis: "cluster competition" evidence for same-address, different-name pairs (the French descriptor-swap problem).
# First principles: every record belongs to one business cluster (records in S2/S3, sometimes an S1). For a pair (q, c)
# at the same address with different names, look at c's and q's *name mates*: other records carrying exactly the same
# name at the same address. If both q and c have their own mates, there are two separate clusters at the address, so c is
# probably a different business. If c's name exists nowhere else, c is more likely a noisy variant of q's business.
# Also: does *another S1 entity* carry c's name at that address (then c has its own owner)?
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SH = r"D:\AmazonMLChallenge2026\work\v2"
W4 = r"D:\AmazonMLChallenge2026\work\v4"
STOP = {"r", "rue", "st", "ave", "blvd", "all", "imp", "rte", "ch", "des", "les", "du", "de", "la", "le", "no", "n", "rd",
        "dr", "ln", "the", "of", "main", "road", "cross", "street", "avenue"}


def street(addr, hn):
    t = addr.split()
    if hn and hn in t:
        i = t.index(hn)
        for u in t[i + 1:i + 6]:
            if u.isalpha() and len(u) >= 3 and u not in STOP:
                return u
    return ""


def load(split):
    df = pq.read_table(SH + rf"\{split}_records.parquet",
                       columns=["entity_id", "country", "src", "squash", "core", "hn", "addr"]).to_pandas()
    df["st"] = [street(a, h) for a, h in zip(df.addr, df.hn)]
    df["akey"] = np.where((df.hn != "") & (df.st != ""), df.country + "|" + df.hn + "|" + df.st, "")
    df["nakey"] = np.where((df.akey != "") & (df.squash.str.len() >= 3), df.akey + "|" + df.squash, "")
    return df


def cluster_feats(df, q, c):
    """per pair: mates of c's name at the address (pool, excl. c), mates of q's name at the address (pool),
    another S1 entity with c's name at the address, number of distinct pool names at the address"""
    pool = df.src.to_numpy() != 1
    nak = df.nakey.to_numpy()
    ak = df.akey.to_numpy()
    cnt_pool_nak = pd.Series(nak[pool & (nak != "")]).value_counts()
    cnt_s1_nak = pd.Series(nak[~pool & (nak != "")]).value_counts()
    names_at_addr = pd.Series(nak[pool & (nak != "")]).str.rsplit("|", n=1).str[0].value_counts()
    mc = pd.Series(nak[c]).map(cnt_pool_nak).fillna(0).to_numpy() - 1          # excluding c itself
    mq = pd.Series(nak[q]).map(cnt_pool_nak).fillna(0).to_numpy()
    s1c = pd.Series(nak[c]).map(cnt_s1_nak).fillna(0).to_numpy()                 # S1 entities with c's exact name here
    s1c = s1c - (nak[q] == nak[c])                                                # don't count q itself
    nd = pd.Series(ak[c]).map(names_at_addr).fillna(0).to_numpy()
    return mc, mq, s1c, nd


if __name__ == "__main__":
    tr = load("train")
    vz = np.load(W4 + r"\val_pairs.npz")
    q, c, y = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
    same_addr = (tr.akey.to_numpy()[q] == tr.akey.to_numpy()[c]) & (tr.akey.to_numpy()[q] != "")
    diff_name = tr.squash.to_numpy()[q] != tr.squash.to_numpy()[c]
    sel = np.flatnonzero(same_addr & diff_name)
    mc, mq, s1c, nd = cluster_feats(tr, q[sel], c[sel])
    d = pd.DataFrame({"true": y[sel] == 1, "country": tr.country.to_numpy()[q[sel]],
                      "c_mates": np.minimum(mc, 2), "q_mates": np.minimum(mq, 2), "other_s1": s1c > 0})
    d["bucket"] = np.select([d.other_s1, (d.c_mates > 0) & (d.q_mates > 0), (d.c_mates > 0) & (d.q_mates == 0),
                             (d.c_mates == 0) & (d.q_mates > 0)],
                            ["another S1 has c's name here", "both q and c have own mates (2 clusters)",
                             "only c has mates", "only q has mates"], "neither has mates")
    print(f"VALIDATION same-address different-name pairs: {len(d):,}, true rate {d.true.mean():.3f}")
    print(d.groupby(["bucket", "country"]).true.agg(["mean", "size"]).unstack().round(3).to_string())

    te = load("test")
    # France candidate pairs of the same kind, from the v4 candidate file
    import csv
    cand = pd.read_csv(r"D:\AmazonMLChallenge2026\output\lb_probe\candidate_pairs_v4.tsv", sep="\t", dtype=str,
                       keep_default_na=False, quoting=csv.QUOTE_NONE)
    idx = pd.Series(te.index, index=te.entity_id)
    cand = cand[cand.source1_entity_id.map(te.set_index("entity_id").country) == "France"]
    lists = cand.candidate_entity_ids.str.split(",")
    tq = idx.reindex(np.repeat(cand.source1_entity_id.to_numpy(), lists.str.len().to_numpy())).to_numpy()
    tc = idx.reindex(np.concatenate(lists.to_numpy())).to_numpy()
    same_addr = (te.akey.to_numpy()[tq] == te.akey.to_numpy()[tc]) & (te.akey.to_numpy()[tq] != "")
    diff_name = te.squash.to_numpy()[tq] != te.squash.to_numpy()[tc]
    s = np.flatnonzero(same_addr & diff_name)
    mc, mq, s1c, nd = cluster_feats(te, tq[s], tc[s])
    dt = pd.DataFrame({"c_mates": np.minimum(mc, 2), "q_mates": np.minimum(mq, 2), "other_s1": s1c > 0})
    dt["bucket"] = np.select([dt.other_s1, (dt.c_mates > 0) & (dt.q_mates > 0), (dt.c_mates > 0) & (dt.q_mates == 0),
                              (dt.c_mates == 0) & (dt.q_mates > 0)],
                             ["another S1 has c's name here", "both q and c have own mates (2 clusters)",
                              "only c has mates", "only q has mates"], "neither has mates")
    # v4 acceptance of these French pairs
    res = pd.read_csv(r"D:\AmazonMLChallenge2026\output\matching_results_v4.tsv", sep="\t", dtype=str,
                      keep_default_na=False, quoting=3)
    res = res[res.source1_entity_id.map(te.set_index("entity_id").country) == "France"]
    mm = res.assign(c=res.matched_entity_ids.str.split(",")).explode("c")
    pk = set(zip(mm.source1_entity_id, mm.c))
    eid = te.entity_id.to_numpy()
    dt["v4_acc"] = [(a, b) in pk for a, b in zip(eid[tq[s]], eid[tc[s]])]
    print(f"\nTEST FRANCE same-address different-name candidate pairs: {len(dt):,}")
    print(dt.groupby("bucket").agg(pairs=("v4_acc", "size"), v4_accepts=("v4_acc", "mean")).round(3).to_string())
