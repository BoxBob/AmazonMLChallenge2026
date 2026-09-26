# Are the extra test pool records near-duplicate distractors? Label-free comparison of train vs test.
# For every S2/S3 record: does its (country, squashed name) equal some S1 entity's? On train we can split this by
# matched vs unmatched (distractor) records; on test we only see the mixture.
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SH = r"D:\AmazonMLChallenge2026\work\v2"
D = r"D:\AmazonMLChallenge2026\dataset"
cols = ["entity_id", "country", "src", "squash", "hn", "addr"]
tr = pq.read_table(SH + r"\train_records.parquet", columns=cols).to_pandas()
te = pq.read_table(SH + r"\test_records.parquet", columns=cols).to_pandas()

gt = pd.read_csv(D + r"\train\train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
m = gt.assign(c=gt.matched_entity_ids.str.split(",")).explode("c")
matched = set(m.c[m.c != ""])
owner = dict(zip(m.c, m.source1_entity_id))


def stats(df, label=None):
    s1 = df[df.src == 1]
    key = df.country + "|" + df.squash
    s1keys = set(key[df.src == 1])
    s1_hn = set((df.country + "|" + df.squash + "|" + df.hn)[df.src == 1])
    pool = df[df.src != 1].copy()
    pool["name_hits_s1"] = key[df.src != 1].isin(s1keys)
    pool["name_hn_hits_s1"] = (pool.country + "|" + pool.squash + "|" + pool.hn).isin(s1_hn)
    return pool


ptr = stats(tr)
ptr["matched"] = ptr.entity_id.isin(matched)
print("TRAIN pool records: share whose exact name equals some S1 name  (and name+house number)")
print(ptr.groupby("matched")[["name_hits_s1", "name_hn_hits_s1"]].mean().round(3).to_string())
print(f"train: matched share {ptr.matched.mean():.3f}")
# for unmatched records whose name hits an S1 name: is the house number different? (= near-duplicate distractor)
u = ptr[~ptr.matched & ptr.name_hits_s1]
print(f"train unmatched with an S1 name: {len(u):,} ({len(u) / (~ptr.matched).sum():.1%} of distractors); "
      f"of those, same name AND same house number as that S1: {u.name_hn_hits_s1.mean():.1%}")

pte = stats(te)
print("\nTEST pool records (mixture of matches and distractors), per country:")
print(pte.groupby("country")[["name_hits_s1", "name_hn_hits_s1"]].mean().round(3).to_string())
print("TRAIN pool records, per country (for comparison):")
print(ptr.groupby("country")[["name_hits_s1", "name_hn_hits_s1"]].mean().round(3).to_string())
# implied distractor share on test if matched/unmatched name-hit rates equal train's
a, b = ptr[ptr.matched].name_hits_s1.mean(), ptr[~ptr.matched].name_hits_s1.mean()
for ctry, g in pte.groupby("country"):
    x = g.name_hits_s1.mean()
    print(f"{ctry}: implied distractor share = {(a - x) / (a - b):.2f}  (train overall {1 - ptr.matched.mean():.2f})")

print("\n=== per-country estimate with country-specific train rates, two independent fingerprints")
for ctry in ("India", "US"):
    g = ptr[ptr.country == ctry]
    for col in ("name_hits_s1", "name_hn_hits_s1"):
        a, b = g[g.matched][col].mean(), g[~g.matched][col].mean()
        x = pte[pte.country == ctry][col].mean()
        share = (a - x) / (a - b)
        n_s1 = (te[(te.country == ctry)].src == 1).sum(); n_pool = len(pte[pte.country == ctry])
        print(f"{ctry:6s} {col:16s}: train matched {a:.3f} / distractor {b:.3f}; test {x:.3f} -> distractor share {share:.2f}, "
              f"implied true matches per S1 {n_pool * (1 - share) / n_s1:.2f} (train {g.matched.sum() / (tr[(tr.country == ctry)].src == 1).sum():.2f})")
