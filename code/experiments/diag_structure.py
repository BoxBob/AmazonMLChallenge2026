# Why can the top 100 reach 0.991 when our pairwise model looks near its information limit? Check two things on the
# labelled training data:
#  1) data artifacts: does the ORDER of rows or the ID NUMBERS carry cluster membership? (if so, top scores may come
#     from exploiting it; we flag it but do not rely on it: fair-play review of top packages)
#  2) cluster structure our pairwise design ignores: S1 clusters with records in BOTH S2 and S3, and how often a
#     blank-address true match has an *addressed* sibling in the same cluster (evidence we never use directly)
import csv

import numpy as np
import pandas as pd

D = r"D:\AmazonMLChallenge2026\dataset\train"


def rd(p):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


s1, s2, s3 = rd(D + "/train_source1.tsv"), rd(D + "/train_source2.tsv"), rd(D + "/train_source3.tsv")
gt = rd(D + "/train_ground_truth.tsv")
m = gt.assign(c=gt.matched_entity_ids.str.split(",")).explode("c")
m = m[m.c != ""][["source1_entity_id", "c"]]
pos = pd.concat([pd.Series(np.arange(len(d)), index=d.entity_id) for d in (s1, s2, s3)])
num = lambda s: s.str.split("-").str[1].astype(np.int64)

m2 = m[m.c.str.startswith("S2")].sample(300_000, random_state=0)
a, b = pos.reindex(m2.source1_entity_id).to_numpy(), pos.reindex(m2.c).to_numpy()
print("1) ARTIFACTS")
print(f"   corr(S1 row position, matched S2 row position) = {np.corrcoef(a / len(s1), b / len(s2))[0, 1]:+.4f}")
ia, ib = num(m2.source1_entity_id).to_numpy(), num(m2.c).to_numpy()
print(f"   corr(S1 id number, matched S2 id number)      = {np.corrcoef(ia, ib)[0, 1]:+.4f} "
      f"(spearman {pd.Series(ia).corr(pd.Series(ib), method='spearman'):+.4f})")
g = m[m.c.str.startswith("S2")].assign(p=lambda d: pos.reindex(d.c).to_numpy()).groupby("source1_entity_id").p
sz = g.size()
multi = g.apply(lambda s: np.diff(np.sort(s.to_numpy()))) [sz >= 2]
gaps = np.concatenate(multi.head(150_000).to_numpy())
print(f"   rows between S2 records of the same cluster: median {np.median(gaps):,.0f}, share <=10 rows {np.mean(gaps <= 10):.2%} "
      f"(random: median ~{len(s2) / 3:,.0f})")

print("2) CLUSTER STRUCTURE")
k = m.assign(src=m.c.str[:2]).groupby(["source1_entity_id", "src"]).size().unstack(fill_value=0)
print(f"   S1 clusters with >=1 S2 AND >=1 S3 record: {((k['S2'] > 0) & (k['S3'] > 0)).mean():.1%}; "
      f"with >=2 records total: {(k.sum(1) >= 2).mean():.1%}")
addr = pd.concat([s2.set_index("entity_id").business_address, s3.set_index("entity_id").business_address])
mm = m.assign(blank=addr.reindex(m.c).to_numpy() == "")
addressed = mm[~mm.blank].groupby("source1_entity_id").size()
bl = mm[mm.blank]
has = bl.source1_entity_id.map(addressed).fillna(0) > 0
print(f"   blank-address true matches: {len(bl):,}; their cluster also holds an ADDRESSED record: {has.mean():.1%}")
name = pd.concat([s2.set_index("entity_id").business_name, s3.set_index("entity_id").business_name])
# within-cluster S2<->S3 name agreement vs S1<->S2/S3 name agreement (is S2-S3 easier to link than S1-S2?)
from rapidfuzz import fuzz, process
smp = m.sample(200_000, random_state=1)
s1name = s1.set_index("entity_id").business_name
pairs = smp.merge(smp, on="source1_entity_id")
pairs = pairs[(pairs.c_x < pairs.c_y) & (pairs.c_x.str[:2] != pairs.c_y.str[:2])].head(100_000)
sim_ss = process.cpdist(list(name.reindex(pairs.c_x)), list(name.reindex(pairs.c_y)), scorer=fuzz.token_set_ratio, workers=-1)
sim_1s = process.cpdist(list(s1name.reindex(smp.source1_entity_id)), list(name.reindex(smp.c)), scorer=fuzz.token_set_ratio, workers=-1)
print(f"   name similarity S2<->S3 within a cluster: median {np.median(sim_ss):.0f}, <60 in {np.mean(sim_ss < 60):.1%} | "
      f"S1<->S2/S3: median {np.median(sim_1s):.0f}, <60 in {np.mean(sim_1s < 60):.1%}")
