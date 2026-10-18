import pandas as pd, pyarrow.parquet as pq, numpy as np
from anyascii import anyascii
D = r"D:\AmazonMLChallenge2026\dataset\train"
gt = pd.read_csv(D + "/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3, nrows=200000)
s1 = pd.read_csv(D + "/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3).set_index("entity_id")
s2 = pd.read_csv(D + "/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3).set_index("entity_id")
m = gt.assign(c=gt.matched_entity_ids.str.split(",")).explode("c"); m = m[m.c.str.startswith("S2")]
m = m[m.c.isin(s2.index)]
m["n1"] = s1.business_name.reindex(m.source1_entity_id).values; m["n2"] = s2.business_name.reindex(m.c).values
m["a2"] = s2.business_address.reindex(m.c).values
nat = m[~m.n2.map(str.isascii)].sample(25, random_state=3)
for r in nat.itertuples(): print(f"{r.n1[:40]:40} | {anyascii(r.n2)[:45]:45} | {anyascii(r.a2)[-35:]}")
print("frac S2 names non-ascii (matched):", (~m.n2.map(str.isascii)).mean())
print("frac S2 addr non-ascii:", (~m.a2.map(str.isascii)).mean())
