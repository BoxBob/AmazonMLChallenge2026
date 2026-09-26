# Diagnose the validation (0.981) vs hidden-test (0.965) gap without test labels.
# Compares, per country, what the v3 model *predicts* on test with what it predicts (and what is true) on validation,
# plus the structural statistics of each split (pool size per S1, blank rates, name ambiguity).
import numpy as np
import pandas as pd
import pyarrow.compute as pac
import pyarrow.parquet as pq

SH = r"D:\AmazonMLChallenge2026\work\v2"
OUT = r"D:\AmazonMLChallenge2026\output"
D = r"D:\AmazonMLChallenge2026\dataset"


def load(split):
    t = pq.read_table(SH + rf"\{split}_records.parquet", columns=["entity_id", "country", "src", "addr", "squash", "native"])
    return t.to_pandas()


tr, te = load("train"), load("test")
print("=== structure per country: pool records per S1, blank addresses, native script, S1 name uniqueness")
rows = []
for name, df in (("train", tr), ("test", te)):
    for ctry, g in df.groupby("country"):
        s1 = g[g.src == 1]
        pool = g[g.src != 1]
        uniq = s1.groupby("squash").size()
        rows.append(dict(split=name, country=ctry, s1=len(s1), pool_per_s1=len(pool) / len(s1),
                         pool_blank_addr=(pool.addr == "").mean(), pool_native=pool.native.mean(),
                         s1_unique_name=(s1.squash.map(uniq) == 1).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))

# truth stats on train (all entities)
gt = pd.read_csv(D + r"\train\train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
gt["n"] = gt.matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))
ctry = tr.set_index("entity_id").country
gt["country"] = gt.source1_entity_id.map(ctry)
print("\n=== TRUE match statistics on train, per country")
print(gt.groupby("country").n.agg(mean="mean", singleton=lambda x: (x == 0).mean()).round(4).to_string())

pred = pd.read_csv(OUT + r"\matching_results_v3.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
pred["n"] = pred.matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))
pred["country"] = pred.source1_entity_id.map(te.set_index("entity_id").country)
print("\n=== PREDICTED match statistics on test (v3), per country")
print(pred.groupby("country").n.agg(mean="mean", empty=lambda x: (x == 0).mean(),
                                   n1=lambda x: (x == 1).mean(), n4plus=lambda x: (x >= 4).mean()).round(4).to_string())
