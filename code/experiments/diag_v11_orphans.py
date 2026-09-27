# Unassigned S2/S3 records per country: truth (train) vs our predictions (test v10). A France excess = France recall loss.
import numpy as np, pandas as pd, csv
rd = lambda p, **k: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, **k)
for split, res in (("train", "dataset/train/train_ground_truth.tsv"), ("test", "output/matching_results_v10.tsv")):
    r = rd(res)
    used = set(x for s in r.matched_entity_ids for x in s.split(",") if x)
    for src in (2, 3):
        d = rd(f"dataset/{split}/{split}_source{src}.tsv")
        d["assigned"] = d.entity_id.isin(used)
        d["blank"] = d.business_address.str.strip() == ""
        g = d.groupby("country").agg(records=("assigned", "size"), assigned=("assigned", "mean"), blank=("blank", "mean"))
        print(split, f"S{src}"); print(g.round(3).to_string())
        if split == "test" and src == 2:
            fr = d[(d.country == "France") & ~d.assigned].sample(40, random_state=0)
            fr[["entity_id", "business_name", "business_address"]].to_csv("work/v11/fr_unassigned_s2_sample.tsv", sep="\t", index=False)
