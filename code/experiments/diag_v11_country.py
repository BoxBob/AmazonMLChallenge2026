# Per-country prediction profile on test (v10) vs the truth profile in train: empty rate, matches per S1, S2/S3 mix
import numpy as np, pandas as pd, csv
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
def prof(s1, res, label):
    d = s1.merge(res, left_on="entity_id", right_on="source1_entity_id", how="left").fillna("")
    lst = d.matched_entity_ids.str.split(",").map(lambda x: [i for i in x if i])
    n = lst.map(len); n2 = lst.map(lambda x: sum(i.startswith("S2") for i in x)); n3 = n - n2
    out = pd.DataFrame({"country": d.country, "empty": n == 0, "n": n, "n2": n2, "n3": n3, "n>=6": n >= 6})
    print(label); print(out.groupby("country").agg(["mean"]).round(3).droplevel(1, axis=1).to_string()); print()
    return out
prof(rd("dataset/train/train_source1.tsv"), rd("dataset/train/train_ground_truth.tsv"), "TRAIN ground truth")
o = prof(rd("dataset/test/test_source1.tsv"), rd("output/matching_results_v10.tsv"), "TEST predictions v10")
print("TEST n distribution by country:"); print(pd.crosstab(o.country, o.n.clip(0, 8), normalize="index").round(3))
