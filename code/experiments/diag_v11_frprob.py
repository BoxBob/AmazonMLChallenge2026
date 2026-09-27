# Test: distribution of pair probabilities by country (kept pairs from the v10 checkpoints) -> is France's missing recall
# sitting just under the threshold (classification) or absent (blocking)?
import glob, numpy as np, pandas as pd, csv, pyarrow.parquet as pq
k = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(r"work/v10full/test_chunks/kept_*.parquet"))])
rec = pq.read_table(r"work/v2/test_records.parquet", columns=["entity_id", "src"]).to_pandas()
s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", dtype=str, usecols=["entity_id", "country"], quoting=3)
ctry = pd.Series(s1.country.values, index=s1.entity_id).reindex(rec.entity_id).to_numpy()
k["ctry"] = ctry[k.q.to_numpy()]
print(f"kept pairs {len(k):,}; min p {k.p.min():.3f}; thresholds {k.groupby('ctry').t.agg(['min','max']).to_dict()}")
n1 = pd.Series(ctry[rec.src.to_numpy() == 1]).value_counts()
bins = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001]
h = k.groupby(["ctry", pd.cut(k.p, bins)], observed=True).size().unstack(0)
print("pairs per S1 by probability bucket:"); print((h / n1).round(3))
