# v10 = US/India rows from the v10 run (top-80 + multilingual CE, validated 0.9836) + France rows from the 0.974 file.
import csv, pandas as pd, pyarrow.parquet as pq
OUT = r"D:\AmazonMLChallenge2026\output"
ctry = pq.read_table(r"D:\AmazonMLChallenge2026\work\v2\test_records.parquet", columns=["entity_id", "country"]).to_pandas().set_index("entity_id").country
new = pd.read_csv(OUT + r"\lb_probe\matching_results_v10_raw.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
best = pd.read_csv(OUT + r"\lb_probe\matching_results_fr_noswap_f300.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
assert (new.source1_entity_id.to_numpy() == best.source1_entity_id.to_numpy()).all()
is_fr = best.source1_entity_id.map(ctry) == "France"
out = best.copy()
out.loc[~is_fr, "matched_entity_ids"] = new.loc[~is_fr, "matched_entity_ids"].to_numpy()
out.to_csv(OUT + r"\matching_results_v10.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
print(f"v10 written; rows changed vs 0.974 file {(out.matched_entity_ids != best.matched_entity_ids).mean():.2%}")
