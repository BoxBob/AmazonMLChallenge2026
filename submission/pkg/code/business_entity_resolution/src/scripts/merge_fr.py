# Submission = US/India rows from a new run + France rows from the 0.974/0.975 France file (France changes only after a
# leaderboard check).  usage: python merge_fr.py <new_raw.tsv> <out.tsv>
import csv, sys, pandas as pd, pyarrow.parquet as pq
OUT = r"D:\AmazonMLChallenge2026\output"
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
ctry = pq.read_table(r"D:\AmazonMLChallenge2026\work\v2\test_records.parquet", columns=["entity_id", "country"]).to_pandas().set_index("entity_id").country
new, best = rd(sys.argv[1]), rd(OUT + r"\lb_probe\matching_results_fr_noswap_f300.tsv")
assert (new.source1_entity_id.to_numpy() == best.source1_entity_id.to_numpy()).all()
is_fr = best.source1_entity_id.map(ctry) == "France"
out = best.copy()
out.loc[~is_fr, "matched_entity_ids"] = new.loc[~is_fr, "matched_entity_ids"].to_numpy()
out.to_csv(sys.argv[2], sep="\t", index=False, quoting=csv.QUOTE_NONE)
print(f"{sys.argv[2]} written ({is_fr.sum():,} French rows from the France file, {(~is_fr).sum():,} US/India rows from the new run)")
