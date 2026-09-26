# Inspect French silver pairs (same name + same house number) that v3 rejected, with the raw records.
import numpy as np, pandas as pd, pyarrow.parquet as pq, csv
SH = r"D:\AmazonMLChallenge2026\work\v2"; D = r"D:\AmazonMLChallenge2026\dataset\test"
te = pq.read_table(SH + r"\test_records.parquet", columns=["entity_id", "country", "src", "squash", "hn", "core", "addr"]).to_pandas()
raw = pd.concat([pd.read_csv(f"{D}/test_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE) for s in (1, 2, 3)]).set_index("entity_id")
fr = te[(te.country == "France") & (te.squash.str.len() >= 4) & (te.hn != "")]
k = fr.squash + "|" + fr.hn
sp = pd.DataFrame({"k": k[fr.src == 1], "q": fr.entity_id[fr.src == 1]}).merge(pd.DataFrame({"k": k[fr.src != 1], "c": fr.entity_id[fr.src != 1]}), on="k")
res = pd.read_csv(r"D:\AmazonMLChallenge2026\output\matching_results_v3.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
pred = res.set_index("source1_entity_id").matched_entity_ids.map(lambda s: set(s.split(",")) if s else set())
sp["acc"] = [c in pred[q] for q, c in zip(sp.q, sp.c)]
sp["n_s1_same_key"] = sp.k.map(k[fr.src == 1].value_counts())
print(f"France silver pairs {len(sp):,}, accepted {sp.acc.mean():.3f}")
print("acceptance by how many S1 entities share that name+number:")
print(sp.groupby(pd.cut(sp.n_s1_same_key, [0, 1, 2, 5, 1000])).acc.agg(["mean", "size"]).round(3).to_string())
one = sp[(sp.n_s1_same_key == 1) & ~sp.acc]
print(f"\nrejected silver pairs whose name+number is UNIQUE among French S1: {len(one):,} ({len(one)/len(sp):.1%} of all French silver pairs)")
for r in one.sample(15, random_state=1).itertuples():
    a, b = raw.loc[r.q], raw.loc[r.c]
    print(f"  S1 {a.business_name[:34]:34} | {a.business_address[:55]}")
    print(f"     {b.business_name[:34]:34} | {b.business_address[:55]}   [{r.c[:2]}]  S1 has {len(pred[r.q])} matches")

cand = pd.read_csv(r"D:\AmazonMLChallenge2026\output\candidate_pairs_v3.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
cand = cand.set_index("source1_entity_id").candidate_entity_ids
fr_q = set(sp.q)
cmap = {q: set(cand[q].split(",")) for q in fr_q}
sp["in_cands"] = [c in cmap[q] for q, c in zip(sp.q, sp.c)]
print("\nFrench silver pairs: in candidate set?  (by acceptance and S1 uniqueness of name+number)")
sp["uniq"] = sp.n_s1_same_key == 1
print(sp.groupby(["uniq", "acc"]).in_cands.agg(["mean", "size"]).round(3).to_string())
# same breakdown for US / India for comparison
