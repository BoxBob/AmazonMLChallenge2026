# France silver misses: pairs S1<->S2/S3 with the same (country, squashed full name, house number), the key unique among
# S1, that the submission does NOT predict. Why: owned by another S1 / not predicted at all? Examples.
import numpy as np, pandas as pd, pyarrow.parquet as pq, csv
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
t = pq.read_table(r"D:\AmazonMLChallenge2026\work\v2\test_records.parquet",
                  columns=["entity_id", "country", "src", "squash", "hn", "core", "addr"]).to_pandas()
t["k"] = np.where((t.squash.str.len() >= 4) & (t.hn != ""), t.country + "|" + t.squash + "|" + t.hn, "")
raw = pd.concat([rd(f"D:/AmazonMLChallenge2026/dataset/test/test_source{s}.tsv") for s in (1, 2, 3)]).set_index("entity_id")
s1 = t[(t.src == 1) & (t.k != "")]
u = s1.k.value_counts(); s1 = s1[s1.k.map(u) == 1]
pool = t[(t.src != 1) & (t.k != "")]
sil = s1[["entity_id", "k", "country"]].merge(pool[["entity_id", "k"]], on="k", suffixes=("_q", "_c"))
for f, lab in (("output/matching_results_v10.tsv", "v10 (France rows = 0.974 file)"),
               ("output/matching_results_v10_frnew.tsv", "v10_frnew")):
    r = rd("D:/AmazonMLChallenge2026/" + f)
    ex = r.assign(c=r.matched_entity_ids.str.split(",")).explode("c")
    ex = ex[ex.c.notna() & (ex.c != "")]
    own = pd.Series(ex.source1_entity_id.values, index=ex.c.values)
    own = own[~own.index.duplicated()]
    pred = set(zip(ex.source1_entity_id, ex.c))
    sil["acc"] = [(a, b) in pred for a, b in zip(sil.entity_id_q, sil.entity_id_c)]
    sil["other_owner"] = sil.entity_id_c.map(own).fillna("")
    fr = sil[sil.country == "France"]
    miss = fr[~fr.acc]
    print(f"{lab}: France silver pairs {len(fr):,}, predicted {fr.acc.mean():.4f}; misses {len(miss):,}: "
          f"owned by another S1 {np.mean(miss.other_owner != ''):.1%}, unassigned {np.mean(miss.other_owner == ''):.1%}")
    for c_ in ("US", "India"):
        x = sil[sil.country == c_]
        print(f"   {c_}: predicted {x.acc.mean():.4f}; misses owned by another S1 {np.mean(x[~x.acc].other_owner != ''):.1%}")
miss = miss.sample(min(25, len(miss)), random_state=0)
for _, m in miss.iterrows():
    q, c, o = raw.loc[m.entity_id_q], raw.loc[m.entity_id_c], m.other_owner
    print(f"\nS1 {q.business_name[:45]:45s} | {q.business_address[:60]}")
    print(f"   {m.entity_id_c[:2]} {c.business_name[:45]:45s} | {c.business_address[:60]}")
    if o:
        print(f"   OWNER {raw.loc[o].business_name[:42]:42s} | {raw.loc[o].business_address[:60]}")
