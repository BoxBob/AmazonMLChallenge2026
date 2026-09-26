# Label-free silver report for any number of submission files: share of silver pairs (same country + full squashed
# name + house number, the key unique among that country's S1 entities) that each submission predicts, per country.
# On labelled validation, true silver pairs are accepted 99.7-100%; a low value flags a blocking/transfer failure.
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SH = r"D:\AmazonMLChallenge2026\work\v2"
te = pq.read_table(SH + r"\test_records.parquet", columns=["entity_id", "country", "src", "squash", "hn"]).to_pandas()
ok = (te.squash.str.len() >= 4) & (te.hn != "")
key = np.where(ok, te.country + "|" + te.squash + "|" + te.hn, "")
s1 = pd.DataFrame({"k": key[te.src == 1], "q": te.entity_id[te.src == 1]})
s1 = s1[s1.k != ""]
s1 = s1[s1.k.map(s1.k.value_counts()) == 1]
pool = pd.DataFrame({"k": key[te.src != 1], "c": te.entity_id[te.src != 1]})
sil = s1.merge(pool[pool.k != ""], on="k")
sil["country"] = sil.k.str.split("|").str[0]
print(f"silver pairs (unique name+number among S1): {len(sil):,}")
for path in sys.argv[1:]:
    res = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    m = res.assign(c=res.matched_entity_ids.str.split(",")).explode("c")
    pred = set((m.source1_entity_id + "|" + m.c).tolist())
    acc = [f"{a}|{b}" in pred for a, b in zip(sil.q, sil.c)]
    n = res.matched_entity_ids.map(lambda s: len(s.split(",")) if s else 0)
    ctry = res.source1_entity_id.map(te.set_index("entity_id").country)
    print(f"\n{path.split(chr(92))[-1]}")
    print("  silver acceptance:", pd.Series(acc).groupby(sil.country.to_numpy()).mean().round(4).to_dict())
    print("  mean matches per S1:", n.groupby(ctry).mean().round(3).to_dict())
    print("  empty share:", (n == 0).groupby(ctry).mean().round(4).to_dict())
