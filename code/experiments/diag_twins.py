# Label-free: per S1 entity, how many pool records carry the same name (and the same name + house number)?
# "same name, different house number" = the near-duplicate distractor pattern (P07).
import numpy as np, pandas as pd, pyarrow.parquet as pq
SH = r"D:\AmazonMLChallenge2026\work\v2"
for split in ("train", "test"):
    df = pq.read_table(SH + rf"\{split}_records.parquet", columns=["country", "src", "squash", "hn"]).to_pandas()
    df = df[df.squash.str.len() >= 4]
    pool = df[df.src != 1]
    kn = (pool.country + "|" + pool.squash).value_counts()
    knh = (pool.country + "|" + pool.squash + "|" + pool.hn).value_counts()
    s1 = df[df.src == 1]
    n_name = (s1.country + "|" + s1.squash).map(kn).fillna(0)
    n_nh = (s1.country + "|" + s1.squash + "|" + s1.hn).map(knh).fillna(0).where(s1.hn != "", 0)
    has_hn = s1.hn != ""
    out = pd.DataFrame({"country": s1.country, "same_name": n_name, "same_name_same_hn": n_nh,
                        "same_name_diff_hn": (n_name - n_nh).where(has_hn)})
    print(f"--- {split}: mean pool records per S1 entity")
    print(out.groupby("country").mean().round(3).to_string())
