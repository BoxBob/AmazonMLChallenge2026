# Why did v5 (France self-training) not beat v4 on the leaderboard (0.972 vs 0.973)?
# v5 changed only French decisions. Compare the pairs v5 ADDED and REMOVED relative to v4 by the attributes that
# identify a French business (legal form, house number, street, postcode), with raw examples.
import csv

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SH = r"D:\AmazonMLChallenge2026\work\v2"
OUT = r"D:\AmazonMLChallenge2026\output"
D = r"D:\AmazonMLChallenge2026\dataset\test"
te = pq.read_table(SH + r"\test_records.parquet",
                   columns=["entity_id", "country", "src", "squash", "suffix", "hn", "pc", "addr"]).to_pandas()
te = te.set_index("entity_id")
raw = pd.concat([pd.read_csv(f"{D}/test_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False,
                             quoting=csv.QUOTE_NONE) for s in (1, 2, 3)]).set_index("entity_id")


def pairs(path):
    r = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    r = r[r.source1_entity_id.map(te.country) == "France"]
    m = r.assign(c=r.matched_entity_ids.str.split(",")).explode("c")
    m = m[m.c != ""]
    return set(zip(m.source1_entity_id, m.c))


v4, v5 = pairs(OUT + r"\matching_results_v4.tsv"), pairs(OUT + r"\matching_results_v5.tsv")
added, removed, kept = v5 - v4, v4 - v5, v4 & v5
print(f"France pairs: v4 {len(v4):,}  v5 {len(v5):,}  kept {len(kept):,}  ADDED {len(added):,}  REMOVED {len(removed):,}")


def street_word(addr, hn):
    t = addr.split()
    if hn in t:
        i = t.index(hn)
        for u in t[i + 1:i + 6]:
            if u.isalpha() and len(u) >= 3 and u not in {"rue", "ave", "blvd", "all", "imp", "rte", "ch", "des", "les", "du", "de", "la", "le"}:
                return u
    return ""


def describe(ps, name):
    q = [a for a, b in ps]
    c = [b for a, b in ps]
    A, B = te.loc[q], te.loc[c]
    same = lambda col: np.mean((A[col].to_numpy() == B[col].to_numpy()) & (A[col].to_numpy() != ""))
    miss = lambda col: np.mean((A[col].to_numpy() == "") | (B[col].to_numpy() == ""))
    sw = [street_word(a, h) for a, h in zip(A.addr, A.hn)]
    swc = [street_word(a, h) for a, h in zip(B.addr, B.hn)]
    st_same = np.mean([(x == y) and x != "" for x, y in zip(sw, swc)])
    # how many OTHER S1 entities share the S1's squashed name (templated-name ambiguity)
    s1 = te[te.src == 1]
    amb = pd.Series(A.squash.to_numpy()).map(s1.squash.value_counts()).mean()
    print(f"{name:8s} n={len(ps):7,}  same name {same('squash'):.3f} | same legal form {same('suffix'):.3f} "
          f"(missing {miss('suffix'):.3f}) | same house no. {same('hn'):.3f} (missing {miss('hn'):.3f}) | "
          f"same street word {st_same:.3f} | same postcode {same('pc'):.3f} | S1 entities sharing the name {amb:.1f} | "
          f"cand blank addr {np.mean(B.addr.to_numpy() == ''):.3f}")


rng = np.random.default_rng(0)
for name, ps in (("kept", kept), ("ADDED", added), ("REMOVED", removed)):
    ps = list(ps)
    describe([ps[i] for i in rng.choice(len(ps), min(20000, len(ps)), replace=False)], name)
for name, ps in (("ADDED", added), ("REMOVED", removed)):
    ps = list(ps)
    print(f"\n--- {name} examples")
    for i in rng.choice(len(ps), 14, replace=False):
        q, c = ps[i]
        a, b = raw.loc[q], raw.loc[c]
        print(f"  S1 {a.business_name[:32]:32} | {a.business_address[:58]}")
        print(f"  {c[:2]} {b.business_name[:32]:32} | {b.business_address[:58]}")
