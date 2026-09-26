# Does French address normalization penalise true French pairs? Compare pairs that are almost certainly the same
# business (same full name + same house number + same street word) across countries: how similar do their normalized
# addresses look, and which tokens cause the mismatch?
import csv
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

SH = r"D:\AmazonMLChallenge2026\work\v2"
OUT = r"D:\AmazonMLChallenge2026\output"
STOP = {"r", "rue", "st", "ave", "blvd", "all", "imp", "rte", "ch", "des", "les", "du", "de", "la", "le", "no", "n"}


def street(addr, hn):
    t = addr.split()
    if hn and hn in t:
        i = t.index(hn)
        for u in t[i + 1:i + 6]:
            if u.isalpha() and len(u) >= 3 and u not in STOP:
                return u
    return ""


te = pq.read_table(SH + r"\test_records.parquet", columns=["entity_id", "country", "squash", "hn", "addr", "src"]).to_pandas()
te["st"] = [street(a, h) for a, h in zip(te.addr, te.hn)]
te = te.set_index("entity_id")
res = pd.read_csv(OUT + r"\lb_probe\matching_results_fr_noswap_f300.tsv", sep="\t", dtype=str, keep_default_na=False,
                  quoting=csv.QUOTE_NONE)
m = res.assign(c=res.matched_entity_ids.str.split(",")).explode("c")
m = m[m.c != ""].sample(400_000, random_state=0)
A, B = te.loc[m.source1_entity_id.to_numpy()], te.loc[m.c.to_numpy()]
anchor = (A.squash.to_numpy() == B.squash.to_numpy()) & (A.hn.to_numpy() == B.hn.to_numpy()) & (A.hn.to_numpy() != "") \
         & (A.st.to_numpy() == B.st.to_numpy()) & (A.st.to_numpy() != "")
a_ts = process.cpdist(list(A.addr), list(B.addr), scorer=fuzz.token_set_ratio, workers=-1)
a_r = process.cpdist(list(A.addr), list(B.addr), scorer=fuzz.ratio, workers=-1)
d = pd.DataFrame({"country": A.country.to_numpy(), "anchor": anchor, "a_tset": a_ts, "a_ratio": a_r})
print("ANCHOR pairs (same name + house no. + street, accepted): address similarity by country")
print(d[d.anchor].groupby("country")[["a_tset", "a_ratio"]].describe(percentiles=[.1, .25, .5]).round(1).T.to_string())
# which tokens appear on only one side of French anchor pairs?
cnt = {k: Counter() for k in ("France", "US", "India")}
tot = Counter()
for (ctry, a, b) in zip(A.country.to_numpy()[anchor], A.addr.to_numpy()[anchor], B.addr.to_numpy()[anchor]):
    sa, sb = set(a.split()), set(b.split())
    for t in sa ^ sb:
        cnt[ctry][t] += 1
    tot[ctry] += 1
for ctry in ("France", "US", "India"):
    print(f"\n{ctry}: tokens most often on ONE side only of anchor pairs (share of {tot[ctry]:,} pairs)")
    print(", ".join(f"{t} {n / tot[ctry]:.1%}" for t, n in cnt[ctry].most_common(25)))
# St / Saint and Ste / Sainte in raw French addresses
raw = pd.concat([pd.read_csv(fr"D:\AmazonMLChallenge2026\dataset\test\test_source{s}.tsv", sep="\t", dtype=str,
                             keep_default_na=False, quoting=csv.QUOTE_NONE) for s in (1, 2, 3)])
raw = raw[raw.country == "France"].business_address.str.lower()
for pat in (r"\bst[-\s]", r"\bsaint[-\s]", r"\bste[-\s]", r"\bsainte[-\s]", r"\bbis\b", r"\bter\b", r"\bcedex\b"):
    print(f"French addresses matching {pat!r}: {raw.str.contains(pat).mean():.2%}")
