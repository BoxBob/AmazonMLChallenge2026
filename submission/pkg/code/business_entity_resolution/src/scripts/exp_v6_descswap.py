# v6 France variant: remove "descriptor swap" pairs from v4's France output.
# Evidence: v5 and the France-IDF variant both ADDED same-address pairs whose names differ by a substituted word
# (`OR Club SARL` <-> `OR Ecole SARL`) and both scored lower (0.972, 0.971 vs v4 0.973). So in France the descriptor word
# identifies the business. v4 already accepts many such pairs; removing them is the opposite, untested direction.
# A substitution = a name word replaced by a *different* word (not a typo: no close spelling match on the other side).
import csv
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

SH = r"D:\AmazonMLChallenge2026\work\v2"
OUT = r"D:\AmazonMLChallenge2026\output"
STOP = {"r", "rue", "st", "ave", "blvd", "all", "imp", "rte", "ch", "des", "les", "du", "de", "la", "le", "no", "n"}

te = pq.read_table(SH + r"\test_records.parquet", columns=["entity_id", "country", "core", "hn", "addr"]).to_pandas()
te = te.set_index("entity_id")
BASE = sys.argv[2] if len(sys.argv) > 2 else OUT + r"\matching_results_v4.tsv"   # file to filter
DEST = sys.argv[3] if len(sys.argv) > 3 else None                                 # output path
v4 = pd.read_csv(BASE, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
fr = v4[v4.source1_entity_id.map(te.country) == "France"]
m = fr.assign(c=fr.matched_entity_ids.str.split(",")).explode("c")
m = m[m.c != ""]


def street(addr, hn):
    t = addr.split()
    if hn and hn in t:
        i = t.index(hn)
        for u in t[i + 1:i + 6]:
            if u.isalpha() and len(u) >= 3 and u not in STOP:
                return u
    return ""


A, B = te.loc[m.source1_entity_id.to_numpy()], te.loc[m.c.to_numpy()]
same_addr = np.array([(ha == hb and ha != "" and street(aa, ha) == street(ab, hb) and street(aa, ha) != "")
                      for ha, hb, aa, ab in zip(A.hn, B.hn, A.addr, B.addr)])


FR_CORE = te[te.country == "France"].core.str.split().explode().value_counts()
MIN_FREQ = int(sys.argv[1]) if len(sys.argv) > 1 else 300          # a "vocabulary" word, not a garbled typo
LEGALISH = {"sarl", "sas", "sa", "sasu", "eurl", "sci", "snc", "ei", "eirl", "france", "and", "et", "fils", "associe",
            "associes", "cie", "co", "pvt", "ltd", "inc", "llc", "llp"}


def substitution(a, b):
    """True if the names share a word but each has a word the other lacks that is not a spelling variant"""
    ta, tb = set(a.split()), set(b.split())
    if not (ta & tb):
        return False
    da, db = ta - tb, tb - ta
    if not da or not db:
        return False                                   # one name contains the other: an addition, not a swap
    for x in da:
        if any(Levenshtein.normalized_similarity(x, y) >= 0.75 or x.startswith(y) or y.startswith(x) for y in db):
            return False                               # typo / truncation, not a different word
    da, db = da - LEGALISH, db - LEGALISH
    if not da or not db:
        return False                                   # only legal-form words differ
    return all(FR_CORE.get(t, 0) >= MIN_FREQ for t in da | db)   # both sides are real template vocabulary words


swap = np.array([substitution(a, b) for a, b in zip(A.core, B.core)])
m["swap"] = swap & same_addr
print(f"v4 France accepted pairs: {len(m):,}; descriptor swaps at the same address: {m.swap.sum():,} "
      f"({m.swap.mean():.1%}); French S1 entities affected: {m[m.swap].source1_entity_id.nunique():,}")
rng = np.random.default_rng(0)
ex = m[m.swap].sample(12, random_state=1)
print("examples (would be removed):")
for q, c in zip(ex.source1_entity_id, ex.c):
    print(f"   {te.loc[q].core[:30]:30} | {te.loc[q].addr[:40]:40} <-> {te.loc[c].core[:30]:30} | {te.loc[c].addr[:40]}")

kept = m[~m.swap].groupby("source1_entity_id").c.agg(",".join)
out = v4.copy()
isfr = out.source1_entity_id.map(te.country) == "France"
out.loc[isfr, "matched_entity_ids"] = out.loc[isfr, "source1_entity_id"].map(kept).fillna("").to_numpy()
path = DEST or OUT + rf"\lb_probe\matching_results_fr_noswap_f{MIN_FREQ}.tsv"
out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE)
print(f"wrote {path}; rows identical to v4 {(out.matched_entity_ids == v4.matched_entity_ids).mean():.4f}")
