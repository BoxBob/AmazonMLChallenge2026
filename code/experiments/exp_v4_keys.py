# v4 experiment E2 (pitfall P14): language-independent exact keys.
#  ex_name_hn   = country | full squashed core name | house number
#  ex_hn_street = country | house number | first *distinctive* street word (skips street types / articles in any language)
# Measures (a) validation: extra true pairs vs v3 candidates and candidates added, (b) test: coverage of French silver
# pairs that v3 never generated.
import csv
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import valscore as V

SH = r"D:\AmazonMLChallenge2026\work\v2"
W3 = r"D:\AmazonMLChallenge2026\work\v3"
CAP = 50
STREET_STOP = {"r", "rue", "st", "str", "ave", "blvd", "rd", "dr", "ln", "pl", "ct", "hwy", "ste", "apt", "fl", "bldg",
               "ch", "all", "imp", "rte", "fg", "qu", "res", "bat", "no", "n", "s", "e", "w", "ne", "nw", "se", "sw",
               "de", "du", "des", "la", "le", "les", "l", "d", "of", "the", "and", "sq", "pkwy", "cir", "ter", "ctr",
               "jn", "nr", "opp", "sec", "twp", "unit", "road", "street", "avenue", "lane", "cross", "main", "floor",
               "plot", "flat", "house", "door", "shop", "gali", "marg", "nagar", "chemin", "allee", "impasse",
               "boulevard", "place", "quai", "route", "cours", "lieu", "dit", "po", "box", "suite"}


def hn_street(addr, pc):
    toks = addr.split()
    for i, t in enumerate(toks):
        if t[0].isdigit() and t != pc:
            for u in toks[i + 1:i + 6]:
                if u.isalpha() and len(u) >= 3 and u not in STREET_STOP:
                    return t + " " + u
            return ""
    return ""


def keys(df):
    k1 = np.where((df.squash.str.len() >= 4) & (df.hn != ""), df.country + "|" + df.squash + "|" + df.hn, "")
    hs = [hn_street(a, p) for a, p in zip(df.addr.tolist(), df.pc.tolist())]
    k2 = np.where(np.array(hs, dtype=object) != "", df.country + "|" + pd.Series(hs, index=df.index), "")
    return {"ex_name_hn": k1, "ex_hn_street": k2}


def pairs_for(df, key, qidx):
    pool = df.index[(df.src != 1).to_numpy() & (key != "")]
    pk = pd.Series(key[pool], index=pool)
    sizes = pk.map(pk.value_counts())
    pk = pk[sizes <= CAP]
    qk = pd.Series(key[qidx], index=qidx)
    qk = qk[qk != ""]
    j = pd.DataFrame({"k": qk.values, "q": qk.index}).merge(pd.DataFrame({"k": pk.values, "c": pk.index}), on="k")
    return j.q.to_numpy(np.int64), j.c.to_numpy(np.int64)


cols = ["entity_id", "country", "src", "squash", "hn", "addr", "pc"]
tr = pq.read_table(SH + r"\train_records.parquet", columns=cols).to_pandas()
vz = np.load(W3 + r"\val_pairs.npz")
have = set((vz["q"].astype(np.int64) * V.N + vz["c"].astype(np.int64)).tolist())
gk = set(V.GK.tolist())
tot = len(V.GK)
K = keys(tr)
allnew = set()
for name, key in K.items():
    q, c = pairs_for(tr, key, V.VAL)
    kk = set((q * V.N + c).tolist())
    new = kk - have
    allnew |= new
    print(f"VAL {name:13s}: {len(kk) / len(V.VAL):.2f} pairs/S1, new vs v3 candidates {len(new) / len(V.VAL):.2f}/S1, "
          f"new TRUE pairs {len(new & gk):,} ({len(new & gk) / tot:.2%} of true pairs)")
print(f"VAL both keys: new TRUE pairs {len(allnew & gk):,} ({len(allnew & gk) / tot:.2%}), v3 recall 97.57% -> "
      f"{(len(have & gk) + len(allnew & gk)) / tot:.2%} before prefilter")

te = pq.read_table(SH + r"\test_records.parquet", columns=cols).to_pandas()
Kt = keys(te)
test_q = te.index[te.src == 1].to_numpy()
cand = pd.read_csv(r"D:\AmazonMLChallenge2026\output\candidate_pairs_v3.tsv", sep="\t", dtype=str,
                   keep_default_na=False, quoting=csv.QUOTE_NONE)
idx = pd.Series(te.index, index=te.entity_id)
cq = cand.source1_entity_id.map(idx).to_numpy()
lists = cand.candidate_entity_ids.str.split(",")
lens = lists.str.len().to_numpy()
cc = idx.reindex(np.concatenate(lists.to_numpy())).to_numpy()
have_t = set((np.repeat(cq, lens).astype(np.int64) * len(te) + cc.astype(np.int64)).tolist())
ctry = te.country.to_numpy()
for name, key in Kt.items():
    q, c = pairs_for(te, key, test_q)
    kk = q * len(te) + c
    new = np.array([k not in have_t for k in kk.tolist()])
    s = pd.DataFrame({"country": ctry[q], "new": new})
    print(f"TEST {name:13s}: new pairs per S1 by country:",
          (s.groupby("country").new.sum() / pd.Series(ctry[test_q]).value_counts()).round(3).to_dict())
