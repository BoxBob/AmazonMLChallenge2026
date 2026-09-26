# Silver-label check of the France recall hypothesis.
# Silver positive = a pool record with the same (country, squashed name, house number) as the S1 entity.
# On validation (labels known) we measure silver precision and the share of silver pairs v3 accepts; on test we compare
# that acceptance share across countries. A much lower share for France means France is under-matched.
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

import valscore as V

SH = r"D:\AmazonMLChallenge2026\work\v2"
W3 = r"D:\AmazonMLChallenge2026\work\v3"
W4 = r"D:\AmazonMLChallenge2026\work\v4"


def silver_pairs(df):
    s = df[(df.squash.str.len() >= 4) & (df.hn != "")]
    k = s.country + "|" + s.squash + "|" + s.hn
    s1 = pd.DataFrame({"k": k[s.src == 1], "q": s.index[s.src == 1]})
    pool = pd.DataFrame({"k": k[s.src != 1], "c": s.index[s.src != 1]})
    return s1.merge(pool, on="k")[["q", "c"]]


# ---------- validation: silver precision and v3 acceptance
tr = pq.read_table(SH + r"\train_records.parquet", columns=["country", "src", "squash", "hn"]).to_pandas()
sp = silver_pairs(tr)
sp = sp[sp.q.isin(V.VAL)]
key = sp.q.to_numpy(np.int64) * V.N + sp.c.to_numpy(np.int64)
pos = np.minimum(np.searchsorted(V.GK, key), len(V.GK) - 1)
sp["true"] = V.GK[pos] == key
vz = np.load(W3 + r"\val_pairs.npz")
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
pr = np.load(W4 + r"\e1_val_probs.npz")
band, pce, p2 = pr["band"], pr["pce"], pr["vp2_v3"]
inb = (p2[band] >= 0.01) & (p2[band] <= 0.99)
b_ = band[inb]
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
X = np.column_stack([lg(p2[b_]), lg(pce[inb])])
pb = p2.copy()
pb[b_] = LogisticRegression(C=10.0).fit(X[np.isin(vq[b_], V.A)], vy[b_][np.isin(vq[b_], V.A)]).predict_proba(X)[:, 1]
mq, mc = V.threshold_rule(vq, vc, pb, 0.725)
pred = set((mq * V.N + mc).tolist())
sp["accepted"] = [k in pred for k in key.tolist()]
sp["country"] = tr.country.to_numpy()[sp.q.to_numpy()]
print("VALIDATION silver pairs (same name + same house number):")
print(sp.groupby("country").agg(pairs=("true", "size"), silver_precision=("true", "mean"),
                                v3_accepts=("accepted", "mean")).round(4).to_string())
print("  accepted share among TRUE silver pairs:", sp[sp.true].groupby("country").accepted.mean().round(4).to_dict())

# ---------- test: v3 acceptance of silver pairs per country
te = pq.read_table(SH + r"\test_records.parquet", columns=["entity_id", "country", "src", "squash", "hn"]).to_pandas()
st = silver_pairs(te)
ids = te.entity_id.to_numpy()
res = pd.read_csv(r"D:\AmazonMLChallenge2026\output\matching_results_v3.tsv", sep="\t", dtype=str,
                  keep_default_na=False, quoting=3)
m = res.assign(c=res.matched_entity_ids.str.split(",")).explode("c")
m = m[m.c != ""]
predk = set((m.source1_entity_id + "|" + m.c).tolist())
st["accepted"] = [f"{a}|{b}" in predk for a, b in zip(ids[st.q.to_numpy()], ids[st.c.to_numpy()])]
st["country"] = te.country.to_numpy()[st.q.to_numpy()]
n_s1 = te[te.src == 1].groupby("country").size()
g = st.groupby("country").agg(pairs=("accepted", "size"), v3_accepts=("accepted", "mean"))
g["silver_per_s1"] = g.pairs / n_s1
print("\nTEST silver pairs:")
print(g.round(4).to_string())
