# Second France suspect: same-name pairs where the candidate has NO address. With a blank address identity rests on the
# name alone; if k S1 entities of the country share the name, the pair's precision is bounded by ~1/k.
import csv, numpy as np, pandas as pd, pyarrow.parquet as pq
import valscore as V
SH = r"D:\AmazonMLChallenge2026\work\v2"; W4 = r"D:\AmazonMLChallenge2026\work\v4"; OUT = r"D:\AmazonMLChallenge2026\output"
def s1_name_counts(df):
    s1 = df[df.src == 1]
    return (df.country + "|" + df.squash).map((s1.country + "|" + s1.squash).value_counts()).fillna(0).to_numpy()
tr = pq.read_table(SH + r"\train_records.parquet", columns=["country", "src", "squash", "addr"]).to_pandas()
k_tr = s1_name_counts(tr)
vz = np.load(W4 + r"\val_pairs.npz"); q, c, y = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
pb = np.load(W4 + r"\val_p_v4.npy")
E = (tr.squash.to_numpy()[q] == tr.squash.to_numpy()[c]) & (tr.addr.to_numpy()[c] == "")
acc = pb >= 0.70
kb = pd.cut(k_tr[q], [0, 1, 2, 4, 10, 1e9], labels=["1", "2", "3-4", "5-10", ">10"])
d = pd.DataFrame({"k": kb, "true": y == 1, "acc": acc})[E]
print("VALIDATION same-name blank-address pairs by #S1 entities sharing the name (k):")
print(d.groupby("k", observed=True).agg(pairs=("true", "size"), true_rate=("true", "mean"), accepted=("acc", "mean"),
      precision_of_accepted=("true", lambda s: s[d.loc[s.index, "acc"]].mean())).round(3).to_string())
te = pq.read_table(SH + r"\test_records.parquet", columns=["entity_id", "country", "src", "squash", "addr"]).to_pandas()
k_te = s1_name_counts(te); te["k"] = k_te; te = te.set_index("entity_id")
res = pd.read_csv(OUT + r"\matching_results_v4.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
m = res.assign(c=res.matched_entity_ids.str.split(",")).explode("c"); m = m[m.c != ""]
A, B = te.loc[m.source1_entity_id.to_numpy()], te.loc[m.c.to_numpy()]
isE = (A.squash.to_numpy() == B.squash.to_numpy()) & (B.addr.to_numpy() == "")
t = pd.DataFrame({"country": A.country.to_numpy(), "k": pd.cut(A.k.to_numpy(), [0, 1, 2, 4, 10, 1e9], labels=["1", "2", "3-4", "5-10", ">10"]), "E": isE})
print("\nTEST v4 ACCEPTED same-name blank-address pairs, by country and k:")
print(t[t.E].groupby(["k", "country"], observed=True).size().unstack(fill_value=0).to_string())
