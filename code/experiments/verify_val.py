# Independent check of the v2 validation score: rebuild predictions from saved artifacts,
# write them as entity-id lists, and score with a from-scratch implementation of the README metric.
import csv, numpy as np, pandas as pd, pyarrow.parquet as pq, xgboost as xgb, re, ast
W = r"D:\AmazonMLChallenge2026\work\v2"; D = r"D:\AmazonMLChallenge2026\dataset\train"
src = open("improved_src.py", encoding="utf-8").read()
ns = {"np": np, "pd": pd}
for fn in ("group_features",):   # reuse the exact notebook function source, nothing else
    m = re.search(rf"^def {fn}\(.*?(?=^\S)", src, re.S | re.M); exec(m.group(0), ns)
ns["G_FEATURES"] = ast.literal_eval(re.search(r"^G_FEATURES = (\[.*?\])", src, re.S | re.M).group(1))
rec = pq.read_table(W + r"\train_records.parquet", columns=["entity_id", "src"]).to_pandas()
eid, srcs = rec.entity_id.to_numpy(), rec.src.to_numpy()
z = np.load(W + r"\val_pairs.npz"); q, c, X = z["q"], z["c"], z["X"]
tr = np.load(W + r"\train_pairs.npz")["q"]
print("val entities also in train pairs:", len(np.intersect1d(np.unique(q), np.unique(tr))))
m1 = xgb.Booster(); m1.load_model(W + r"\m1.json"); m1.set_param({"device": "cpu", "nthread": 4})
m2 = xgb.Booster(); m2.load_model(W + r"\m2.json"); m2.set_param({"device": "cpu", "nthread": 4})
p1 = m1.predict(xgb.DMatrix(X, feature_names=m1.feature_names), iteration_range=(0, m1.best_iteration + 1))
G = ns["group_features"](q, srcs[c], p1)
p2 = m2.predict(xgb.DMatrix(np.hstack([X, G]), feature_names=m2.feature_names), iteration_range=(0, m2.best_iteration + 1))
df = pd.DataFrame({"q": q, "c": c, "s": srcs[c], "p": p2})
df = df[df.p >= 0.7].sort_values("p", ascending=False).drop_duplicates("c")
df["r"] = df.groupby(["q", "s"]).cumcount(); df = df[df.r < np.where(df.s == 2, 5, 6)]
val_ids = eid[np.unique(q)]
pred = df.assign(a=eid[df.q], b=eid[df.c]).groupby("a").b.agg(",".join)
out = pd.DataFrame({"source1_entity_id": val_ids, "matched_entity_ids": pd.Series(val_ids).map(pred).fillna("").to_numpy()})
out.to_csv(W + r"\val_predictions_v2.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
# --- independent scorer: plain python over the TSV files
truth = {}
with open(D + r"\train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        a, _, b = line.rstrip("\n").partition("\t"); truth[a] = set(x for x in b.split(",") if x)
scores = []
with open(W + r"\val_predictions_v2.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        a, _, b = line.rstrip("\n").partition("\t"); P = set(x for x in b.split(",") if x); T = truth[a]
        if not T and not P: scores.append(1.0); continue
        tp = len(P & T)
        if tp == 0: scores.append(0.0); continue
        pr, rc = tp / len(P), tp / len(T); scores.append(1.25 * pr * rc / (0.25 * pr + rc))
print(f"independent macro F0.5 on {len(scores):,} validation entities: {np.mean(scores):.4f}")
