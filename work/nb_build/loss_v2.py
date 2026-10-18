import numpy as np, pandas as pd, pyarrow.parquet as pq, xgboost as xgb, re, ast
from rapidfuzz import process, fuzz
W = r"D:\AmazonMLChallenge2026\work\v2"; D = r"D:\AmazonMLChallenge2026\dataset\train"
src = open("improved_src.py", encoding="utf-8").read(); ns = {"np": np, "pd": pd}
exec(re.search(r"^def group_features\(.*?(?=^\S)", src, re.S | re.M).group(0), ns)
ns["G_FEATURES"] = ast.literal_eval(re.search(r"^G_FEATURES = (\[.*?\])", src, re.S | re.M).group(1))
rec = pq.read_table(W + r"\train_records.parquet", columns=["entity_id","src","country","core","addr","native"]).to_pandas()
N = len(rec); srcs = rec.src.to_numpy(); core = rec.core.to_numpy(); addr = rec.addr.to_numpy()
z = np.load(W + r"\val_pairs.npz"); q, c, X, y = z["q"].astype(np.int64), z["c"].astype(np.int64), z["X"], z["y"]
m1 = xgb.Booster(); m1.load_model(W + r"\m1.json"); m1.set_param({"device": "cpu", "nthread": 8})
m2 = xgb.Booster(); m2.load_model(W + r"\m2.json"); m2.set_param({"device": "cpu", "nthread": 8})
p1 = m1.predict(xgb.DMatrix(X, feature_names=m1.feature_names), iteration_range=(0, m1.best_iteration + 1))
p = m2.predict(xgb.DMatrix(np.hstack([X, ns["group_features"](q, srcs[c], p1)]), feature_names=m2.feature_names), iteration_range=(0, m2.best_iteration + 1))
ids = pd.Index(rec.entity_id)
gt = pd.read_csv(D + "/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
m = gt.matched_entity_ids.str.split(",").explode(); m = m[m != ""]
GQ = ids.get_indexer(gt.source1_entity_id.loc[m.index]); GC = ids.get_indexer(m.values)
VAL = np.unique(q); sel = np.isin(GQ, VAL); GQ, GC = GQ[sel], GC[sel]
def resolve(q, c, p, t=0.7):
    d = pd.DataFrame({"q": q, "c": c, "s": srcs[c], "p": p}); d = d[d.p >= t].sort_values("p", ascending=False).drop_duplicates("c")
    d["r"] = d.groupby(["q","s"]).cumcount(); d = d[d.r < np.where(d.s == 2, 5, 6)]; return d.q.to_numpy(), d.c.to_numpy()
GK = set((GQ.astype(np.int64) * N + GC).tolist())
def ent(pq_, pc_):
    k = pq_.astype(np.int64) * N + pc_; hit = np.array([x in GK for x in k.tolist()])
    nt = np.bincount(np.searchsorted(VAL, GQ), minlength=len(VAL)); npd = np.bincount(np.searchsorted(VAL, pq_), minlength=len(VAL))
    tp = np.bincount(np.searchsorted(VAL, pq_[hit]), minlength=len(VAL))
    with np.errstate(all="ignore"):
        P = np.where(npd > 0, tp / np.maximum(npd, 1), 0); R = np.where(nt > 0, tp / np.maximum(nt, 1), 0)
        f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    return np.where((nt == 0) & (npd == 0), 1.0, f), nt, npd, tp
pq_, pc_ = resolve(q, c, p); f, nt, npd, tp = ent(pq_, pc_)
print(f"v2 macro F0.5 {f.mean():.4f}   total loss {1 - f.mean():.4f}")
lab = y == 1
k_pred = set((pq_.astype(np.int64) * N + pc_).tolist())
f_nofp = ent(*[np.array(a) for a in zip(*[(a, b) for a, b in zip(pq_, pc_) if a * N + b in GK])] if len(pq_) else (pq_, pc_))[0].mean()
allq = np.concatenate([pq_, q[lab]]); allc = np.concatenate([pc_, c[lab]]); u = np.unique(allq.astype(np.int64) * N + allc)
f_nofn = ent(u // N, u % N)[0].mean()
f_oracle = ent(q[lab], c[lab])[0].mean()
print(f"  remove all false positives        -> {f_nofp:.4f}  (FP cost {f_nofp - f.mean():.4f})")
print(f"  add all missed in-candidate TPs   -> {f_nofn:.4f}  (classifier-FN cost {f_nofn - f.mean():.4f})")
print(f"  perfect classifier on candidates  -> {f_oracle:.4f}  (blocking cost {1 - f_oracle:.4f})")
ctry = rec.country.to_numpy()[VAL]
bucket = pd.cut(nt, [-1, 0, 1, 3, 5, 20], labels=["singleton", "1", "2-3", "4-5", "6+"])
df = pd.DataFrame({"f": f, "ctry": ctry, "b": bucket})
print("\nloss share by country:\n", ((1 - df.f).groupby(df.ctry).sum() / (1 - df.f).sum()).round(3).to_string(), "\nmean F:\n", df.groupby("ctry").f.mean().round(4).to_string())
print("\nloss share by #true:\n", ((1 - df.f).groupby(df.b, observed=True).sum() / (1 - df.f).sum()).round(3).to_string())
print("\nmean F by #true:\n", df.groupby("b", observed=True).f.mean().round(4).to_string())
sg = nt == 0; print(f"\nsingletons wrongly matched: {(npd[sg] > 0).sum()} / {sg.sum()}")
print(f"entities with matches but empty prediction: {((nt > 0) & (npd == 0)).sum()}")
print(f"entities with >=1 FP: {(npd > tp).sum()}, with >=1 missed: {(tp < nt).sum()}")
# error categories
fp = np.array([(a, b) for a, b in zip(pq_, pc_) if a * N + b not in GK])
kc = set((q.astype(np.int64) * N + c).tolist())
blk = np.array([(a, b) for a, b in zip(GQ, GC) if a * N + b not in kc])
fn = np.array([(a, b) for a, b in zip(q[lab], c[lab]) if a * N + b not in k_pred])
nat = rec.native.to_numpy()
def cats(arr, t):
    a, b = arr[:, 0], arr[:, 1]
    ns_ = process.cpdist(list(core[a]), list(core[b]), scorer=fuzz.token_set_ratio, workers=-1)
    as_ = process.cpdist(list(addr[a]), list(addr[b]), scorer=fuzz.token_set_ratio, workers=-1); bl = addr[b] == ""
    print(f"\n{t}: {len(arr):,}")
    print(f"  cand addr blank {bl.mean():.1%} | name sim<60 {(ns_<60).mean():.1%} | name sim>=90 {(ns_>=90).mean():.1%} | addr sim>=90 {((as_>=90)&~bl).mean():.1%} | native cand {nat[b].mean():.1%} | India {(rec.country.to_numpy()[a]=='India').mean():.1%}")
    rng = np.random.default_rng(0)
    for j in rng.choice(len(arr), 6, replace=False):
        print(f"    {core[a[j]][:32]!r:34} {addr[a[j]][:45]!r:47} <-> {core[b[j]][:32]!r:34} {addr[b[j]][:45]!r}")
cats(fp, "FALSE POSITIVES"); cats(fn, "IN-CANDIDATE MISSES"); cats(blk, "BLOCKING MISSES")
