import numpy as np, pandas as pd, pyarrow.parquet as pq, xgboost as xgb, csv
from rapidfuzz import process, fuzz
W = r"D:\AmazonMLChallenge2026\work\baseline"; D = r"D:\AmazonMLChallenge2026\dataset\train"
rec = pq.read_table(W + r"\train_records.parquet", columns=["entity_id","core","addr","country","src","pc"]).to_pandas()
ids = pd.Index(rec.entity_id); N = len(rec)
c = pd.read_parquet(W + r"\valpairs_cands.parquet"); X = pd.read_parquet(W + r"\valpairs_features.parquet")
b = xgb.Booster(); b.load_model(W + r"\xgb_baseline.json"); b.set_param({"device": "cuda"})
c["p"] = b.predict(xgb.DMatrix(X.to_numpy(np.float32), feature_names=list(X.columns)))
VQ = np.sort(c.q.unique())  # all val S1 have >=1 cand? check below
gt = pd.read_csv(D + r"\train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
gq = ids.get_indexer(gt.source1_entity_id)
m = gt.matched_entity_ids.str.split(",").explode(); m = m[m != ""]
G = pd.DataFrame({"q": ids.get_indexer(gt.source1_entity_id.loc[m.index]), "c": ids.get_indexer(m.values)})
# reconstruct VAL_Q exactly as notebook
rng = np.random.default_rng(42); s1 = np.flatnonzero(rec.src.to_numpy() == 1); perm = rng.permutation(s1)
VAL = np.sort(perm[300_000:400_000]); print("val S1 with no candidates:", len(np.setdiff1d(VAL, VQ)))
G = G[G.q.isin(VAL)]
T = 0.625
d = c[c.p >= T].sort_values("p", ascending=False).drop_duplicates("c")
d["r"] = d.groupby(["q", c.loc[d.index, "c"].map(lambda x: 0) if False else rec.src.to_numpy()[d.c]]).cumcount()
d = d[d.r < np.where(rec.src.to_numpy()[d.c] == 2, 5, 6)]
def score(pred):
    key = lambda df: df.q.astype(np.int64) * N + df.c
    gk = set(key(G)); pk = key(pred)
    tp = pred[pk.isin(gk)].groupby("q").size()
    df = pd.DataFrame(index=VAL); df["nt"] = G.groupby("q").size(); df["np"] = pred.groupby("q").size(); df["tp"] = tp
    df = df.fillna(0); P = df.tp / df.np.where(df.np > 0, 1); R = df.tp / df.nt.where(df.nt > 0, 1)
    f = np.where(df.tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0); f = np.where((df.nt == 0) & (df.np == 0), 1.0, f)
    df["f"] = f; return df
base = score(d[["q","c"]]); print(f"\nBASE macro F0.5 {base.f.mean():.4f}")
lab = c.label.to_numpy() == 1
nofp = d[d.label == 1][["q","c"]]; print(f"remove all false positives   -> {score(nofp).f.mean():.4f}")
allin = pd.concat([d[["q","c"]], c[lab][["q","c"]]]).drop_duplicates(); print(f"add missed in-candidate TPs  -> {score(allin[allin.set_index(['q','c']).index.isin(c[lab].set_index(['q','c']).index) | True]).f.mean():.4f} (with FPs kept)")
print(f"perfect classifier (ceiling) -> {score(c[lab][['q','c']]).f.mean():.4f}")
print(f"perfect everything           -> 1.0")
base["country"] = rec.country.to_numpy()[base.index]
base["bucket"] = pd.cut(base.nt, [-1, 0, 1, 3, 5, 20], labels=["singleton", "1", "2-3", "4-5", "6+"])
print("\nby country:\n", base.groupby("country").f.agg(["mean", "size"]).round(4))
print("\nby #true matches:\n", base.groupby("bucket", observed=True).f.agg(["mean", "size"]).round(4))
sg = base[base.nt == 0]; print(f"\nsingletons: {len(sg)}, predicted non-empty (score 0): {(sg.np > 0).sum()} ({(sg.np>0).mean():.1%})")
nz = base[base.nt > 0]; print(f"non-singletons with 0 predictions: {(nz.np == 0).sum()} ({(nz.np==0).mean():.1%}); with any FP: {((nz.np > nz.tp)).sum()} ({(nz.np>nz.tp).mean():.1%})")
# loss attribution
loss = (1 - base.f).sum() / len(base); print(f"\ntotal loss {loss:.4f}")
print(f"  from singletons  {(1 - sg.f).sum()/len(base):.4f}")
print(f"  from matched ents {(1 - nz.f).sum()/len(base):.4f}")
# categorize missed true pairs
ck = c.set_index(["q","c"]).index; gi = G.set_index(["q","c"]).index
blockmiss = G[~gi.isin(ck)]; clfmiss = c[lab & (c.p < T)]; fp = d[d.label == 0]
core = rec.core.to_numpy(); addr = rec.addr.to_numpy(); pc = rec.pc.to_numpy()
raw = {}
for s in (1, 2, 3):
    t = pd.read_csv(D + f"/train_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3, usecols=["entity_id", "business_name"])
    raw.update(dict.fromkeys([], None)); raw[s] = t
rawname = pd.concat(raw.values()).set_index("entity_id").business_name.reindex(rec.entity_id).to_numpy()
def cats(df, title):
    q, cc = df.q.to_numpy(), df.c.to_numpy()
    ns = process.cpdist(list(core[q]), list(core[cc]), scorer=fuzz.token_set_ratio, workers=-1)
    as_ = process.cpdist(list(addr[q]), list(addr[cc]), scorer=fuzz.token_set_ratio, workers=-1)
    blank = addr[cc] == ""; nonlat = np.array([not str(x).isascii() for x in rawname[cc]])
    nonlat_q = np.array([not str(x).isascii() for x in rawname[q]])
    print(f"\n{title}: {len(df):,} pairs")
    print(f"  candidate address blank            {blank.mean():6.1%}")
    print(f"  name sim <60 (trade/DBA name)      {(ns < 60).mean():6.1%}")
    print(f"  address sim <60 (non-blank)        {((as_ < 60) & ~blank).mean():6.1%}")
    print(f"  name<60 AND (addr<60 or blank)     {((ns < 60) & ((as_ < 60) | blank)).mean():6.1%}")
    print(f"  native-script name (cand / S1)     {nonlat.mean():6.1%} / {nonlat_q.mean():6.1%}")
    print(f"  postcode equal (both present)      {((pc[q]==pc[cc]) & (pc[q]!='')).mean():6.1%}")
    print(f"  India share                        {(rec.country.to_numpy()[q]=='India').mean():6.1%}")
cats(blockmiss, "TRUE PAIRS MISSED BY BLOCKING")
cats(clfmiss, "TRUE PAIRS IN CANDIDATES BUT BELOW THRESHOLD")
cats(fp, "FALSE POSITIVES (predicted)")
cats(c[lab].sample(50000, random_state=0), "reference: random TRUE pairs")
# how many true pairs lost by one-owner/cap rule
above = c[lab & (c.p >= T)]; kept = d[d.label == 1]
print(f"\ntrue pairs above threshold {len(above):,}, lost by one-owner/cap rule {len(above) - len(kept):,}")
