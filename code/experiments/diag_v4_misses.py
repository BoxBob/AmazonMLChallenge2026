# What do v4's missed true pairs look like? (a) candidates the classifier rejected, (b) pairs blocking never produced.
import csv, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from rapidfuzz import fuzz, process
import valscore as V
SH = r"D:\AmazonMLChallenge2026\work\v2"; W4 = r"D:\AmazonMLChallenge2026\work\v4"; D = r"D:\AmazonMLChallenge2026\dataset\train"
tb = pq.read_table(SH + r"\train_records.parquet", columns=["entity_id", "core", "addr", "hn", "native", "country"]).combine_chunks()
take = lambda col, i: np.array(tb[col].take(pa.array(i)).to_pylist(), dtype=object)
vz = np.load(W4 + r"\val_pairs.npz"); q, c, y = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
pb = np.load(W4 + r"\val_p_v4.npy")
mq, mc = V.threshold_rule(q, c, pb, 0.70)
pred = np.isin(q * V.N + c, mq * V.N + mc)
fn = (y == 1) & ~pred
print(f"classifier misses {fn.sum():,}; probability quantiles {np.quantile(pb[fn], [.1, .25, .5, .75, .9]).round(3)}")
blk = np.setdiff1d(V.GK, q[y == 1] * V.N + c[y == 1]); bq, bc = blk // V.N, blk % V.N
raw = {s: pd.read_csv(f"{D}/train_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE).set_index("entity_id") for s in (1, 2, 3)}
eid = tb["entity_id"].to_numpy(zero_copy_only=False)
def rawrec(i):
    e = eid[i]; r = raw[int(e[1])].loc[e]; return f"{r.business_name[:30]:30} | {r.business_address[:52]}"
def profile(qq, cc, name):
    ns = process.cpdist(list(take("core", qq)), list(take("core", cc)), scorer=fuzz.token_set_ratio, workers=-1)
    a_q, a_c = take("addr", qq), take("addr", cc)
    as_ = process.cpdist(list(a_q), list(a_c), scorer=fuzz.token_set_ratio, workers=-1)
    blank = a_c == ""
    print(f"\n{name}: {len(qq):,} pairs | cand addr blank {blank.mean():.1%} | name sim<60 {(ns < 60).mean():.1%} | "
          f"name>=90 {(ns >= 90).mean():.1%} | addr>=90 (non-blank) {((as_ >= 90) & ~blank).mean():.1%} | "
          f"native cand {take('native', cc).astype(bool).mean():.1%} | India {(take('country', qq) == 'India').mean():.1%}")
    for j in np.random.default_rng(1).choice(len(qq), 10, replace=False):
        print("   ", rawrec(qq[j])); print("      ", rawrec(cc[j]))
profile(q[fn], c[fn], "CLASSIFIER MISSES (in candidates, rejected)")
profile(bq, bc, "BLOCKING MISSES (never candidates)")
