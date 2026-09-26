# v4 experiment E1 (pitfall P11): S1-side ambiguity features.
# First principles: each S2/S3 record belongs to at most one S1 entity. If a name is unique among the S1 entities of a
# country, a blank-address record carrying that name can only belong to that entity; if many S1 entities share the name,
# the name alone is weak evidence. Count ambiguity on the *S1 side* over the full S1 table (same on train and test).
import os
import re
import sys
import time
import ast
import gc

os.environ["HF_HOME"] = r"D:\AmazonMLChallenge2026\work\hf"
os.environ["JOBLIB_TEMP_FOLDER"] = r"D:\AmazonMLChallenge2026\work\tmp"
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pac  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
import xgboost as xgb  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import valscore as V  # noqa: E402

SHARED = r"D:\AmazonMLChallenge2026\work\v2"
W3 = r"D:\AmazonMLChallenge2026\work\v3"
W4 = r"D:\AmazonMLChallenge2026\work\v4"
os.makedirs(W4, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


tbl = pq.read_table(SHARED + r"\train_records.parquet",
                    columns=["country", "src", "squash", "core", "addr", "hn", "name"]).combine_chunks()


class Rec:
    src = tbl["src"].to_numpy()
    n = tbl.num_rows

    @staticmethod
    def take(col, idx):
        return tbl[col].take(pa.array(idx)).to_pylist()


# ---- the v3 group / cross-source functions, taken verbatim from the v3 source
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3_src.py"), encoding="utf-8").read()
ns = {"np": np, "pd": pd, "process": process, "fuzz": fuzz}
for fn in ("group_features", "cross_source_feats", "group_ctx"):
    exec(re.search(rf"^def {fn}\(.*?(?=^\S)", src, re.S | re.M).group(0), ns)
ns["G_FEATURES"] = ast.literal_eval(re.search(r"^G_FEATURES = (\[.*?\])", src, re.S | re.M).group(1))


# ---- S1-side ambiguity counts
def s1_counts(key, blank):
    codes = pac.dictionary_encode(key).combine_chunks().indices.to_numpy()
    s1 = Rec.src == 1
    cnt_s1 = np.bincount(codes[s1], minlength=codes.max() + 1)
    cnt_pool = np.bincount(codes[~s1], minlength=codes.max() + 1)
    return (np.where(blank, 0, cnt_s1[codes]).astype(np.float32),
            np.where(blank, 0, cnt_pool[codes]).astype(np.float32))


name_blank = pac.less(pac.utf8_length(tbl["squash"]), 3).to_numpy(zero_copy_only=False)
addr_blank = pac.equal(pac.utf8_length(tbl["addr"]), 0).to_numpy(zero_copy_only=False)
S1_NAME, POOL_NAME = s1_counts(pac.binary_join_element_wise(tbl["country"], tbl["squash"], "|"), name_blank)
S1_ADDR, POOL_ADDR = s1_counts(pac.binary_join_element_wise(tbl["country"], tbl["addr"], "|"), addr_blank)
AMB_NAMES = ["s1_name_cnt_q", "s1_name_cnt_c", "pool_name_cnt_q", "s1_addr_cnt_q", "s1_addr_cnt_c",
             "pool_addr_cnt_q", "name_unique_pair"]
log(f"S1 name counts: share of S1 entities with a unique name {np.mean(S1_NAME[Rec.src == 1] == 1):.1%}")


def amb_feats(q, c):
    f = np.column_stack([S1_NAME[q], S1_NAME[c], POOL_NAME[q], S1_ADDR[q], S1_ADDR[c], POOL_ADDR[q]])
    f = np.log1p(f)
    uniq = ((S1_NAME[q] == 1) & (S1_NAME[c] <= 1)).astype(np.float32)   # the S1 name is unique and c's name is not another S1's
    return np.column_stack([f, uniq]).astype(np.float32)


# ---- data: the v3 pairs
tz, vz = np.load(W3 + r"\train_pairs.npz"), np.load(W3 + r"\val_pairs.npz")
tq, tc, ty = tz["q"].astype(np.int64), tz["c"].astype(np.int64), tz["y"]
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
m1v3 = xgb.Booster(model_file=W3 + r"\m1.json")
F1_V3 = m1v3.feature_names
F1 = F1_V3 + AMB_NAMES
tX = np.hstack([tz["X"], amb_feats(tq, tc)])
vX = np.hstack([vz["X"], amb_feats(vq, vc)])
del tz
gc.collect()
log(f"train {tX.shape}, val {vX.shape}")

PAR = dict(objective="binary:logistic", eval_metric=["logloss", "aucpr"], device="cuda", tree_method="hist",
           max_depth=9, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, max_bin=256,
           seed=42)


def fit(X, y, Xv, yv, names):
    d = xgb.QuantileDMatrix(X, y, feature_names=names)
    dv = xgb.QuantileDMatrix(Xv, yv, ref=d, feature_names=names)
    b = xgb.train(PAR, d, 3000, evals=[(dv, "val")], early_stopping_rounds=60, verbose_eval=False)
    del d, dv
    gc.collect()
    return b


def pred(b, X, names):
    return b.predict(xgb.DMatrix(X, feature_names=names), iteration_range=(0, b.best_iteration + 1))


fold = np.random.default_rng(44).integers(0, 2, size=Rec.n)[tq].astype(np.int8)
oof = np.zeros(len(tq), np.float32)
for f in (0, 1):
    mk = fold != f
    b = fit(tX[mk], ty[mk], vX, vy, F1)
    oof[~mk] = pred(b, tX[~mk], F1)
    log(f"fold {f} best iteration {b.best_iteration}")
    del b
m1 = fit(tX, ty, vX, vy, F1)
vp1 = pred(m1, vX, F1)
m1.save_model(W4 + r"\e1_m1.json")
log(f"m1 best iteration {m1.best_iteration}")
print(pd.Series(m1.get_score(importance_type="gain")).reindex(AMB_NAMES).round(1).to_string())
F2 = F1 + ns["G_FEATURES"] + ["xs_name", "xs_addr", "xs_hn_eq", "ss_name", "ss_addr", "ss_hn_eq"]
vG = np.hstack([vX, ns["group_ctx"](Rec, vq, vc, vp1)])
m2 = fit(np.hstack([tX, ns["group_ctx"](Rec, tq, tc, oof)]), ty, vG, vy, F2)
vp2 = pred(m2, vG, F2)
m2.save_model(W4 + r"\e1_m2.json")
log(f"m2 best iteration {m2.best_iteration}")
del tX
gc.collect()

# ---- v3 reference probabilities on the same pairs (v3 m1/m2)
m2v3 = xgb.Booster(model_file=W3 + r"\m2.json")
vp1_v3 = pred(m1v3, vz["X"], F1_V3)
vp2_v3 = pred(m2v3, np.hstack([vz["X"], ns["group_ctx"](Rec, vq, vc, vp1_v3)]), m2v3.feature_names)

# ---- cross-encoder on the union of both uncertain bands, then blend (fit on A, report B)
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(W3 + r"\ce_model")
ce = AutoModelForSequenceClassification.from_pretrained(W3 + r"\ce_model").cuda().eval()


def ser(idx):
    out = []
    for n, a, h in zip(Rec.take("name", idx), Rec.take("addr", idx), Rec.take("hn", idx)):
        if h:
            a = (" " + a + " ").replace(f" {h} ", f" [NUM] {h} [/NUM] ", 1).strip()
        out.append(f"[COL] name [VAL] {n} [COL] addr [VAL] {a if a else 'none'}")
    return out


band = np.flatnonzero(((vp2 >= 0.01) & (vp2 <= 0.99)) | ((vp2_v3 >= 0.01) & (vp2_v3 <= 0.99)))
pce = np.zeros(len(band), np.float32)
with torch.no_grad():
    for a in range(0, len(band), 256):
        b_ = band[a:a + 256]
        enc = tok(ser(vq[b_]), ser(vc[b_]), truncation=True, max_length=96, padding=True, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.float16):
            pce[a:a + 256] = torch.sigmoid(ce(**{k: v.cuda() for k, v in enc.items()}).logits.squeeze(-1).float()).cpu().numpy()
np.savez(W4 + r"\e1_val_probs.npz", band=band, pce=pce, vp2=vp2, vp2_v3=vp2_v3)
log(f"cross-encoder scored {len(band):,} band pairs")


def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def blended(p2):
    inb = (p2[band] >= 0.01) & (p2[band] <= 0.99)
    b_ = band[inb]
    X = np.column_stack([lg(p2[b_]), lg(pce[inb])])
    inA = np.isin(vq[b_], V.A)
    lr = LogisticRegression(C=10.0).fit(X[inA], vy[b_][inA])
    out = p2.copy()
    out[b_] = lr.predict_proba(X)[:, 1]
    return out


for name, p in (("v3 group model", vp2_v3), ("E1 group model", vp2), ("v3 blend", blended(vp2_v3)),
                ("E1 blend", blended(vp2))):
    t, fa, fb, fall = V.ab_score(p, qq=vq, cc=vc)
    print(f"{name:16s} t={t:.3f}  A={fa:.4f}  B={fb:.4f}  all={fall:.4f}", flush=True)
