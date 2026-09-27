# v11 (NEXT_PLAN phase 2, "B-lite"): collective evidence from siblings (FAMER / TransClean / ComEM "select").
# T30: 97.9% of blank-address true matches have an ADDRESSED sibling in the same cluster, but every pair is decided alone.
# Here each undecided pair (q, c) also looks at the pairs of q that are already confidently accepted ("anchors"):
# does c agree with an anchor (name / address / house number), is c a second record from the same source as an anchor,
# how strongly does another S1 claim c. A small XGBoost stacker (cross-fitted on validation halves A<->B, so every
# prediction is out-of-fold) re-scores the undecided zone; a control stacker WITHOUT the sibling features isolates the
# collective gain from mere re-stacking (T27 showed re-stacking alone is neutral).
import ast
import os
import re
import time

os.environ["HF_HOME"] = r"D:\AmazonMLChallenge2026\work\hf"
os.environ["TEMP"] = os.environ["TMP"] = r"D:\AmazonMLChallenge2026\work\tmp"
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import xgboost as xgb
from rapidfuzz import fuzz, process
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import valscore as V

SH = r"D:\AmazonMLChallenge2026\work\v2"
W4 = r"D:\AmazonMLChallenge2026\work\v4"
W9 = r"D:\AmazonMLChallenge2026\work\v9"
W10 = r"D:\AmazonMLChallenge2026\work\v10m80"
W11 = r"D:\AmazonMLChallenge2026\work\v11"
ZONE, ANCHOR = (0.01, 0.99), 0.8
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


tb = pq.read_table(SH + r"\train_records.parquet", columns=["name", "addr", "hn", "src", "core"]).combine_chunks()


class Rec:
    src = tb["src"].to_numpy()
    n = tb.num_rows

    @staticmethod
    def take(col, idx):
        return tb[col].take(pa.array(idx)).to_pylist()


m1 = xgb.Booster(model_file=W4 + r"\m1.json")
m2 = xgb.Booster(model_file=W4 + r"\m2.json")
pr = lambda b, X: b.predict(xgb.DMatrix(X, feature_names=b.feature_names), iteration_range=(0, b.best_iteration + 1))
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))

# ---------------- 1. v10 validation probabilities: group model p2, v9 cross-encoder on the band, blend (as in v10)
vz = np.load(W10 + r"\val_pairs.npz")
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
FEAT = m1.feature_names
Xv = vz["X"]
p2_path = W11 + r"\val_p2_v10.npy"
if os.path.exists(p2_path):
    p2 = np.load(p2_path)
else:
    src = open("v10m_src.py", encoding="utf-8").read()
    ns = {"np": np, "pd": pd, "process": process, "fuzz": fuzz}
    for fn in ("group_features", "cross_source_feats", "group_ctx"):
        exec(re.search(rf"^def {fn}\(.*?(?=^\S)", src, re.S | re.M).group(0), ns)
    ns["G_FEATURES"] = ast.literal_eval(re.search(r"^G_FEATURES = (\[.*?\])", src, re.S | re.M).group(1))
    p1 = pr(m1, Xv)
    p2 = pr(m2, np.hstack([Xv, ns["group_ctx"](Rec, vq, vc, p1)]))
    np.save(p2_path, p2)
    log("group-model probabilities computed")
band = np.flatnonzero((p2 >= 0.01) & (p2 <= 0.99))
pce_path = W11 + r"\val_pce_v9_narrow.npy"
if os.path.exists(pce_path):
    pce = np.load(pce_path)
else:
    tok = AutoTokenizer.from_pretrained(W9 + r"\mce_model")
    ce = AutoModelForSequenceClassification.from_pretrained(W9 + r"\mce_model").cuda().eval()

    def ser(idx):
        out = []
        for n, a, h in zip(Rec.take("name", idx), Rec.take("addr", idx), Rec.take("hn", idx)):
            if h:
                a = (" " + a + " ").replace(f" {h} ", f" [NUM] {h} [/NUM] ", 1).strip()
            out.append(f"[COL] name [VAL] {n} [COL] addr [VAL] {a if a else 'none'}")
        return out

    pce = np.zeros(len(band), np.float32)
    with torch.no_grad():
        for a in range(0, len(band), 128):
            enc = tok(ser(vq[band[a:a + 128]]), ser(vc[band[a:a + 128]]), truncation=True, max_length=96,
                      padding=True, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.float16):
                pce[a:a + 128] = torch.sigmoid(ce(**{k: v.cuda() for k, v in enc.items()}).logits.squeeze(-1).float()).cpu().numpy()
    del ce
    torch.cuda.empty_cache()
    np.save(pce_path, pce)
    log(f"v9 cross-encoder scored {len(band):,} band pairs")
Xb = np.column_stack([lg(p2[band]), lg(pce)])
inA_b = np.isin(vq[band], V.A)
pb = p2.copy()
pb[band] = LogisticRegression(C=10.0).fit(Xb[inA_b], vy[band][inA_b]).predict_proba(Xb)[:, 1]
t, fa, fb, fall = V.ab_score(pb, qq=vq, cc=vc)
log(f"{'v10 blend (reference)':32s}: t={t:.3f} A={fa:.4f} B={fb:.4f} all={fall:.4f}")

# ---------------- 2. collective features for the undecided zone
zone = np.flatnonzero((pb >= ZONE[0]) & (pb <= ZONE[1]))
anc = np.flatnonzero(pb >= ANCHOR)
log(f"zone pairs {len(zone):,} ({vy[zone].mean():.1%} true); anchor pairs {len(anc):,}")
src_c = Rec.src[vc]
# competition for c: best probability any OTHER S1 gives c
o = np.lexsort((-pb, vc))
cs, ps = vc[o], pb[o]
first = np.r_[True, cs[1:] != cs[:-1]]
best_of_c = pd.Series(ps[first], index=cs[first])
second_of_c = pd.Series(ps[np.r_[False, (~first[1:]) & first[:-1]]], index=cs[np.r_[False, (~first[1:]) & first[:-1]]])
b1 = best_of_c.reindex(vc[zone]).to_numpy()
b2 = second_of_c.reindex(vc[zone]).fillna(0).to_numpy()
rival = np.where(np.isclose(b1, pb[zone]), b2, b1)                   # strongest other claim on c

# zone x anchors of the same q (excluding c itself)
Z = pd.DataFrame({"z": zone, "q": vq[zone]})
A_ = pd.DataFrame({"a": anc, "q": vq[anc]})
J = Z.merge(A_, on="q")
J = J[vc[J.z.to_numpy()] != vc[J.a.to_numpy()]]
log(f"zone x anchor sibling pairs {len(J):,}")
zc, ac = vc[J.z.to_numpy()], vc[J.a.to_numpy()]
core_z, core_a = Rec.take("core", zc), Rec.take("core", ac)
addr_z, addr_a = Rec.take("addr", zc), Rec.take("addr", ac)
hn_z, hn_a = np.array(Rec.take("hn", zc), dtype=object), np.array(Rec.take("hn", ac), dtype=object)
J["s_name"] = process.cpdist(core_z, core_a, scorer=fuzz.token_set_ratio, workers=-1)
J["s_name_r"] = process.cpdist(core_z, core_a, scorer=fuzz.ratio, workers=-1)
az = np.array([bool(x) for x in addr_z])
aa = np.array([bool(x) for x in addr_a])
J["s_addr"] = np.where(az & aa, process.cpdist(addr_z, addr_a, scorer=fuzz.token_set_ratio, workers=-1), -1)
J["same_hn"] = np.where((hn_z != "") & (hn_a != ""), (hn_z == hn_a).astype(np.int8), -1)
J["same_src"] = (Rec.src[zc] == Rec.src[ac]).astype(np.int8)
J["p_anchor"] = pb[J.a.to_numpy()]
J["agree"] = ((J.s_name >= 85) & (J.s_addr != 0) & (J.s_addr != -1) | (J.s_name >= 95)).astype(np.int8)
g = J.groupby("z")
F = pd.DataFrame(index=zone)
F["n_anchor"] = pd.Series(vq[anc]).value_counts().reindex(vq[zone]).fillna(0).to_numpy()
for col, how in (("s_name", "max"), ("s_name_r", "max"), ("s_addr", "max"), ("same_hn", "max"), ("agree", "sum"),
                 ("p_anchor", "max"), ("same_src", "sum")):
    F[f"sib_{col}_{how}"] = g[col].agg(how).reindex(zone).fillna(-2).to_numpy()
# the sibling with the most similar name: its address / house-number agreement (the blank-address case)
best = J.sort_values(["z", "s_name"], ascending=[True, False]).drop_duplicates("z").set_index("z")
F["bestsib_addr"] = best.s_addr.reindex(zone).fillna(-2).to_numpy()
F["bestsib_hn"] = best.same_hn.reindex(zone).fillna(-2).to_numpy()
F["rival"] = rival
F["c_blank_addr"] = (~np.array([bool(x) for x in Rec.take("addr", vc[zone])])).astype(np.int8)
F["same_src_anchors"] = pd.Series(1, index=pd.MultiIndex.from_arrays([vq[anc], src_c[anc]])).groupby(level=[0, 1]).size() \
    .reindex(pd.MultiIndex.from_arrays([vq[zone], src_c[zone]])).fillna(0).to_numpy()
BASE_F = ["pb", "p2", "pce"] + [f for f in ("n_tset", "a_tset", "hn_eq", "pf_rank", "addr_blank_c", "rank_core") if f in FEAT]
F["pb"], F["p2"] = lg(pb[zone]), lg(p2[zone])
pce_full = np.full(len(pb), np.nan, np.float32)
pce_full[band] = pce
F["pce"] = lg(np.nan_to_num(pce_full[zone], nan=0.5))
for f in BASE_F[3:]:
    F[f] = Xv[zone, FEAT.index(f)]
COLL_F = [c for c in F.columns if c not in BASE_F]
log(f"features: base {BASE_F} | collective {COLL_F}")

# ---------------- 3. cross-fitted stackers (A->B, B->A): control (base only) vs collective
yz = vy[zone]
inA = np.isin(vq[zone], V.A)
prm = dict(objective="binary:logistic", eval_metric="logloss", max_depth=6, eta=0.05, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=5, tree_method="hist")
for name, cols in (("control stacker (base features)", BASE_F), ("collective stacker", BASE_F + COLL_F)):
    out = np.zeros(len(zone), np.float32)
    for tr, te in ((inA, ~inA), (~inA, inA)):
        dtr = xgb.DMatrix(F.loc[F.index[tr], cols].to_numpy(np.float32), label=yz[tr])
        bst = xgb.train(prm, dtr, 400)
        out[te] = bst.predict(xgb.DMatrix(F.loc[F.index[te], cols].to_numpy(np.float32)))
    pn = pb.copy()
    pn[zone] = out
    t, fa, fb, fall = V.ab_score(pn, qq=vq, cc=vc)
    log(f"{name:32s}: t={t:.3f} A={fa:.4f} B={fb:.4f} all={fall:.4f}")
    if cols is not BASE_F:
        imp = pd.Series(bst.get_score(importance_type="gain")).sort_values(ascending=False)
        imp.index = [cols[int(i[1:])] for i in imp.index]
        print(imp.head(15).round(1).to_string())
        np.save(W11 + r"\val_pb_collective.npy", pn)
log("done")
