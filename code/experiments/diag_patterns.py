# Pattern-class audit: does France accept each kind of pair at the rate the labelled countries say it should?
# Classes are defined from record attributes only (no model), so they are comparable across countries. On validation
# we know each class's true-match rate; if the data generator is the same in every country, a class whose acceptance
# in France deviates strongly from validation is miscalibrated there.
import ast
import csv
import os
import re

os.environ["HF_HOME"] = r"D:\AmazonMLChallenge2026\work\hf"
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import xgboost as xgb
from rapidfuzz import fuzz, process
from sklearn.linear_model import LogisticRegression

import valscore as V

SH = r"D:\AmazonMLChallenge2026\work\v2"
W4 = r"D:\AmazonMLChallenge2026\work\v4"
OUT = r"D:\AmazonMLChallenge2026\output"
STOP = {"r", "rue", "st", "ave", "blvd", "rd", "dr", "ln", "all", "imp", "rte", "ch", "des", "les", "du", "de", "la",
        "le", "no", "n", "s", "e", "w", "the", "of", "main", "cross", "road"}
cols = ["entity_id", "country", "src", "squash", "core", "hn", "addr"]


def street(addr, hn):
    t = addr.split()
    if hn and hn in t:
        i = t.index(hn)
        for u in t[i + 1:i + 6]:
            if u.isalpha() and len(u) >= 3 and u not in STOP:
                return u
    return ""


def classes(tb, q, c):
    """pattern class of each (q, c) pair, from raw attributes"""
    take = lambda col, idx: np.array(tb[col].take(pa.array(idx)).to_pylist(), dtype=object)
    cq, cc = take("core", q), take("core", c)
    sq, sc = take("squash", q), take("squash", c)
    hq, hc = take("hn", q), take("hn", c)
    aq, ac = take("addr", q), take("addr", c)
    stq = np.array([street(a, h) for a, h in zip(aq, hq)], dtype=object)
    stc = np.array([street(a, h) for a, h in zip(ac, hc)], dtype=object)
    ns = process.cpdist(list(cq), list(cc), scorer=fuzz.token_set_ratio, workers=-1)
    same_name = sq == sc
    same_hn = (hq == hc) & (hq != "")
    addr_match = same_hn & (stq == stc) & (stq != "")
    blank = ac == ""
    cls = np.full(len(q), "F other", dtype=object)
    cls[same_name & blank] = "E same name, cand. address blank"
    cls[same_name & ~blank & (hq != "") & (hc != "") & ~same_hn] = "B same name, different house no."
    cls[~same_name & addr_match & (ns >= 50)] = "D name variant, same address"
    cls[~same_name & addr_match & (ns < 50)] = "C different name (trade/acronym), same address"
    cls[same_name & addr_match] = "A same name, same address"
    return cls


# ---------------- validation: true rate and v4 acceptance per class
tr = pq.read_table(SH + r"\train_records.parquet", columns=cols).combine_chunks()
vz = np.load(W4 + r"\val_pairs.npz")
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
cache = W4 + r"\val_p_v4.npy"
if os.path.exists(cache):
    pb = np.load(cache)
else:
    src = open("v4_src.py", encoding="utf-8").read()

    class Rec:
        src = tr["src"].to_numpy()
        n = tr.num_rows

        @staticmethod
        def take(col, idx):
            full = pq.read_table(SH + r"\train_records.parquet", columns=[col]).column(0) if col not in tr.column_names else tr[col]
            return full.take(pa.array(idx)).to_pylist()

    ns_ = {"np": np, "pd": pd, "process": process, "fuzz": fuzz}
    for fn in ("group_features", "cross_source_feats", "group_ctx"):
        exec(re.search(rf"^def {fn}\(.*?(?=^\S)", src, re.S | re.M).group(0), ns_)
    ns_["G_FEATURES"] = ast.literal_eval(re.search(r"^G_FEATURES = (\[.*?\])", src, re.S | re.M).group(1))
    m1 = xgb.Booster(model_file=W4 + r"\m1.json")
    m2 = xgb.Booster(model_file=W4 + r"\m2.json")
    pr = lambda b, X: b.predict(xgb.DMatrix(X, feature_names=b.feature_names), iteration_range=(0, b.best_iteration + 1))
    p1 = pr(m1, vz["X"])
    p2 = pr(m2, np.hstack([vz["X"], ns_["group_ctx"](Rec, vq, vc, p1)]))
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    CE = r"D:\AmazonMLChallenge2026\work\v3\ce_model"
    tok = AutoTokenizer.from_pretrained(CE)
    ce = AutoModelForSequenceClassification.from_pretrained(CE).cuda().eval()
    names = pq.read_table(SH + r"\train_records.parquet", columns=["name"]).column(0)

    def ser(idx):
        out = []
        for n, a, h in zip(names.take(pa.array(idx)).to_pylist(), tr["addr"].take(pa.array(idx)).to_pylist(),
                           tr["hn"].take(pa.array(idx)).to_pylist()):
            if h:
                a = (" " + a + " ").replace(f" {h} ", f" [NUM] {h} [/NUM] ", 1).strip()
            out.append(f"[COL] name [VAL] {n} [COL] addr [VAL] {a if a else 'none'}")
        return out

    band = np.flatnonzero((p2 >= 0.01) & (p2 <= 0.99))
    pce = np.zeros(len(band), np.float32)
    with torch.no_grad():
        for a in range(0, len(band), 256):
            b_ = band[a:a + 256]
            enc = tok(ser(vq[b_]), ser(vc[b_]), truncation=True, max_length=96, padding=True, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.float16):
                pce[a:a + 256] = torch.sigmoid(ce(**{k: v.cuda() for k, v in enc.items()}).logits.squeeze(-1).float()).cpu().numpy()
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    Xb = np.column_stack([lg(p2[band]), lg(pce)])
    pb = p2.copy()
    pb[band] = LogisticRegression(C=10.0).fit(Xb, vy[band]).predict_proba(Xb)[:, 1]
    np.save(cache, pb)
mq, mc = V.threshold_rule(vq, vc, pb, 0.70)
acc_val = np.isin(vq * V.N + vc, mq * V.N + mc)
cls_v = classes(tr, vq, vc)
vctry = tr["country"].take(pa.array(vq)).to_numpy(zero_copy_only=False)
dfv = pd.DataFrame({"cls": cls_v, "true": vy == 1, "acc": acc_val, "country": vctry})

# ---------------- test: v4 and v5 acceptance per class and country
te = pq.read_table(SH + r"\test_records.parquet", columns=cols).combine_chunks()
tid = pd.Index(te["entity_id"].to_pandas())
tctry = te["country"].to_numpy(zero_copy_only=False)
cand = pd.read_csv(OUT + r"\candidate_pairs_v4.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
cand = cand.sample(frac=0.25, random_state=0)                  # 25% of S1 entities is plenty for rates
lists = cand.candidate_entity_ids.str.split(",")
tq = tid.get_indexer(np.repeat(cand.source1_entity_id.to_numpy(), lists.str.len().to_numpy()))
tcc = tid.get_indexer(np.concatenate(lists.to_numpy()))
cls_t = classes(te, tq, tcc)
dft = pd.DataFrame({"cls": cls_t, "country": tctry[tq]})
key = pd.Series(tq.astype(np.int64) * te.num_rows + tcc)
for ver in ("v4", "v5"):
    r = pd.read_csv(OUT + rf"\matching_results_{ver}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    m = r.assign(c=r.matched_entity_ids.str.split(",")).explode("c")
    m = m[m.c != ""]
    k = set((tid.get_indexer(m.source1_entity_id).astype(np.int64) * te.num_rows + tid.get_indexer(m.c)).tolist())
    dft[ver] = key.isin(k).to_numpy()

pd.set_option("display.width", 250)
print("VALIDATION (US+India): share of candidate pairs, TRUE match rate, v4 acceptance")
g = dfv.groupby("cls").agg(share=("true", "size"), true_rate=("true", "mean"), v4_accept=("acc", "mean"))
g["share"] = g.share / len(dfv)
print(g.round(4).to_string())
print("\nVALIDATION by country: TRUE rate / v4 acceptance")
print(dfv.groupby(["cls", "country"]).agg(true_rate=("true", "mean"), v4_accept=("acc", "mean")).unstack().round(4).to_string())
print("\nTEST: share of candidate pairs and acceptance per country (v4 / v5)")
t = dft.groupby(["cls", "country"]).agg(share=("v4", "size"), v4=("v4", "mean"), v5=("v5", "mean"))
t["share"] = t.share / t.groupby(level=1).share.transform("sum")
print(t.unstack().round(4).to_string())
