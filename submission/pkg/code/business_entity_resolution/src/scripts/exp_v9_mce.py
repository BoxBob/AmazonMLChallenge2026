# v9 experiment: multilingual cross-encoder (Apache-2.0 cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) vs the current English
# MiniLM cross-encoder. Same fine-tuning recipe; compared on US/India validation through the same v4 group model and blend.
# Motivation (T25): the cross-encoder is the component that moves France most (+20k French pairs), and it is English-only.
import ast
import math
import os
import re
import sys
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import valscore as V

SH = r"D:\AmazonMLChallenge2026\work\v2"
W4 = r"D:\AmazonMLChallenge2026\work\v4"
W9 = r"D:\AmazonMLChallenge2026\work\v9"
os.makedirs(W9, exist_ok=True)
NEW_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
OLD_DIR = r"D:\AmazonMLChallenge2026\work\v3\ce_model"
NEW_DIR = W9 + r"\mce_model"
N_ENT, TOPK, BATCH, LR, MAX_LEN = 150_000, 8, 64, 3e-5, 96
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


def serialize(idx):
    out = []
    for n, a, h in zip(Rec.take("name", idx), Rec.take("addr", idx), Rec.take("hn", idx)):
        if h:
            a = (" " + a + " ").replace(f" {h} ", f" [NUM] {h} [/NUM] ", 1).strip()
        out.append(f"[COL] name [VAL] {n} [COL] addr [VAL] {a if a else 'none'}")
    return out


def batches(tok, qi, ci, bs):
    for a in range(0, len(qi), bs):
        enc = tok(serialize(qi[a:a + bs]), serialize(ci[a:a + bs]), truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        yield a, {k: v.cuda() for k, v in enc.items()}


# ---------------- 1. fine-tune the multilingual cross-encoder (identical recipe; skipped if cached)
tz = np.load(W4 + r"\train_pairs.npz")
m1 = xgb.Booster(model_file=W4 + r"\m1.json")
m2 = xgb.Booster(model_file=W4 + r"\m2.json")
tq, tc, ty = tz["q"].astype(np.int64), tz["c"].astype(np.int64), tz["y"]
pf_rank = tz["X"][:, m1.feature_names.index("pf_rank")]
del tz
if not os.path.exists(NEW_DIR + r"\config.json"):
    tok = AutoTokenizer.from_pretrained(NEW_MODEL)
    tok.add_special_tokens({"additional_special_tokens": ["[COL]", "[VAL]", "[NUM]", "[/NUM]"]})
    model = AutoModelForSequenceClassification.from_pretrained(NEW_MODEL, num_labels=1)
    model.resize_token_embeddings(len(tok))
    model.cuda().train()
    ents = np.random.default_rng(3).choice(np.unique(tq), N_ENT, replace=False)
    sel = np.random.default_rng(4).permutation(np.flatnonzero(np.isin(tq, ents) & (pf_rank < TOPK)))
    yt = torch.from_numpy(ty[sel].astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = math.ceil(len(sel) / BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    log(f"fine-tuning {NEW_MODEL}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, {len(sel):,} pairs")
    run = 0.0
    for step, (a, enc) in enumerate(batches(tok, tq[sel], tc[sel], BATCH)):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), yt[a:a + BATCH].cuda())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if step % 2000 == 0:
            log(f"  step {step:,}/{steps:,} loss {run:.4f}")
    model.save_pretrained(NEW_DIR)
    tok.save_pretrained(NEW_DIR)
    del model
    torch.cuda.empty_cache()
    log("fine-tuning done")

# ---------------- 2. v4 group-model probabilities on validation (cached)
vz = np.load(W4 + r"\val_pairs.npz")
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
p2_path = W9 + r"\val_p2_v4.npy"
if os.path.exists(p2_path):
    p2 = np.load(p2_path)
else:
    src = open("v4_src.py", encoding="utf-8").read()
    ns = {"np": np, "pd": pd, "process": process, "fuzz": fuzz}
    for fn in ("group_features", "cross_source_feats", "group_ctx"):
        exec(re.search(rf"^def {fn}\(.*?(?=^\S)", src, re.S | re.M).group(0), ns)
    ns["G_FEATURES"] = ast.literal_eval(re.search(r"^G_FEATURES = (\[.*?\])", src, re.S | re.M).group(1))
    pr = lambda b, X: b.predict(xgb.DMatrix(X, feature_names=b.feature_names), iteration_range=(0, b.best_iteration + 1))
    p1 = pr(m1, vz["X"])
    p2 = pr(m2, np.hstack([vz["X"], ns["group_ctx"](Rec, vq, vc, p1)]))
    np.save(p2_path, p2)
band = np.flatnonzero((p2 >= 0.01) & (p2 <= 0.99))
log(f"validation band pairs {len(band):,}")

# ---------------- 3. score the band with both cross-encoders; blend each (fit on half A), score on half B / all
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
res = {}
for name, d in (("english MiniLM (v3/v4)", OLD_DIR), ("multilingual mMiniLMv2", NEW_DIR)):
    tok = AutoTokenizer.from_pretrained(d)
    ce = AutoModelForSequenceClassification.from_pretrained(d).cuda().eval()
    pce = np.zeros(len(band), np.float32)
    with torch.no_grad():
        for a, enc in batches(tok, vq[band], vc[band], 256):
            with torch.autocast("cuda", dtype=torch.float16):
                pce[a:a + 256] = torch.sigmoid(ce(**enc).logits.squeeze(-1).float()).cpu().numpy()
    del ce
    torch.cuda.empty_cache()
    np.save(W9 + rf"\val_pce_{'new' if d == NEW_DIR else 'old'}.npy", pce)
    X = np.column_stack([lg(p2[band]), lg(pce)])
    inA = np.isin(vq[band], V.A)
    lr = LogisticRegression(C=10.0).fit(X[inA], vy[band][inA])
    pb = p2.copy()
    pb[band] = lr.predict_proba(X)[:, 1]
    t, fa, fb, fall = V.ab_score(pb, qq=vq, cc=vc)
    auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(vy[band], pce)
    res[name] = fb
    log(f"{name:26s}: band AUC alone {auc:.4f} | blend weights {lr.coef_.round(3).tolist()} | t={t:.3f} A={fa:.4f} B={fb:.4f} all={fall:.4f}")
t, fa, fb, fall = V.ab_score(p2, qq=vq, cc=vc)
log(f"{'no cross-encoder':26s}: t={t:.3f} A={fa:.4f} B={fb:.4f} all={fall:.4f}")
