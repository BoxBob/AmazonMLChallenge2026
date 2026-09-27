# v11 (NEXT_PLAN phase 1, "A-lite"): a data-rich, harder-trained multilingual cross-encoder.
# v9's cross-encoder saw 1.2M pairs from 150k entities (top-8), one epoch, no augmentation, and only the 3% band.
# Here (Ditto 2020; Peeters & Bizer 2024; AnyMatch 2024):
#   * data: all 400k training entities, top-16 candidates, hard negatives kept (GBDT p1 >= 0.001) + a sample of easy ones,
#     GBDT-disagreeing pairs over-sampled x2  -> ~3.5M pairs; warm start from the v9 model (effectively a 2nd epoch)
#   * Ditto augmentation: attribute deletion (blank the candidate's address: our largest error class) and token deletion
#   * wide band (0.001-0.999) vs the v9 band (0.01-0.99)
# Evaluated on the v10 validation candidate set (top-80) through the same v4 group model + blend -> hold-out B.
# Gate (NEXT_PLAN): hold-out >= v10 + 0.001. Resumable: the model is checkpointed every CKPT_EVERY steps.
import ast
import json
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
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import valscore as V

SH = r"D:\AmazonMLChallenge2026\work\v2"
W4 = r"D:\AmazonMLChallenge2026\work\v4"
W9 = r"D:\AmazonMLChallenge2026\work\v9"
W10 = r"D:\AmazonMLChallenge2026\work\v10m80"
W11 = r"D:\AmazonMLChallenge2026\work\v11"
os.makedirs(W11, exist_ok=True)
TAG = os.environ.get("V11_TAG", "")          # "" = first recipe; "b" = no attribute deletion / no error duplicates
OLD_DIR = W9 + r"\mce_model"
NEW_DIR = W11 + rf"\mce_model{TAG}"
CKPT_DIR = W11 + rf"\mce_ckpt{TAG}"
TOPK, BATCH, LR, MAX_LEN, TARGET_PAIRS, CKPT_EVERY = 16, 64, 2e-5, 96, 3_500_000, 5000
P_ADDR_DEL = float(os.environ.get("V11_ADDR_DEL", "0.15"))
P_TOK_DEL = float(os.environ.get("V11_TOK_DEL", "0.05"))
ERR_DUP = os.environ.get("V11_ERR_DUP", "1") == "1"
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


def serialize(idx, rng=None):
    out = []
    for n, a, h in zip(Rec.take("name", idx), Rec.take("addr", idx), Rec.take("hn", idx)):
        if rng is not None:                                    # Ditto augmentation (training only)
            if a and rng.random() < P_ADDR_DEL:
                a, h = "", ""
            toks = n.split()
            if len(toks) >= 3 and rng.random() < P_TOK_DEL:
                del toks[rng.integers(len(toks))]
                n = " ".join(toks)
        if h:
            a = (" " + a + " ").replace(f" {h} ", f" [NUM] {h} [/NUM] ", 1).strip()
        out.append(f"[COL] name [VAL] {n} [COL] addr [VAL] {a if a else 'none'}")
    return out


def batches(tok, qi, ci, bs, rng=None, start=0):
    for a in range(start, len(qi), bs):
        enc = tok(serialize(qi[a:a + bs]), serialize(ci[a:a + bs], rng), truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        yield a, {k: v.cuda() for k, v in enc.items()}


m1 = xgb.Booster(model_file=W4 + r"\m1.json")
m2 = xgb.Booster(model_file=W4 + r"\m2.json")
pr = lambda b, X: b.predict(xgb.DMatrix(X, feature_names=b.feature_names), iteration_range=(0, b.best_iteration + 1))

# ---------------- 1. training set: 400k entities, top-16, hard negatives + sampled easy ones, GBDT errors x2
sel_path = W11 + (r"\train_sel.npz" if ERR_DUP else r"\train_sel_nodup.npz")
if os.path.exists(sel_path):
    z = np.load(sel_path)
    sq, sc, sy = z["q"], z["c"], z["y"]
else:
    tz = np.load(W4 + r"\train_pairs.npz")
    tq, tc, ty, X = tz["q"].astype(np.int64), tz["c"].astype(np.int64), tz["y"], tz["X"]
    pf_rank = X[:, m1.feature_names.index("pf_rank")]
    p1 = pr(m1, X)
    del X, tz
    top = pf_rank < TOPK
    pos, hard = top & (ty == 1), top & (ty == 0) & (p1 >= 0.001)
    easy = np.flatnonzero(top & (ty == 0) & (p1 < 0.001))
    n_easy = max(0, TARGET_PAIRS - pos.sum() - hard.sum())
    easy = np.random.default_rng(5).choice(easy, min(n_easy, len(easy)), replace=False)
    err = np.flatnonzero(top & (np.abs(ty - p1) > 0.3)) if ERR_DUP else np.array([], np.int64)  # GBDT errors x2
    idx = np.concatenate([np.flatnonzero(pos), np.flatnonzero(hard), easy, err])
    idx = np.random.default_rng(6).permutation(idx)
    log(f"training pairs: entities {len(np.unique(tq)):,} | top-{TOPK} positives {pos.sum():,}, hard negatives {hard.sum():,}, "
        f"easy negatives sampled {len(easy):,}, GBDT-error duplicates {len(err):,} -> {len(idx):,}")
    sq, sc, sy = tq[idx], tc[idx], ty[idx].astype(np.float32)
    np.savez(sel_path, q=sq, c=sc, y=sy)
    del tq, tc, ty, p1, pf_rank

# ---------------- 2. warm-start fine-tuning from the v9 model (resumable)
if not os.path.exists(NEW_DIR + r"\config.json"):
    steps = math.ceil(len(sq) / BATCH)
    state = json.load(open(CKPT_DIR + r"\state.json")) if os.path.exists(CKPT_DIR + r"\state.json") else {"step": 0}
    src_dir = CKPT_DIR if state["step"] else OLD_DIR
    tok = AutoTokenizer.from_pretrained(OLD_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(src_dir).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    for _ in range(state["step"]):
        sched.step()
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(100 + state["step"])
    yt = torch.from_numpy(sy.astype(np.float32))
    log(f"fine-tuning from {src_dir}: {len(sq):,} pairs, {steps:,} steps, resuming at step {state['step']:,}")
    run = 0.0
    for i, (a, enc) in enumerate(batches(tok, sq, sc, BATCH, rng, start=state["step"] * BATCH)):
        step = state["step"] + i
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
        if step and step % CKPT_EVERY == 0:
            model.save_pretrained(CKPT_DIR)
            json.dump({"step": step + 1}, open(CKPT_DIR + r"\state.json", "w"))
    model.save_pretrained(NEW_DIR)
    tok.save_pretrained(NEW_DIR)
    del model
    torch.cuda.empty_cache()
    log("fine-tuning done")

# ---------------- 3. v4 group-model probabilities on the v10 validation candidates (top-80; cached)
vz = np.load(W10 + r"\val_pairs.npz")
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
p2_path = W11 + r"\val_p2_v10.npy"
if os.path.exists(p2_path):
    p2 = np.load(p2_path)
else:
    src = open("v10m_src.py", encoding="utf-8").read()
    ns = {"np": np, "pd": pd, "process": process, "fuzz": fuzz}
    for fn in ("group_features", "cross_source_feats", "group_ctx"):
        exec(re.search(rf"^def {fn}\(.*?(?=^\S)", src, re.S | re.M).group(0), ns)
    ns["G_FEATURES"] = ast.literal_eval(re.search(r"^G_FEATURES = (\[.*?\])", src, re.S | re.M).group(1))
    X = vz["X"]
    p1 = pr(m1, X)
    p2 = pr(m2, np.hstack([X, ns["group_ctx"](Rec, vq, vc, p1)]))
    del X
    np.save(p2_path, p2)
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
t, fa, fb, fall = V.ab_score(p2, qq=vq, cc=vc)
log(f"{'no cross-encoder':34s}: t={t:.3f} A={fa:.4f} B={fb:.4f} all={fall:.4f}")

# ---------------- 4. score the wide band with both cross-encoders; blend (fit on half A) for each band width
wide = np.flatnonzero((p2 >= 0.001) & (p2 <= 0.999))
log(f"validation wide-band pairs {len(wide):,} (narrow {((p2 >= 0.01) & (p2 <= 0.99)).sum():,})")
res = {}
for name, d in (("v9 mce", OLD_DIR), (f"v11{TAG} mce", NEW_DIR)):
    cache = W11 + rf"\val_pce_{name.split()[0]}.npy"
    if os.path.exists(cache):
        pce = np.load(cache)
    else:
        tok = AutoTokenizer.from_pretrained(d)
        ce = AutoModelForSequenceClassification.from_pretrained(d).cuda().eval()
        pce = np.zeros(len(wide), np.float32)
        with torch.no_grad():
            for a, enc in batches(tok, vq[wide], vc[wide], 256):
                with torch.autocast("cuda", dtype=torch.float16):
                    pce[a:a + 256] = torch.sigmoid(ce(**enc).logits.squeeze(-1).float()).cpu().numpy()
        del ce
        torch.cuda.empty_cache()
        np.save(cache, pce)
    for lo, hi in ((0.01, 0.99), (0.001, 0.999)):
        m = (p2[wide] >= lo) & (p2[wide] <= hi)
        band = wide[m]
        Xb = np.column_stack([lg(p2[band]), lg(pce[m])])
        inA = np.isin(vq[band], V.A)
        lr = LogisticRegression(C=10.0).fit(Xb[inA], vy[band][inA])
        pb = p2.copy()
        pb[band] = lr.predict_proba(Xb)[:, 1]
        t, fa, fb, fall = V.ab_score(pb, qq=vq, cc=vc)
        auc = roc_auc_score(vy[band], pce[m])
        res[f"{name} band {lo}-{hi}"] = fb
        log(f"{name + f' band {lo}-{hi}':34s}: CE AUC {auc:.4f} | blend {lr.coef_.round(3).tolist()} | "
            f"t={t:.3f} A={fa:.4f} B={fb:.4f} all={fall:.4f}")
json.dump(res, open(W11 + rf"\mce_results{TAG}.json", "w"), indent=1)
log("done")
