# Experiment A: Ditto-style cross-encoder re-ranker for uncertain pairs.
# Records are serialized as "[COL] name [VAL] ... [COL] addr [VAL] ..." with the house number wrapped in
# [NUM] ... [/NUM] (Ditto's span typing), then a small Apache-2.0 cross-encoder is fine-tuned on hard pairs.
import math
import os
import sys
import time

os.environ["HF_HOME"] = r"D:\AmazonMLChallenge2026\work\hf"
os.environ["TEMP"] = os.environ["TMP"] = r"D:\AmazonMLChallenge2026\work\tmp"
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

W = r"D:\AmazonMLChallenge2026\work\v2"
W3 = r"D:\AmazonMLChallenge2026\work\v3"
MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"
MAX_LEN = 96
BATCH = 64
LR = 3e-5
N_TRAIN_ENT = int(os.environ.get("CE_TRAIN_ENT", 150_000))
TOPK_TRAIN = 8
BAND = (0.01, 0.99)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


tbl = pq.read_table(W + r"\train_records.parquet", columns=["name", "addr", "hn"]).combine_chunks()


def serialize(idx):
    names = tbl["name"].take(pa.array(idx)).to_pylist()
    addrs = tbl["addr"].take(pa.array(idx)).to_pylist()
    hns = tbl["hn"].take(pa.array(idx)).to_pylist()
    out = []
    for n, a, h in zip(names, addrs, hns):
        if h:
            a = (" " + a + " ").replace(f" {h} ", f" [NUM] {h} [/NUM] ", 1).strip()
        out.append(f"[COL] name [VAL] {n} [COL] addr [VAL] {a if a else 'none'}")
    return out


tok = AutoTokenizer.from_pretrained(MODEL)
tok.add_special_tokens({"additional_special_tokens": ["[COL]", "[VAL]", "[NUM]", "[/NUM]"]})
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
model.resize_token_embeddings(len(tok))
model.cuda()
log(f"model {MODEL}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

# ---------------- training pairs: top-8 candidates by prefilter score for a sample of training entities
tz = np.load(W + r"\train_pairs.npz")
feat_names = __import__("xgboost").Booster(model_file=W + r"\m1.json").feature_names
pf_rank = tz["X"][:, feat_names.index("pf_rank")]
tq, tc, ty = tz["q"].astype(np.int64), tz["c"].astype(np.int64), tz["y"]
ents = np.random.default_rng(3).choice(np.unique(tq), N_TRAIN_ENT, replace=False)
sel = np.flatnonzero(np.isin(tq, ents) & (pf_rank < TOPK_TRAIN))
sel = np.random.default_rng(4).permutation(sel)
log(f"CE training pairs {len(sel):,} (positives {ty[sel].mean():.1%})")
del tz


def batches(qi, ci, bs):
    for a in range(0, len(qi), bs):
        enc = tok(serialize(qi[a:a + bs]), serialize(ci[a:a + bs]), truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        yield a, {k: v.cuda(non_blocking=True) for k, v in enc.items()}


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = math.ceil(len(sel) / BATCH)
sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()
model.train()
yt = torch.from_numpy(ty[sel].astype(np.float32))
run = 0.0
for step, (a, enc) in enumerate(batches(tq[sel], tc[sel], BATCH)):
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(**enc).logits.squeeze(-1)
    loss = lossf(logits.float(), yt[a:a + BATCH].cuda())
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
model.save_pretrained(W3 + r"\ce_model")
tok.save_pretrained(W3 + r"\ce_model")
log("training done")

# ---------------- score the uncertain band of validation pairs
vz = np.load(W + r"\val_pairs.npz")
vq, vc = vz["q"].astype(np.int64), vz["c"].astype(np.int64)
p2 = np.load(W3 + r"\val_p_exp.npz")["p2"]   # base = v3 feature-experiment model
band = np.flatnonzero((p2 >= BAND[0]) & (p2 <= BAND[1]))
log(f"val band pairs {len(band):,} ({len(band) / len(p2):.1%} of val pairs)")
model.eval()
pce = np.zeros(len(band), np.float32)
with torch.no_grad():
    for a, enc in batches(vq[band], vc[band], 256):
        with torch.autocast("cuda", dtype=torch.float16):
            pce[a:a + 256] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
np.savez(W3 + r"\val_ce.npz", band=band, pce=pce)
log("val scoring done")

# ---------------- blend with the GBDT probability (weights fit on half A, reported on half B)
sys.path.insert(0, r"D:\AmazonMLChallenge2026\work\nb_build")
import valscore as V  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
yv = vz["y"]
inA = np.isin(vq[band], V.A)
Xb = np.column_stack([lg(p2[band]), lg(pce)])
lr = LogisticRegression(C=10.0).fit(Xb[inA], yv[band][inA])
log(f"blend weights {lr.coef_.round(3)} bias {lr.intercept_.round(3)}")
pb = p2.copy()
pb[band] = lr.predict_proba(Xb)[:, 1]
pce_full = p2.copy()
pce_full[band] = pce
for name, p in (("exp p2 (base)", p2), ("CE replaces p2 in band", pce_full), ("blend", pb)):
    t, fa, fb, fall = V.ab_score(p)
    print(f"{name:24s} t={t:.3f}  A={fa:.4f}  B={fb:.4f}  all={fall:.4f}", flush=True)
