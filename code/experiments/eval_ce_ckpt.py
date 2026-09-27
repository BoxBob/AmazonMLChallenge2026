# Quick check of a cross-encoder checkpoint on the v10 validation band (0.01-0.99), same blend + A/B protocol as exp_v11_mce.
# usage: python eval_ce_ckpt.py <model_dir> <tag>
import os, sys
os.environ["HF_HOME"] = r"D:\AmazonMLChallenge2026\work\hf"
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import valscore as V
W11 = r"D:\AmazonMLChallenge2026\work\v11"
d, tag = sys.argv[1], sys.argv[2]
tb = pq.read_table(r"D:\AmazonMLChallenge2026\work\v2\train_records.parquet", columns=["name", "addr", "hn"]).combine_chunks()
take = lambda col, idx: tb[col].take(pa.array(idx)).to_pylist()
def ser(idx):
    out = []
    for n, a, h in zip(take("name", idx), take("addr", idx), take("hn", idx)):
        if h:
            a = (" " + a + " ").replace(f" {h} ", f" [NUM] {h} [/NUM] ", 1).strip()
        out.append(f"[COL] name [VAL] {n} [COL] addr [VAL] {a if a else 'none'}")
    return out
vz = np.load(r"D:\AmazonMLChallenge2026\work\v10m80\val_pairs.npz")
vq, vc, vy = vz["q"].astype(np.int64), vz["c"].astype(np.int64), vz["y"]
p2 = np.load(W11 + r"\val_p2_v10.npy")
band = np.flatnonzero((p2 >= 0.01) & (p2 <= 0.99))
tok = AutoTokenizer.from_pretrained(r"D:\AmazonMLChallenge2026\work\v9\mce_model")
ce = AutoModelForSequenceClassification.from_pretrained(d).cuda().eval()
pce = np.zeros(len(band), np.float32)
with torch.no_grad():
    for a in range(0, len(band), 128):
        enc = tok(ser(vq[band[a:a + 128]]), ser(vc[band[a:a + 128]]), truncation=True, max_length=96, padding=True, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.float16):
            pce[a:a + 128] = torch.sigmoid(ce(**{k: v.cuda() for k, v in enc.items()}).logits.squeeze(-1).float()).cpu().numpy()
np.save(W11 + rf"\val_pce_{tag}_narrow.npy", pce)
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
for nm, pc in (("v9 mce", np.load(W11 + r"\val_pce_v9_narrow.npy")), (tag, pce)):
    Xb = np.column_stack([lg(p2[band]), lg(pc)]); inA = np.isin(vq[band], V.A)
    lr = LogisticRegression(C=10.0).fit(Xb[inA], vy[band][inA])
    pb = p2.copy(); pb[band] = lr.predict_proba(Xb)[:, 1]
    t, fa, fb, fall = V.ab_score(pb, qq=vq, cc=vc)
    print(f"{nm:14s}: band AUC {roc_auc_score(vy[band], pc):.4f} | blend {lr.coef_.round(3).tolist()} | t={t:.3f} A={fa:.4f} B={fb:.4f} all={fall:.4f}", flush=True)
