# Experiment D: per-entity expected-F0.5 maximisation (decision-theoretic approach) vs global threshold.
import numpy as np, pandas as pd, pyarrow.parquet as pq, xgboost as xgb, re, ast, torch, os
W = r"D:\AmazonMLChallenge2026\work\v2"; W3 = r"D:\AmazonMLChallenge2026\work\v3"; D = r"D:\AmazonMLChallenge2026\dataset\train"
rec = pq.read_table(W + r"\train_records.parquet", columns=["entity_id", "src"]).to_pandas()
N = len(rec); srcs = rec.src.to_numpy()
z = np.load(W + r"\val_pairs.npz"); q = z["q"].astype(np.int64); c = z["c"].astype(np.int64); y = z["y"]
cache = W3 + r"\val_p_v2.npz"
if os.path.exists(cache):
    p1, p2 = np.load(cache)["p1"], np.load(cache)["p2"]
else:
    src = open("improved_src.py", encoding="utf-8").read(); ns = {"np": np, "pd": pd}
    exec(re.search(r"^def group_features\(.*?(?=^\S)", src, re.S | re.M).group(0), ns)
    ns["G_FEATURES"] = ast.literal_eval(re.search(r"^G_FEATURES = (\[.*?\])", src, re.S | re.M).group(1))
    m1 = xgb.Booster(); m1.load_model(W + r"\m1.json"); m2 = xgb.Booster(); m2.load_model(W + r"\m2.json")
    p1 = m1.predict(xgb.DMatrix(z["X"], feature_names=m1.feature_names), iteration_range=(0, m1.best_iteration + 1))
    p2 = m2.predict(xgb.DMatrix(np.hstack([z["X"], ns["group_features"](q, srcs[c], p1)]), feature_names=m2.feature_names), iteration_range=(0, m2.best_iteration + 1))
    np.savez(cache, p1=p1, p2=p2)
ids = pd.Index(rec.entity_id)
gt = pd.read_csv(D + "/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
m = gt.matched_entity_ids.str.split(",").explode(); m = m[m != ""]
GQ = ids.get_indexer(gt.source1_entity_id.loc[m.index]); GC = ids.get_indexer(m.values)
VAL = np.unique(q); s_ = np.isin(GQ, VAL); GQ, GC = GQ[s_].astype(np.int64), GC[s_].astype(np.int64)
GK = np.sort(GQ * N + GC)
def f_ent(qset, pq_, pc_):
    k = pq_ * N + pc_; pos = np.minimum(np.searchsorted(GK, k), len(GK) - 1); hit = GK[pos] == k
    inq = np.isin(GQ, qset); nt = np.bincount(np.searchsorted(qset, GQ[inq]), minlength=len(qset))
    npd = np.bincount(np.searchsorted(qset, pq_), minlength=len(qset)); tp = np.bincount(np.searchsorted(qset, pq_[hit]), minlength=len(qset))
    with np.errstate(all="ignore"):
        P = np.where(npd > 0, tp / np.maximum(npd, 1), 0); R = np.where(nt > 0, tp / np.maximum(nt, 1), 0)
        f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    return np.where((nt == 0) & (npd == 0), 1.0, f)

def owner(q, c, p):
    """one owner per S2/S3 record: keep each c only for its highest-probability q"""
    o = np.lexsort((-p, c)); first = np.r_[True, c[o][1:] != c[o][:-1]]; keep = np.zeros(len(c), bool); keep[o[first]] = True
    return keep

def threshold_rule(q, c, p, t):
    k = owner(q, c, p) & (p >= t); d = pd.DataFrame({"q": q[k], "c": c[k], "s": srcs[c[k]], "p": p[k]}).sort_values("p", ascending=False)
    d = d[d.groupby(["q", "s"]).cumcount() < np.where(d.s == 2, 5, 6)]; return d.q.to_numpy(), d.c.to_numpy()

@torch.no_grad()
def expected_f_rule(q, c, p, a=1.0, b=0.0, lam=0.0, nmax=12, S=512, seed=0):
    """DTA: per entity choose the top-k (by prob) that maximises Monte-Carlo expected F0.5.
    p calibrated as sigmoid(a*logit(p)+b); lam = expected #true matches outside the candidate set (Poisson)."""
    k = owner(q, c, p)
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    pc = 1 / (1 + np.exp(-(a * lp + b)))
    d = pd.DataFrame({"q": q[k], "c": c[k], "s": srcs[c[k]], "p": pc[k]}).sort_values(["q", "p"], ascending=[True, False])
    d = d[d.groupby(["q", "s"]).cumcount() < np.where(d.s == 2, 5, 6)]
    d["r"] = d.groupby("q").cumcount(); d = d[d.r < nmax]
    uq, qi = np.unique(d.q.to_numpy(), return_inverse=True)
    P = np.zeros((len(uq), nmax), np.float32); P[qi, d.r.to_numpy()] = d.p.to_numpy()
    best_k = np.zeros(len(uq), np.int64)
    g = torch.Generator(device="cuda").manual_seed(seed)
    for a0 in range(0, len(uq), 20000):
        Pt = torch.from_numpy(P[a0:a0 + 20000]).cuda()
        Y = (torch.rand((Pt.shape[0], S, nmax), device="cuda", generator=g) < Pt[:, None, :]).float()
        M = torch.poisson(torch.full((Pt.shape[0], S), lam, device="cuda"), generator=g) if lam > 0 else 0
        T = Y.sum(2) + M                                          # total true matches per sample
        TP = torch.cumsum(Y, 2)                                   # TP for prefix sizes 1..nmax
        ks = torch.arange(1, nmax + 1, device="cuda").float()
        F = torch.where(TP > 0, 1.25 * TP / (0.25 * T[..., None] + ks), torch.zeros_like(TP))
        EF = torch.cat([(T == 0).float().mean(1, keepdim=True), F.mean(1)], 1)   # k = 0 .. nmax
        best_k[a0:a0 + 20000] = EF.argmax(1).cpu().numpy()
    keep = d.r.to_numpy() < best_k[qi]
    return d.q.to_numpy()[keep], d.c.to_numpy()[keep]

rng = np.random.default_rng(7); half = rng.random(len(VAL)) < 0.5
A, B = VAL[half], VAL[~half]
inA = np.isin(q, A)
for name, p in (("stage-1 p1", p1), ("group-context p2", p2)):
    thr = {t: f_ent(A, *threshold_rule(q[inA], c[inA], p[inA], t)).mean() for t in np.arange(0.5, 0.91, 0.025)}
    tb = max(thr, key=thr.get)
    fB_thr = f_ent(B, *threshold_rule(q[~inA], c[~inA], p[~inA], tb)).mean()
    best, bestcfg = -1, None
    for a in (0.8, 1.0, 1.25, 1.5):
        for b in (-1.0, -0.5, 0.0, 0.5):
            for lam in (0.0, 0.1):
                fA = f_ent(A, *expected_f_rule(q[inA], c[inA], p[inA], a, b, lam)).mean()
                if fA > best: best, bestcfg = fA, (a, b, lam)
    fB_dta = f_ent(B, *expected_f_rule(q[~inA], c[~inA], p[~inA], *bestcfg)).mean()
    print(f"{name}: threshold t={tb:.3f}  A={thr[tb]:.4f}  B={fB_thr:.4f}  |  expected-F0.5 cfg(a,b,lam)={bestcfg}  A={best:.4f}  B={fB_dta:.4f}  (gain on B {fB_dta - fB_thr:+.4f})")
