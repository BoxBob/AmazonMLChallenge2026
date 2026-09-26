# v7 France threshold on the address-normalized probabilities (v7fr), with the swap filter, forecast by FES using the
# calibration learned from the v7fr tie: French borderline pairs are ~+0.1..+0.3 more precise than US/India priors.
import csv
import json
import subprocess
import sys

import numpy as np
import pandas as pd

import fes

OUT = r"D:\AmazonMLChallenge2026\output"
BASE = OUT + r"\lb_probe\matching_results_fr_noswap_f300.tsv"          # current best, LB 0.974
z = np.load(r"D:\AmazonMLChallenge2026\work\v7fr\france_v7fr_probs.npz", allow_pickle=True)
q, c, p, ids = z["q"], z["c"], z["p"], z["ids"]
src = np.array([int(x[1]) for x in ids[c]])
MATCH_CAP = {2: 5, 3: 6}


def resolve(t):
    d = pd.DataFrame({"q": q, "c": c, "s": src, "p": p})
    d = d[d.p >= t].sort_values("p", ascending=False).drop_duplicates("c")
    d = d[d.groupby(["q", "s"]).cumcount() < d.s.map(MATCH_CAP)]
    return d.q.to_numpy(), d.c.to_numpy()


base = pd.read_csv(BASE, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
fr_ids = set(ids[np.unique(q)])
is_fr = base.source1_entity_id.isin(fr_ids)
st = json.load(open(fes.STATE))
g = st["marginals"]
pa_, pr_ = dict(st["prec_add"]), dict(st["prec_rem"])
pa_["SWAP"], pr_["SWAP"] = st["t_swap_add"], st["t_swap_rem"]
for t in [float(x) for x in sys.argv[1:]]:
    mq, mc = resolve(t)
    lists = pd.Series(ids[mc]).groupby(ids[mq]).agg(",".join)
    raw = base.copy()
    raw.loc[is_fr, "matched_entity_ids"] = raw.loc[is_fr, "source1_entity_id"].map(lists).fillna("").to_numpy()
    raw_path = OUT + rf"\lb_probe\matching_results_v7fr_raw_t{t:.3f}.tsv"
    raw.to_csv(raw_path, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    out_path = OUT + rf"\lb_probe\matching_results_v7fr_t{t:.3f}.tsv"
    subprocess.run([sys.executable, "exp_v6_descswap.py", "300", raw_path, out_path], capture_output=True, text=True)
    tab = fes.diff_by_class(out_path, base=BASE)
    fc = {s: fes.expected_delta2(tab, {k: min(1, v + s) for k, v in pa_.items()}, pr_, g) for s in (0.1, 0.2, 0.3)}
    print(f"France t={t:.3f}: French pairs {len(mq):,} before swap filter | vs 0.974 file: +{tab.added.sum():,} / -{tab.removed.sum():,} | "
          f"FES forecast (French +0.1/+0.2/+0.3 calibration): {fc[0.1]:+.4f} / {fc[0.2]:+.4f} / {fc[0.3]:+.4f}  -> {out_path.split(chr(92))[-1]}", flush=True)
