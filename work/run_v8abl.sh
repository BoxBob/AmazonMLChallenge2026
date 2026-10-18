#!/usr/bin/env bash
# Runs code/src/v6_france_probe.ipynb with every temp/runtime dir on D:
cd /d/AmazonMLChallenge2026
export TEMP='D:\AmazonMLChallenge2026\work\tmp' TMP='D:\AmazonMLChallenge2026\work\tmp'
export JUPYTER_RUNTIME_DIR='D:\AmazonMLChallenge2026\work\jupyter\runtime' JUPYTER_DATA_DIR='D:\AmazonMLChallenge2026\work\jupyter\data'
export JUPYTER_CONFIG_DIR='D:\AmazonMLChallenge2026\work\jupyter\config' IPYTHONDIR='D:\AmazonMLChallenge2026\work\jupyter\ipython'
export PYTHONPYCACHEPREFIX='D:\AmazonMLChallenge2026\work\pycache' MPLCONFIGDIR='D:\AmazonMLChallenge2026\work\jupyter\mpl'
.venv/Scripts/python -m papermill code/src/v8_france_ablation.ipynb work/v8abl/v8abl_executed.ipynb \
    --log-output --no-progress-bar --kernel python3 --cwd code/src > work/v8abl/run.log 2>&1
echo "exit=$?" >> work/v8abl/run.log
# post-step: descriptor-swap filter (T21) on each stage variant
cd /d/AmazonMLChallenge2026/code/experiments
for S in A B C; do
  PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python exp_v6_descswap.py 300 "D:\AmazonMLChallenge2026\output\lb_probe\matching_results_v8abl_${S}_raw.tsv" "D:\AmazonMLChallenge2026\output\lb_probe\matching_results_v8abl_${S}.tsv" >> /d/AmazonMLChallenge2026/work/v8abl/run.log 2>&1
done
echo "post=$?" >> /d/AmazonMLChallenge2026/work/v8abl/run.log
