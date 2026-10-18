#!/usr/bin/env bash
# Runs code/src/v6_france_probe.ipynb with every temp/runtime dir on D:
cd /d/AmazonMLChallenge2026
export TEMP='D:\AmazonMLChallenge2026\work\tmp' TMP='D:\AmazonMLChallenge2026\work\tmp'
export JUPYTER_RUNTIME_DIR='D:\AmazonMLChallenge2026\work\jupyter\runtime' JUPYTER_DATA_DIR='D:\AmazonMLChallenge2026\work\jupyter\data'
export JUPYTER_CONFIG_DIR='D:\AmazonMLChallenge2026\work\jupyter\config' IPYTHONDIR='D:\AmazonMLChallenge2026\work\jupyter\ipython'
export PYTHONPYCACHEPREFIX='D:\AmazonMLChallenge2026\work\pycache' MPLCONFIGDIR='D:\AmazonMLChallenge2026\work\jupyter\mpl'
.venv/Scripts/python -m papermill code/src/v10_full.ipynb work/v10full/v10full_executed.ipynb \
    --log-output --no-progress-bar --kernel python3 --cwd code/src > work/v10full/run.log 2>&1
echo "exit=$?" >> work/v10full/run.log
cd /d/AmazonMLChallenge2026/code/experiments
export PYTHONIOENCODING=utf-8
../../.venv/Scripts/python exp_v6_descswap.py 300 'D:\AmazonMLChallenge2026\output\lb_probe\matching_results_v10_raw.tsv' 'D:\AmazonMLChallenge2026\output\matching_results_v10_frnew.tsv' >> /d/AmazonMLChallenge2026/work/v10full/run.log 2>&1
../../.venv/Scripts/python merge_v10.py >> /d/AmazonMLChallenge2026/work/v10full/run.log 2>&1
cd /d/AmazonMLChallenge2026
for f in matching_results_v10.tsv matching_results_v10_frnew.tsv; do .venv/Scripts/python utils/validate_submission.py --matching output/$f --candidate output/lb_probe/candidate_pairs_v10.tsv --test-dir dataset/test --check-ids 2>&1 | tail -1 | sed "s/^/$f: /" >> work/v10full/run.log; done
echo "post=done" >> work/v10full/run.log
