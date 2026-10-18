#!/usr/bin/env bash
# Runs code/src/v11_full.ipynb with every temp/runtime dir on D:, then the France swap filter, the France merge and the validator
cd /d/AmazonMLChallenge2026
mkdir -p work/v11full
export TEMP='D:\AmazonMLChallenge2026\work\tmp' TMP='D:\AmazonMLChallenge2026\work\tmp'
export JUPYTER_RUNTIME_DIR='D:\AmazonMLChallenge2026\work\jupyter\runtime' JUPYTER_DATA_DIR='D:\AmazonMLChallenge2026\work\jupyter\data'
export JUPYTER_CONFIG_DIR='D:\AmazonMLChallenge2026\work\jupyter\config' IPYTHONDIR='D:\AmazonMLChallenge2026\work\jupyter\ipython'
export PYTHONPYCACHEPREFIX='D:\AmazonMLChallenge2026\work\pycache' MPLCONFIGDIR='D:\AmazonMLChallenge2026\work\jupyter\mpl'
.venv/Scripts/python -m papermill code/src/v11_full.ipynb work/v11full/v11full_executed.ipynb \
    --log-output --no-progress-bar --kernel python3 --cwd code/src >> work/v11full/run.log 2>&1
echo "exit=$?" >> work/v11full/run.log
cd /d/AmazonMLChallenge2026/code/experiments
export PYTHONIOENCODING=utf-8
../../.venv/Scripts/python exp_v6_descswap.py 300 'D:\AmazonMLChallenge2026\output\lb_probe\matching_results_v11_raw.tsv' 'D:\AmazonMLChallenge2026\output\matching_results_v11_frnew.tsv' >> /d/AmazonMLChallenge2026/work/v11full/run.log 2>&1
../../.venv/Scripts/python merge_fr.py 'D:\AmazonMLChallenge2026\output\lb_probe\matching_results_v11_raw.tsv' 'D:\AmazonMLChallenge2026\output\matching_results_v11.tsv' >> /d/AmazonMLChallenge2026/work/v11full/run.log 2>&1
cd /d/AmazonMLChallenge2026
for f in matching_results_v11.tsv matching_results_v11_frnew.tsv; do .venv/Scripts/python utils/validate_submission.py --matching output/$f --candidate output/lb_probe/candidate_pairs_v11.tsv --test-dir dataset/test --check-ids 2>&1 | tail -1 | sed "s/^/$f: /" >> work/v11full/run.log; done
echo "post=done" >> work/v11full/run.log
