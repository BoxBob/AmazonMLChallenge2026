#!/usr/bin/env bash
# Runs code/src/baseline.ipynb with every temp/runtime dir on D:
cd /d/AmazonMLChallenge2026
export TEMP='D:\AmazonMLChallenge2026\work\tmp' TMP='D:\AmazonMLChallenge2026\work\tmp'
export JUPYTER_RUNTIME_DIR='D:\AmazonMLChallenge2026\work\jupyter\runtime' JUPYTER_DATA_DIR='D:\AmazonMLChallenge2026\work\jupyter\data'
export JUPYTER_CONFIG_DIR='D:\AmazonMLChallenge2026\work\jupyter\config' IPYTHONDIR='D:\AmazonMLChallenge2026\work\jupyter\ipython'
export PYTHONPYCACHEPREFIX='D:\AmazonMLChallenge2026\work\pycache' MPLCONFIGDIR='D:\AmazonMLChallenge2026\work\jupyter\mpl'
.venv/Scripts/python -m papermill code/src/baseline.ipynb work/baseline/baseline_executed.ipynb \
    --log-output --no-progress-bar --kernel python3 --cwd code/src > work/baseline/run.log 2>&1
echo "exit=$?" >> work/baseline/run.log
