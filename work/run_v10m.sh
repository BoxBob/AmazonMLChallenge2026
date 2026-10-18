#!/usr/bin/env bash
cd /d/AmazonMLChallenge2026
export TEMP='D:\AmazonMLChallenge2026\work\tmp' TMP='D:\AmazonMLChallenge2026\work\tmp' HF_HOME='D:\AmazonMLChallenge2026\work\hf'
export JUPYTER_RUNTIME_DIR='D:\AmazonMLChallenge2026\work\jupyter\runtime' JUPYTER_DATA_DIR='D:\AmazonMLChallenge2026\work\jupyter\data'
export JUPYTER_CONFIG_DIR='D:\AmazonMLChallenge2026\work\jupyter\config' IPYTHONDIR='D:\AmazonMLChallenge2026\work\jupyter\ipython'
export PYTHONPYCACHEPREFIX='D:\AmazonMLChallenge2026\work\pycache' PF_M_OVERRIDE=${1:-80}
mkdir -p work/v10m$PF_M_OVERRIDE
.venv/Scripts/python -m papermill code/src/v10_prefilter_m.ipynb work/v10m$PF_M_OVERRIDE/executed.ipynb \
    --log-output --no-progress-bar --kernel python3 --cwd code/src > work/v10m$PF_M_OVERRIDE/run.log 2>&1
echo "exit=$?" >> work/v10m$PF_M_OVERRIDE/run.log
