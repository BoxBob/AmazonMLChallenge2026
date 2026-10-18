#!/usr/bin/env bash
# waits for bi-encoder "b" training, then embeds all train/test records and builds the bi kNN channel
cd /d/AmazonMLChallenge2026
until grep -q "^exit=" work/v12/bienc_b.log; do sleep 60; done
cd code/experiments
export TEMP='D:\AmazonMLChallenge2026\work\tmp' TMP='D:\AmazonMLChallenge2026\work\tmp' HF_HOME='D:\AmazonMLChallenge2026\work\hf'
export PYTHONPYCACHEPREFIX='D:\AmazonMLChallenge2026\work\pycache' PYTHONIOENCODING=utf-8
../../.venv/Scripts/python exp_v12_embed.py >> /d/AmazonMLChallenge2026/work/v12/embed.log 2>&1
echo "exit=$?" >> /d/AmazonMLChallenge2026/work/v12/embed.log
