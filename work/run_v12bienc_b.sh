#!/usr/bin/env bash
cd /d/AmazonMLChallenge2026/code/experiments
export TEMP='D:\AmazonMLChallenge2026\work\tmp' TMP='D:\AmazonMLChallenge2026\work\tmp' HF_HOME='D:\AmazonMLChallenge2026\work\hf'
export PYTHONPYCACHEPREFIX='D:\AmazonMLChallenge2026\work\pycache' PYTHONIOENCODING=utf-8 V12_TAG=b V12_TRAIN_ONLY=1
../../.venv/Scripts/python exp_v12_biencoder.py > /d/AmazonMLChallenge2026/work/v12/bienc_b.log 2>&1
echo "exit=$?" >> /d/AmazonMLChallenge2026/work/v12/bienc_b.log
