# Business Entity Resolution — Baseline

The whole pipeline is a single notebook: `src/baseline.ipynb`. Running it end to end goes from the raw
TSV files to both submission files, `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

## Hardware used
- NVIDIA GeForce RTX 3050 Laptop GPU (6 GB), CUDA 12.8 driver. **A CUDA GPU is required**:
  embeddings, blocking (kNN) and XGBoost all run on the GPU.
- 24 GB RAM, 8 CPU threads. About 25 GB of free disk for caches.
- Runtime is about 65 minutes for a full run on this machine.

## Setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
```
(On Linux use `.venv/bin/python`.)

## Expected layout
`ER_ROOT` is the challenge folder. The notebook reads and writes these paths under it:
```
<ER_ROOT>/
├── dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
├── dataset/test/test_source{1,2,3}.tsv
├── utils/validate_submission.py
├── output/            <- written: candidate_pairs.tsv, matching_results.tsv
└── work/baseline/     <- written: caches (normalized records, embeddings, features, model)
```

## Run
```bash
export ER_ROOT=/path/to/challenge_folder        # Windows PowerShell: $env:ER_ROOT = "D:\..."
.venv/Scripts/python -m papermill src/baseline.ipynb baseline_executed.ipynb --log-output
```
You can also open `src/baseline.ipynb` in Jupyter or VS Code and use **Run All**. The last cell runs
`utils/validate_submission.py` on the two output files.

Caches under `work/baseline/` let a second run skip normalization, embeddings and training-pair
features. Delete that folder to rebuild everything from scratch.

## Notebook sections
1. Load and normalize: transliteration (anyascii), legal-suffix and address-abbreviation canonicalisation, postcode and house-number extraction.
2. Name and address embeddings: char n-gram TF-IDF, then truncated SVD (192 dimensions) on the GPU.
3. Blocking: per-country GPU top-k cosine search by name and by address against S2 and S3, plus exact-key joins.
4. Ground truth, and a train/validation split of S1 entities (300k / 100k).
5. Pair features: rapidfuzz similarities, embedding cosines, postcode and house-number agreement, rank features.
6. and 7. XGBoost pair classifier (`device="cuda"`) with early stopping.
8. Decision rule (one S1 per S2/S3 record, per-source caps) and threshold tuning for macro F0.5.
9. Test inference, writing the output files, and validation.

Seed: 42 (sampling and XGBoost).
