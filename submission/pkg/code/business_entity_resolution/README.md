# Business Entity Resolution: reproduction guide

This folder regenerates the final submission (leaderboard 0.977):
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

## Pipeline overview

1. **Normalization.** Records are normalized with an anyascii fold, a transliteration lexicon learned from the training ground truth, and abbreviation and legal-form handling.
2. **Blocking.** Candidates come from two sources:
   - a GPU kNN over TF-IDF → SVD embeddings, with three channels (name, address, combined) plus a blank-address name channel;
   - 9 language-independent exact keys, such as full name + house number and house number + street.
3. **Prefilter.** An XGBoost prefilter keeps the **top 80 candidates per S1** (`candidate_pairs.tsv`).
4. **Pair scoring.** Three models score each pair:
   - a stage-1 XGBoost with about 96 features: fuzzy name and address similarities, numeric and house-number features, IDF token overlap, candidate-set rarity, and S1-side ambiguity counts;
   - a group-context XGBoost, stacked on out-of-fold stage-1 probabilities, with cross-source sibling features;
   - a Ditto-style multilingual cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, Apache-2.0). It is fine-tuned on 3.55M hard pairs and re-scores the probability band 0.001–0.999. A logistic blend combines it with the group model.
5. **Zone re-stacker.** A small XGBoost re-scores pairs with blend probability between 0.01 and 0.99. It is cross-fitted on validation halves.
6. **Resolution.** Pairs are kept above a threshold tuned for F0.5 on validation (0.75). Each S2/S3 record is assigned to one S1 only, with at most 5 S2 and 6 S3 matches per S1.
7. **France.** France is unseen in training, so French rows come from the v4 model plus a descriptor-swap filter (`exp_v6_descswap.py`). Re-scoring France with the final model tied on the leaderboard.

## Environment

- Python 3.14 in a venv, with `pip install -r requirements.txt`.
- An NVIDIA GPU with CUDA. It was developed on an RTX 3050 6 GB laptop GPU with 24 GB RAM.
- About 60 GB of free disk space for caches.

**Paths.** Set `ER_ROOT` to a project root that contains `dataset/train`, `dataset/test` and `utils/`. The default is `D:\AmazonMLChallenge2026`. The standalone scripts in `src/scripts/` use the same root, hard-coded near their top; edit it if your root differs. All caches go to `<root>/work/`.

## Run order

Run the notebooks with papermill or Jupyter, with `code/src` as the working directory.

| # | Step | Produces |
|---|---|---|
| 0 | `src/notebooks/improved.ipynb` | Shared caches in `work/v2/`: normalized records, TF-IDF/SVD embeddings, kNN arrays, and the fixed 100k validation split used by `valscore.py`. |
| 1 | `src/notebooks/v4.ipynb` | Transliteration lexicon, prefilter, stage-1 and group models (`work/v4/`), the English cross-encoder (`work/v3/ce_model`), and `output/matching_results_v4.tsv`. |
| 2 | `python src/scripts/exp_v6_descswap.py 300` | `output/lb_probe/matching_results_fr_noswap_f300.tsv`, the source of the French rows. |
| 3 | `python src/scripts/exp_v9_mce.py` | `work/v9/mce_model`, the multilingual cross-encoder fine-tuned on 1.2M pairs. |
| 4 | `python src/scripts/exp_v11_mce.py` | `work/v11/mce_model`, a warm start from step 3 trained on 3.55M hard pairs with Ditto augmentation (about 3.7 h). The model is saved **before** the evaluation block at the end, which needs the v10 validation cache and can be skipped. |
| 5 | `src/notebooks/v11_full.ipynb` | Validation (0.9846, hold-out 0.9848), then streaming test inference with checkpointed 25k-S1 chunks. Writes `output/lb_probe/matching_results_v11_raw.tsv` and `output/lb_probe/candidate_pairs_v11.tsv`. |
| 6 | `python src/scripts/merge_fr.py <root>/output/lb_probe/matching_results_v11_raw.tsv <root>/output/matching_results.tsv` | Final matches: US/India from step 5, France from step 2. |
| 7 | `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids` | PASS |

Then copy `output/lb_probe/candidate_pairs_v11.tsv` to `output/candidate_pairs.tsv`.

`src/notebook_builders/` shows how the notebooks were generated:
- `*_src.py` are the notebook sources, converted with `src/scripts/py2nb.py`.
- `make_v10m.py → make_v10full.py → make_v11full.py` are the patch builders. The v11 notebook was built with `V11_BAND="0.001, 0.999"`.

## Models and licenses

- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`: Apache-2.0.
- `cross-encoder/ms-marco-MiniLM-L12-v2`: Apache-2.0.
- XGBoost, rapidfuzz, anyascii, scikit-learn and PyTorch: open-source licenses.
- No external data. No non-commercial models.
