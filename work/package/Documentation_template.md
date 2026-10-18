# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
This is a two-stage pipeline: blocking, then a classifier. For blocking, names and addresses are embedded with character n-gram TF-IDF reduced by SVD. A GPU top-k cosine search, run separately per country, pulls about 40 S2/S3 candidates for each Source-1 entity, and exact-key joins add more. A GPU XGBoost model then scores each candidate pair using string-similarity and field-agreement features. Final matches come from a precision-oriented rule: each S2/S3 record goes to one S1 entity only, and the threshold is tuned for macro F0.5. **Validation macro F0.5 = 0.8871.**

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on `train_ground_truth.tsv` (2,206,821 S1 entities, 7,638,365 true pairs):
- **Singletons:** 5.58% of S1 entities have no true match. The rest have 1–11 matches (0–5 from S2, 0–6 from S3), usually 2–6.
- **Each matched S2/S3 record belongs to exactly one S1 entity.** None is shared, so the clusters don't overlap. We use this as a constraint when deciding matches.
- **Every true pair has the same country (100%).** Blocking is therefore done within each country.
- **About 26% of S2 and S3 records match nothing.** They are distractors.
- **Missing fields:** names are never blank. Addresses are blank in 2.9–3.7% of S2/S3 records.
- **Postcodes:** a 5–6 digit number (a likely postcode) appears in about 11–12% of US addresses and about 1% of India addresses.
- **Noise:**
  - Name typos (`Wilblims` vs `Williams`).
  - Trade/DBA names that share nothing with the legal name (`Dréxkor` for `Maure Williams Colombier Inc`). These can only be matched through the address.
  - Web domains used as names (`maurewilliamscolombier.com`).
  - Native-script names (Tamil or Devanagari) for India records.
  - Reordered address components, and misspelled towns (`Ticonderoga Townshiip`).
- The test set has a third country, **France**, with no training data. Nothing in the pipeline is specific to a country.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (GPU-accelerated).  
**Core Innovation:**
- Two separate blocking channels, one on the name and one on the address. This catches trade-name matches whose names are unrelated.
- A unicode-folding and transliteration normalizer that makes native-script names comparable with Latin ones.
- A decision rule matched to the metric: one owner per record, per-source caps, and a threshold tuned directly on per-entity F0.5.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. **Name embedding kNN.** Character 2–4-gram TF-IDF on the name with its legal form removed, reduced to 192 dimensions by randomized SVD on the GPU (`torch.svd_lowrank`). Top-10 by cosine against S2 and top-10 against S3, within the same country.
  2. **Address embedding kNN.** The same method with character 3–4-grams on the normalized address. Top-10 against S2 and top-10 against S3.
  3. **Exact key: squashed name.** Country plus the core name with spaces removed. This catches domain-style names like `maurewilliamscolombier`.
  4. **Exact key: postcode plus first name token**, within the country.
  - Exact-key blocks with more than 50 records are skipped as too generic. The union of all channels is removed of duplicates, and the rank from each kNN list is kept as a feature.
- **Candidate pairs generated:** 68,794,909 on test (39.7 per S1 entity). On the 100k validation entities: 3,970,351.
- **How you ensured true matches were not lost:**
  - The name and address channels are independent, so a record that fails one can still be found by the other.
  - The TF-IDF vocabulary and SVD basis are fitted on text sampled from both train and test (no labels), so French n-grams are represented.
  - Recall was measured on validation: **83.6% of true pairs** are in the candidate set. That caps achievable macro F0.5 at **0.928**. Blocking recall is the main area for future improvement.

---

## 4. Matching Model

**Features used (30):**
- **Name:**
  - rapidfuzz `ratio`, `token_set_ratio`, `token_sort_ratio` and `partial_ratio` on the core name.
  - Jaro-Winkler on the squashed name, and `token_set_ratio` on the full name including the legal form.
  - Whether one squashed name contains the other, whether the first token matches, and whether the legal suffixes agree (1 = same, 0 = different, −1 = missing).
  - Name lengths and the cosine similarity of the name embeddings.
- **Address:**
  - rapidfuzz `ratio`, `token_set_ratio` and `token_sort_ratio`.
  - Whether the postcodes agree and whether the house numbers agree (1 = same, 0 = different, −1 = missing).
  - Blank-address flags for both records, and the cosine similarity of the address embeddings.
- **Other:**
  - The candidate's rank in the name and address kNN lists, and the exact-key flags.
  - The candidate's source (S2 or S3) and the number of candidates for that S1 entity.
  - "Relative-to-best" features: each score minus the best score among that S1 entity's candidates, for name cosine, address cosine, name token-set and address token-set.
- Country is deliberately **not** a feature, so the model generalizes to France.

Normalization before computing features:
- NFKC-style folding and transliteration (`anyascii`), lowercasing, `&` → `and`, removal of `www.` and web domain endings (`.com`, etc.).
- Canonical legal forms (corporation → corp, private → pvt, limited → ltd, SARL/SAS/SCI, ...), stripped into a separate suffix field.
- Canonical address abbreviations (street → st, road → rd, avenue → ave, rue, boulevard, ...).
- US and Indian state names mapped to their codes.
- Postcode and house-number extraction.

**Model type:** XGBoost binary classifier trained on the GPU.
- Settings: `device="cuda"`, `hist`, depth 8, learning rate 0.08, 80% row and column subsampling, early stopping on validation AUC-PR (best iteration 1626).
- Training data: 11.9M candidate pairs from 300k S1 entities (7.3% positive).

**Threshold selection method:**
1. Keep pairs with probability at or above the threshold.
2. Assign each S2/S3 record only to the S1 entity it scores highest with, because the ground truth never assigns a record to two S1s.
3. Keep at most 5 S2 and 6 S3 matches per S1 entity.
4. The threshold is chosen by grid search (0.20–0.95) to maximize the official **macro F0.5** on the 100k held-out S1 entities, singletons included. Best threshold: **0.625**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.8871** on validation (100,000 held-out S1 entities; the blocking ceiling is 0.9275).
- On validation, at the chosen threshold, 7,885 predicted pairs were false positives, out of 289,769 true pairs in the candidate set. 15,310 true candidate pairs were missed by the classifier. Another 56,693 true pairs were never generated by blocking.
- **Common false positives (wrong merges):**
  - Different businesses at the same address, such as `city of appleton dental health` and `city of appleton behavioral health` at 2112 Lucille St. The same happens in shared Indian commercial buildings.
  - Candidates with the same name but a blank or different address (`naranjo group` with no address). Name-only evidence is weak when names are common.
  - Near-identical names at neighbouring street numbers (`rocky quality mines`, 2102 vs 2105 71st Ave).
- **Common false negatives (missed matches):**
  - Trade/DBA names with the same address but an unrelated name (`adorne choate dynamix` and `iriciranex`), which the model scores at about 0.47.
  - Candidates with a blank address, where the name alone doesn't clear the high threshold (`nns hospital`, `nine services`).
  - Heavy typos in both fields at once (`clyic cleaning` / `c1yic cleaning` with a different town).
  - True matches that blocking never retrieves (16.4% of true pairs). These are mostly records where both the name and the address changed substantially.

---

## 6. Conclusion
A GPU blocking-plus-classifier pipeline with country-agnostic normalization scores 0.887 macro F0.5 on validation. Nearly all of that comes from well-calibrated, precision-first decisions over a candidate set with 83.6% recall. The main lesson is that blocking recall is now the bottleneck: the classifier comes within 0.04 of the blocking ceiling. Planned next steps: higher-recall blocking (larger k, a postcode-only channel, better multilingual name embeddings), and features that separate different businesses sharing an address.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/baseline.ipynb` is the single entry point. It runs every stage in order: normalization, embeddings, blocking, features, XGBoost training, threshold tuning and test inference. It writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`, then runs the official validator.
- `code/business_entity_resolution/README.md` has the setup and exact run commands (`papermill` or Run All).
- `code/business_entity_resolution/requirements.txt` lists pinned library versions (PyTorch comes from the CUDA 12.8 wheel index).
- Licences: every library and model used is open source. XGBoost is Apache-2.0, and PyTorch, scikit-learn, rapidfuzz and pyarrow use permissive licences. No pretrained language model and no external data or lookups are used.

### B. Additional Results
| Stage | Metric | Value |
|---|---|---|
| Blocking (validation) | true pairs retrieved | 289,769 / 346,462 (83.6%) |
| Blocking (validation) | candidates per S1 | 39.7 |
| Blocking (validation) | macro F0.5 of a perfect classifier | 0.9275 |
| Classifier (validation) | macro F0.5 | **0.8871** @ threshold 0.625 |
| Test | predicted matches | 4,767,828 (1,565,860 of 1,732,544 S1 entities have ≥1) |

Top features by XGBoost gain: `addr_blank_c`, `a_tset`, `exact_pc_first`, `hn_eq`, `exact_squash`, `rel_a_tset`, `n_partial`, `pc_eq`.
