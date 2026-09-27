# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** Sumit Singatia [add the other team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We use a **blocking-plus-cascade** approach. For each Source-1 record, multi-channel GPU kNN over TF-IDF → SVD embeddings, together with 9 language-independent exact keys, generates candidates. A learned prefilter keeps the best 80 per S1 record.

Pairs are then scored by a cascade of models:
- a gradient-boosted matcher with about 96 features;
- a group-context model that looks at each S1's whole candidate set;
- a Ditto-style **multilingual cross-encoder**, fine-tuned on 3.55M hard pairs;
- a cross-fitted "undecided-zone" re-stacker.

Decisions are made per S1 entity with one owner per S2/S3 record, at a threshold optimized for macro F0.5. The final result is **0.977 on the leaderboard** and **0.9846 on validation** (0.9848 on a held-out half).

---

## 2. Methodology

### 2.1 Problem Analysis

- **Cluster structure.** S1 entities have 0–11 matches: 3.46 on average, and 5.6% have none. US and India have identical truth profiles. 94% of clusters have 2 or more records, and 85% contain records from both S2 and S3.
- **Noise.** We see:
  - address abbreviations (R/Rue, Rd/Road);
  - reordered address components;
  - missing postcodes and states;
  - landmark-based addresses ("near SBI ATM");
  - Devanagari and other native-script names (transliteration);
  - legal-form changes (Pvt Ltd / LLC / SARL);
  - social handles (`@ehpadpierre`);
  - typos;
  - blank addresses (about 3% of S2/S3 records).
- **Distractors.** Many S2/S3 records are *unmatched*: 26% in train and about 42% in test. Different S1 entities often share the **same name** at different addresses. For a blank-address record, the true rate falls to roughly 1/k when k S1 entities share its name. This is the main **irreducible** error.
- **France (unseen country).** French names are generated from templates such as "{City} {Word} {LegalForm}" and "EHPAD {Name}". As a result, name + house-number collisions are common. One source writes the région ("Hauts-de-France") where the other writes the département ("Nord"). Character-level features transfer poorly. The main v3 → v4 leaderboard gap (0.965 vs 0.981 on validation) was a French blocking failure.
- **No leaks.** Row order and ID numbers carry no cluster information; we checked the correlations.

### 2.2 Solution Strategy

**Approach Type:** Hybrid: blocking, a learned prefilter, a GBDT cascade, a cross-encoder re-ranker, stacking, and constrained resolution.

**Core Innovation:**
1. **Language-independent exact keys**, such as full-name + house-number and house-number + street. These fixed French blocking without any French labels (+0.008 on the leaderboard).
2. A **data-rich multilingual cross-encoder** that re-scores a wide probability band, plus a cross-fitted re-stacker for the undecided zone (+0.002 on the leaderboard).
3. A strict **validation protocol**: a fixed 100k-entity split, thresholds tuned on half A and reported on half B. We also built a label-free France benchmark and a controlled leaderboard probe for every France-only change.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - **Normalization:** an anyascii fold plus a transliteration lexicon learned from the ground-truth alignments (native script ↔ Latin). It also handles legal-form and abbreviation normalization and extracts house number, postcode and street.
  - **Dense channels:**
    - character TF-IDF → 192-d SVD for name (2–4 grams) and address (3–4 grams);
    - exact GPU kNN per country and source, with channels name (k=15), address (k=20) and name+address (k=30);
    - a name channel restricted to blank-address records (k=5).
  - **9 exact keys:** squashed name, postcode + first token, first token + house number, first two tokens, company number, house number + next token, blank-address squashed name, full name + house number, house number + first street word.
  - **Prefilter:** an XGBoost on the rank, cosine and key features keeps the **top 80 per S1**.
- **Candidate pairs generated:** 1,732,544 S1 × 80 = **about 138.6M pairs** (`candidate_pairs.tsv`).
- **How we ensured true matches were not lost:**
  - Recall is measured on validation after every change. Candidate recall is **98.0%**, and a perfect matcher on these candidates would score 0.9939.
  - The top-80 cut was chosen on validation: top-40 gave 97.7%, top-80 gave 98.0%, and hold-out F0.5 went +0.0008.
  - The language-independent keys raised French coverage: S1 entities with no match fell from 9.7% to 6.2%.

---

## 4. Matching Model

**Features used:**
- **Name features:** rapidfuzz ratio, token-set, token-sort and partial ratio; Jaro-Winkler on squashed names; containment; first-token and suffix equality; lengths; native-script flag; IDF-weighted token Jaccard and "only-in-q / only-in-c" token weights; difference from the best in the candidate set.
- **Address features:** ratio, token-set and token-sort ratio; postcode, house-number and company-number equality; house-number edit distance, containment and gap; shared and unshared numbers; street-token overlap; length ratio; address crowding (how many records share the address).
- **Other features:** kNN ranks and cosines per channel with their relative values; exact-key flags; prefilter score and rank; candidate-set size; rarity of name and address tokens within the candidate set; S1-side ambiguity counts (how many S1 records share this name or address) and pool counts.
- **Group context (stage 2):** rank in the group, gap and margin to the best, counts above 0.5 and 0.8, and cross-source agreement with the strongest S2 and S3 siblings.
- **Cross-encoder:** records are serialized as `[COL] name [VAL] … [COL] addr [VAL] …`, with house numbers wrapped in `[NUM] … [/NUM]`.

**Model type:**
- **Stage-1 matcher:** XGBoost (GPU), with 2-fold out-of-fold predictions.
- **Group-context model:** XGBoost stacked on stage 1.
- **Cross-encoder:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0). It was fine-tuned on 3.55M pairs from 400k training entities: positives, hard negatives and sampled easy negatives, with Ditto attribute and token deletion. Its band AUC is 0.957.
- **Blend:** a logistic blend of the group-model and cross-encoder logits on the band 0.001–0.999.
- **Re-stacker:** an XGBoost re-stacker on the undecided zone (probability 0.01–0.99), cross-fitted on validation halves.

**Threshold selection method:**
- The threshold (0.75) is chosen by F0.5 optimization on validation half A and reported on half B.
- **Resolution:** each S2/S3 record goes to its best S1 only, with caps of 5 S2 and 6 S3 matches per S1.
- **France:** a filter removes same-address pairs whose names differ only by frequent descriptor words (a "descriptor swap"), which gave +0.001 on the leaderboard.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**

  | Version | Validation | Leaderboard |
  |---|---|---|
  | v11 (final) | **0.9846** (hold-out half B 0.9848) | **0.977** |
  | v10 | 0.9836 | 0.975 |
  | v4 | 0.9826 | 0.973 |
  | v1 baseline | 0.887 | – |

  Final validation per country: India 0.9820, US 0.9863.
- **Layer contributions (validation):**
  - stage-1 alone: 0.9764;
  - + group model: 0.9789;
  - + cross-encoder blend: 0.9839;
  - + re-stacker: 0.9848.
- **Common false positives (wrong merges):** They cost about 0.002, and there are three kinds:
  - same-name businesses at the same street but a different house number;
  - chain or franchise names;
  - French templated names at the same address with a different descriptor, such as "NVG Ecole" vs "NVG EHPAD".
- **Common false negatives (missed matches):**
  - **Rejected pairs (about 0.0075):** 70% are blank-address candidates whose name also belongs to another S1 entity. In 97.6% of false cases the record truly belongs to a *different* S1 with the same name, so this is largely irreducible.
  - **Blocking misses (about 0.006):** names changed beyond character similarity (transliteration, abbreviation, reordering) together with blank addresses.
- **Negative results we logged:**
  - self-training on France (−0.001, confirmation bias);
  - France-only IDF (−0.002);
  - French address normalization (tie);
  - sibling-evidence features (+0.0001).

---

## 6. Conclusion

A carefully validated cascade reached 0.977 on the leaderboard (baseline 0.887 on validation): recall-oriented blocking with language-independent keys, GBDT matchers with rich features, and a strong multilingual cross-encoder. We learned three lessons:
1. Robust gains came from *structural* fixes such as blocking keys and a better pair model, not from label-free threshold tweaks for the unseen country.
2. Every recall gain paid off only together with a strong re-ranker.
3. The next lever is learned blocking. A contrastive multilingual bi-encoder channel raised candidate recall on validation from 98.0% to 99.4% with 4 extra candidates per S1, and the best possible score from 0.9939 to 0.9983. It was validated but not integrated before the deadline.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains:
- `README.md`: the exact run order;
- `requirements.txt`: pinned from the working venv;
- `src/notebooks/`:
  - `improved.ipynb`: shared caches and the validation split;
  - `v4.ipynb`: lexicon, blocking, prefilter, GBDTs and France output;
  - `v11_full.ipynb`: the final validation and test inference;
- `src/scripts/`:
  - cross-encoder training: `exp_v9_mce.py`, then `exp_v11_mce.py`;
  - the French descriptor-swap filter: `exp_v6_descswap.py`;
  - the France merge: `merge_fr.py`;
  - shared validation scoring: `valscore.py`;
- `src/notebook_builders/`: the notebook sources and patch builders.

**Entry points:**
1. `improved.ipynb`
2. `v4.ipynb`
3. `exp_v6_descswap.py 300`
4. `exp_v9_mce.py`
5. `exp_v11_mce.py`
6. `v11_full.ipynb`
7. `merge_fr.py`, then the validator

### B. Additional Results

| Experiment | Result (validation hold-out B / leaderboard) |
|---|---|
| Prefilter cut 40 → 80 | 0.9828 → 0.9836; leaderboard 0.975 |
| Multilingual vs English cross-encoder | 0.9826 → 0.9828 |
| Data-rich cross-encoder (3.55M pairs) | band AUC 0.955 → 0.957; 0.9836 → 0.9838 (0.9839 with the wide band) |
| Zone re-stacker | +0.0009 on top of the wide-band cross-encoder → 0.9848; leaderboard 0.977 |
| Learned blocking (bi-encoder top-10; not in the submission) | candidate recall 98.0% → 99.4%, oracle ceiling 0.9939 → 0.9983 |
