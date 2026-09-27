# Next plan: from 0.974 to the top-100 level (0.991)

Written after v10 (validation 0.9836 on US/India, leaderboard 0.974, rank ~1000; top-100 at 0.991).

## 1. Where the 0.017 gap is

| Slice | Our level | Needed for ~0.991 | Share of the test | Gap contribution |
|---|---|---|---|---|
| US + India | ~0.9836 (validation, transfers per silver checks) | ~0.992 | 85% | **~0.007** |
| France | ~0.93 (implied from leaderboard) | ~0.985 | 15% | **~0.008** |

Both halves have to improve. Tuning France alone can reach at most about half of the gap.

## 2. What the diagnostics say (these change the strategy)

1. **No leaks.** Row order and ID numbers carry no cluster information (correlation −0.0001 and +0.0029; same-cluster
   records never adjacent). The top teams are winning on modelling, not artifacts.
2. **Our "information limit" was only a *pairwise* limit.** T18 concluded that the remaining rejected true pairs were
   inherently ambiguous, but that looked at one pair at a time. On the labelled data:
   * 94% of S1 clusters have ≥2 records; 85% have records in both S2 and S3;
   * **97.9% of blank-address true matches have an *addressed* sibling in the same cluster**, so the evidence for our
     largest "ambiguous" bucket exists, just not in the pair;
   * S2↔S3 names *within* a cluster are less similar (median 70) than S1↔S2/S3 (92), so naive S2–S3 linking won't
     work; it needs a strong pair model.
3. **Our strongest component is weak and data-starved.** The cross-encoder alone scores *below* the GBDT (band AUC 0.948),
   was trained on 1.2M pairs from 150k entities (7% of the training entities) for one epoch, and only sees the 3% "uncertain band".
   Every time it improved (v3 +0.0034, multilingual +0.0002, recall +0.0008 via re-ranking), the whole system improved.
4. **Blocking:** 2% of true pairs are never candidates. 63% of those rank ≥100 in the SVD channels (59% of them have a blank
   address). More k can't reach them; a *better representation* can.
5. **France:** label-free patches have hit leaderboard resolution (±0.0005 per change). France needs *structural*
   improvements that also help the labelled countries: a better pair model, collective decisions, learned blocking.

## 3. New approaches, ranked by expected value

Each keeps what already works (normalization + transliteration lexicon, exact keys, GBDT features, group model,
France swap filter, FES, validation protocol, version control) and targets a measured loss.

### A. A strong, data-rich pair model (Ditto; Peeters & Bizer 2023–24; AnyMatch 2024)
* **What:** re-train the multilingual cross-encoder on **much more and harder data**:
  - all 400k training entities, top-16 candidates each (≈6M pairs, hard negatives from our own candidate sets), 2 epochs;
  - **Ditto augmentation**: span deletion or shuffling, and *attribute deletion* (blank the address, which is our largest error class);
  - AnyMatch-style difficulty-aware sampling (over-sample pairs the GBDT gets wrong).
  - Then apply it to a **wide band** (p 0.001–0.999, not 0.01–0.99) and feed it into the group model as a feature, not only a late blend.
* **Target:** rejected-true 0.0075 and false-merges 0.0024 (US/India), and France (a better reader of French text).
* **Cost on our hardware:** ~6 h per epoch for mMiniLM on the RTX 3050. A larger base (mDeBERTa-v3-base, MIT, 278M) needs a
  cloud GPU (Kaggle/Colab T4/P100, free tiers). **Check the competition terms before moving data off the machine.**
* **Expected:** +0.002 … +0.004 validation. **Gate:** band AUC ≥ 0.96 and hold-out ≥ 0.9850.

### B. Collective (cluster-level) decisions (Saeedi et al. FAMER; TransClean 2025; ComEM "Select", COLING 2025)
* **What:** stop deciding S1→S2/S3 pairs independently.
  1. Score **S2↔S3 edges** among each entity's top candidates with the pair model from A.
  2. Per entity, pick the **most consistent cluster**: a candidate joins if it agrees with the S1 *or* with an already
     strongly-accepted sibling (transitive evidence for blank addresses), subject to the constraints the data obeys:
     ≤1 S1 per record, ≤5 S2 / ≤6 S3 per entity.
  3. Replace the greedy one-owner rule with a **global assignment** (min-cost flow over all entities: each record ≤1 owner).
  4. Set-level features for the group model: agreement of a candidate with the *accepted core* (iterated twice).
* **Target:** entities with 1 true match (0.941), blank-address pairs (73% of rejected true pairs), same-name distractors.
* **Expected:** +0.002 … +0.004 (largest uncertainty, and largest upside: this is where "ambiguous" pairs become decidable).
* **Gate:** hold-out gain ≥ +0.001, and no loss on singletons.

### C. Learned blocking (SC-Block, Brinkmann et al. ESWC 2024; universal dense blocking 2024)
* **What:** a supervised-contrastive bi-encoder (multilingual MiniLM) on name+address, trained on true pairs with
  in-batch/hard negatives; a 4th kNN channel next to the SVD channels; the prefilter and top-80 are unchanged.
* **Target:** the 63% of blocking misses that rank ≥100 in SVD space (recall 98.0% → ≥99%, ceiling 0.994 → ~0.997).
* **Cost:** training ~1–2 h, embedding 24M records ~1.5 h, ~18 GB of fp16 embeddings on D:.
* **Expected:** +0.001 … +0.003. **Gate:** validation candidate recall ≥ 98.8% at ≤90 candidates per S1.

### D. Use all the training data (industry standard)
* GBDT on all 2.2M training entities (currently 400k = 18%) via external-memory XGBoost / float16 storage; 3-seed average.
* **Expected:** +0.0005 … +0.001. Runs on CPU/RAM in parallel with A–C.

### E. France, structurally
* A–C are language-agnostic and learned, so they help France without French labels.
* **Test-time adaptation, done right:** the lesson from v5 (P19) is to use only *structural* pseudo-labels
  (cluster-consistent, from B) and never confidence-only ones.
* FES stays the France gate; France changes are submitted **bundled**, never one tiny tweak at a time.

### F. Stretch: small open LLM for the hardest groups (ComEM; Peeters 2024)
* Qwen2.5-7B-Instruct / Mistral-7B (**Apache-2.0**; not Jellyfish: CC BY-NC is not allowed) with QLoRA, "Select" prompts
  (S1 + all its candidates), for the ~1% hardest entities only. Needs a cloud GPU; only if A–C plateau.

## 4. Order of work and gates

| Phase | Work | Where | Gate to continue |
|---|---|---|---|
| 1 (now) | submit v10; A-lite (more data + augmentation + wide band) on validation; D in parallel on CPU | laptop | hold-out ≥ +0.001 |
| 2 | B: S2↔S3 scoring + consistent-cluster decision + global assignment, on validation | laptop | hold-out ≥ +0.001 |
| 3 | C: contrastive bi-encoder channel, recall on validation | laptop | recall ≥ 98.8% |
| 4 | one integrated full test run → leaderboard; France via FES + one bundled submission | laptop | leaderboard up |
| 5 | if the leaderboard is still < 0.985: A-full (mDeBERTa) / F on a cloud GPU | cloud (terms permitting) | – |

**Disk (D: ~105 GB free):** A needs ~10 GB (pair store), C ~18 GB (embeddings). Keep ≥30 GB headroom. Obsolete caches
(`work/baseline` 21 GB, `work/v3` 6 GB) can be removed **with approval**; their final outputs are archived in `versions/`.

**Runs:** every long run is launched detached (survives session restarts, P20), writes caches first, and checks disk
before starting (P23).
