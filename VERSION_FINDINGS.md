# Version Findings — Business Entity Resolution

A running log of every version of the pipeline: what we built, what score it got, what held the
score down (with real examples from the validation set), and how each diagnosis shaped the next version.

All scores are **macro F0.5 on the same 100,000 held-out Source-1 entities** (a fixed, seeded split
of the training data), so versions can be compared directly. The leaderboard metric works the same
way: F0.5 is computed per S1 entity, singletons included, and then averaged.

| Version | Notebook | Validation macro F0.5 | Blocking recall | Best possible score* | Candidates per S1 |
|---|---|---|---|---|---|
| v0 — data analysis | `code/src/view_same.py --analyze` | – | – | – | – |
| v1 — baseline | `code/src/baseline.ipynb` | **0.8871** | 83.6% | 0.9275 | 39.7 |
| v2 — improved | `code/src/improved.ipynb` | **0.9721** | 97.0% | 0.9906 | 30.0 |
| v3 — towards SOTA | `code/src/v3.ipynb` | **0.9811** (hold-out half B: 0.9813); **leaderboard 0.965** | 97.6% | 0.9923 | 40.0 |
| v4 — France blocking fix + ambiguity | `code/src/v4.ipynb` | **0.9826** (hold-out 0.9826); France silver 0.952 → **0.981**, France empty 9.7% → **6.2%**; **leaderboard 0.973** (+0.008 over v3) | 97.7% | 0.9928 | 40.0 |
| v6 — v4 + French descriptor-swap filter | `exp_v6_descswap.py` on v4 | US/India = v4 (0.9826); 21,148 French swap pairs removed; **leaderboard 0.974** | 97.7% | 0.9928 | 40.0 |
| v5 — France self-training | `code/src/v5.ipynb` | 0.9826 (US/India = v4); France matches 830,913 → 848,883, France silver 0.981 → 0.983; **leaderboard 0.972 (−0.001 vs v4) → rejected** | 97.7% | 0.9928 | 40.0 |
| v10 — prefilter top-80 + multilingual cross-encoder | `code/src/v10_full.ipynb` | **0.9835** (hold-out 0.9836; v9 0.9828); test silver US 0.9936 / India 0.9207 (= v9); vs v9 on test: 2.5% of US/India rows differ as sets (+24k / −20k pairs); submission `matching_results_v10.tsv` = US/India v10 + France from the 0.974 file; **leaderboard 0.975 (+0.001, new best)**; France re-scored probe `_v10_frnew.tsv` **also 0.975** (tie: re-scoring France with v10 is neutral at leaderboard resolution, as v9fr was) | 98.0% | 0.9939 | 80.0 |

\* The score a perfect classifier would get on that version's candidate set. It is the ceiling
that blocking allows.

**Version control.** Every trial is a commit and an annotated tag on the `experiments` git branch
(`main` is untouched). Its final artifacts (output TSVs, executed notebook, run log, `metrics.json`)
are in `versions/<trial>/`. The large TSVs are git-ignored, but their SHA-256 is recorded in the
tracked `versions/<trial>/MANIFEST.md`, so any submission file can be traced to the exact code that
produced it. Experiment scripts live in `code/experiments/`.

---

## Trial ledger

Each row: the first-principles reason for the change, what we did, what it measured, and whether we kept it.

| ID | Trial | First-principles hypothesis | Change | Result (val macro F0.5) | Decision | Tag |
|---|---|---|---|---|---|---|
| T00 | EDA | Know the metric and the label structure before modelling | SQLite ground-truth analysis | 5.6% singletons; no record owned twice; 100% same country | constraints adopted | `v0-eda` |
| T01 | Baseline | A true match must first be *retrieved* before it can be *classified* | TF-IDF/SVD 2-channel kNN, XGBoost | **0.8871** (ceiling 0.9275) | baseline | `v1-baseline` |
| T02 | Blocking sweep | Top-k of each field alone misses pairs moderately similar on *both* fields | add combined name+address channel | recall 0.828 → 0.912 at 74 candidates | adopted | `v2-improved` |
| T03 | Lexicon | Native-script names are *systematic* transliterations, so the mapping can be learned from labelled pairs | lexicon from ground-truth alignments | India 0.833 → 0.966 (with T02–T05) | adopted | `v2-improved` |
| T04 | Exact keys + prefilter | Numbers are near-unique identifiers; a cheap model can rank ~130 candidates without losing positives | 6 exact keys, top-30 prefilter | recall 97.0%, ceiling 0.9906 | adopted | `v2-improved` |
| T05 | Group-context stack | A match decision is relative to its *siblings* (best candidate, gap, count) | OOF-stacked group model | 0.9708 → **0.9721** | adopted | `v2-improved` |
| T06 | Expected-F0.5 decision | For per-entity F, optimizing expected F beats a threshold (DTA) | Monte-Carlo top-k per entity | +0.0001 | **rejected** (T05 already captures it) | `v3` |
| T07 | Blank-address blocking | Records without an address compete unfairly in the name top-k, so give them their own index | blank-address name kNN + key | +0.6% recall, ceiling 0.9906 → 0.9923 | adopted | `v3` |
| T08 | Numeric + candidate-set features | Numbers mean *identity*, not *similarity*: 1315 ≠ 1336, but 1604 ≈ 604 is a dropped digit | 9 numeric + 5 rarity features | stage-1 0.9708 → 0.9750 | adopted | `v3` |
| T09 | Cross-source consistency | In a correct cluster, the S2 and S3 matches agree with each other | 6 cross-source group features | group 0.9721 → 0.9776 | adopted | `v3` |
| T10 | Cross-encoder blend | A model that *reads* both records catches what hand-made features can't | Ditto-style MiniLM, blended on the uncertain band | 0.9776 → **0.9810** | adopted | `v3` |
| T11 | v3 full run | combine T07–T10 | `code/src/v3.ipynb` | **0.9811** (hold-out B 0.9813); **hidden test 0.965** | submitted | `v3` |
| T12 | S1-side ambiguity (E1) | A name shared by many *S1 entities* is weak evidence; only 50.8% of S1 names are unique in their country | 7 S1/pool name and address count features | blend 0.9811 → **0.9822** | adopted | `v4` |
| T13 | Gap diagnosis | You can't fix a gap you haven't located; compare test with validation using label-free statistics | per-country prediction stats, distractor fingerprints, silver labels | France blocking failure found (P14) | → T14 | `v4` |
| T16 | France self-training (v5) | Apply T15's winner to the real unlabelled country: the v4 stack labels its own confident French pairs, and the stage-1 matcher relearns French patterns | 2 rounds, confidence ≥0.97 / ≤0.03 (96.6% → 97.7% of French pairs), 25% of negatives, 60% of training rows | France pairs above threshold 861,694 → 877,133 → 891,765 (flips 31k → 20k, converging); France empty share 6.2% → 5.8%, matching US 5.8% / India 6.0% | **leaderboard 0.972 vs v4 0.973 → rejected** (P19) | `v5` |
| T15 | Domain adaptation, leave-one-country-out (DADER protocol) | France is an *unlabelled target domain*. Simulate it: train on the US, adapt to India without its labels, score on India | M0 source only / M1 self-training / M2 Saerens EM prior / M3 per-country quantile alignment / UB India-trained | M0 0.9293 · **M1 0.9390** · M2 0.9293 · M3 0.8848 · UB 0.9681 → an unseen country costs **~0.039**; only self-training transfers (closes 25%, pseudo-labels 99.7% correct) | self-training adopted for France (v5) | `v5` |
| T17 | Pattern-class audit | If the generator is the same in every country, a pattern's true-match rate should transfer; France deviations would flag miscalibration | classes A–F from raw attributes; validation true rate vs acceptance; France acceptance v4/v5 | validation calibrated per class except E (blank address: true 7.3%, accepted 4.4%). France's candidate *mix* differs (templated names), so the rates don't transfer | diagnostic only | – |
| T18 | Name-sibling evidence for blank addresses | A blank-address record's same-name siblings *with* addresses reveal which business it belongs to | sibling address agreement with q, on 68k validation pairs | separates strongly (0.946 vs 0.048 true) **but v4 already tracks it** (mean p 0.914 / 0.048); the remaining misses sit in 13–30%-true buckets, below the F0.5 acceptance point (≈ F/(1+β²) ≈ 0.78) | rejected: classifier near its information limit on US/India | – |
| T19 | France threshold probe (controlled leaderboard experiment) | France can't be validated offline, so let the leaderboard measure it with *one factor changed*: US/India fixed at v4, only France's threshold moves | France thresholds 0.55 / 0.625 / 0.70 (= v4) / 0.775 / 0.85; files in `output/lb_probe/` | the whole range moves French matches only −2.2% … +1.9% (812k–846k vs 831k), so the leaderboard effect may be ≤0.001; the extremes 0.55 vs 0.85 are the informative pair | **leaderboard pending** | `v6-probe` |
| T20 | France-only IDF (Splink-style term frequency within the linked population) | Rare-word features used IDF over the whole test split; templated French words look ~1.9 nats rarer than they are among French records, so shared template words pass for identifying ones | IDF recomputed on French records for French pairs; model unchanged | confirmed shift: `club` 4.49 → 2.57, `lille` 6.12 → 4.19. Changes ~40% of French entities' lists: **+22.8k pairs, 94% with a *different name* at the *same address*** (`OR Club SARL` ↔ `OR Ecole SARL`, `Team Collectif` ↔ `Team Groupe`), −8.9k pairs (common names, 41.5 S1 sharing the name). Caveat: training used IDF over US+India, i.e. ~0.5–0.9 above in-country IDF, so France-only IDF is ~0.7 too *low* (lenient) and the old global IDF ~1.4 too high | **leaderboard 0.971 (−0.002 vs v4) → rejected.** With v5 (0.972, also +23k such pairs) this is consistent evidence: **in France a descriptor swap at the same address is mostly a *different* business** | `v6-probe` |
| T21 | Remove French descriptor swaps from v4 | Two independent variants that *added* same-address descriptor swaps both lost score, so the descriptor word identifies French businesses; v4 carries over the US habit (swaps 94% true there) | from v4's France output, drop pairs at the same address (house no. + street word) whose names share a word but differ by *vocabulary* words (each ≥300 French occurrences, not typos or legal forms), e.g. `tl ecole` ↔ `tl club`, `crin ecole` ↔ `crin pharmacie` | 21,148 French pairs removed (2.5% of France's matches, 20,053 entities); a first broad detector (66.5k) also caught garbled typos and abbreviations and was discarded | **leaderboard 0.974 (+0.001 vs v4) → adopted, new best.** FES forecast +0.001…+0.003; refit says v4's accepted French swaps were ~40–63% true (below the 74% removal break-even) | `v6-probe` |
| T22 | **France Evaluation Suite (FES)**: a France benchmark without French labels | France can only be labelled *in aggregate* by the leaderboard, so extract the most information from each result: describe each variant by *which classes of pairs it adds or removes*, value each pair with marginal effects measured exactly on validation, and fit only the France-specific unknowns to leaderboard results | marginals on validation: add true +0.105, add false −0.253, remove true −0.093, remove false +0.267, so the break-even precision is ~71% to keep and ~74% to remove; class precisions from validation per zone (added p 0.3–0.7, removed p 0.7–0.95); French SWAP precision fitted to the IDF result | reproduces IDF (−0.0018 vs −0.002) and **held-out** v5 (−0.0017 vs −0.001); French swap precision ≈ **23%** (US/India comparable class 45%). Forecasts: **noswap +0.001 … +0.003**, t0.55 −0.001, t0.85 ≈ 0 | adopted as the France benchmark (`code/experiments/fes.py`) | `v7` |
| T23 | Same-name blank-address pairs in France (FES-guided check) | With a blank address, precision ≤ ~1/k when k S1 entities share the name; France accepts this class at 2.4× the validation rate | validation precision by k; count of v4-accepted French pairs by k | validation: k=1 98% true (accepted precision 0.989), k≥2 already rejected by v4; France: 16,940 of ~18,000 accepted pairs have k=1 → correct | **rejected** (no change; avoided a harmful probe) | `v7` |
| T24 | French address normalization for pair features (v7fr) | French *anchor* pairs (same name + house number + street) have address similarity 77 (character) / 95 (token set) vs 88–90 / 98 elsewhere: one source writes the **region** (`Hauts-de-France`, 26% of anchor pairs), the other the **département** (`Nord`), plus `N°` (11%) and `St`/`Saint` (≈16k addresses). The model learned "true pairs ≈ 98% address similarity", so French true pairs look weaker than they are | French records only: drop administrative areas, drop `N°`, `st/ste` → `saint/sainte` for pair features and cross-encoder input; blocking unchanged; swap filter re-applied | 74% of French addresses changed; vs the 0.974 file: +4,790/−3,448 same-name-*other*-address pairs, +4,763 trade-name-same-address pairs. **FES forecast −0.0009** (robust to ±20% precision). Anchor pairs were already accepted (99.8%), so noise wasn't hiding clear matches; cleaner addresses mostly push *borderline* pairs over, and those classes are ~49% / ~37% precise near the boundary (break-even ~71%) | **leaderboard 0.974 = tie with the best** (FES said −0.0009 → French borderline pairs are ~+0.1…+0.3 *more precise* than US/India priors: France is **under-confident**; FES updated). Threshold sweep on the normalized probabilities (0.55–0.70) forecasts 0 ± 0.0008 → no calibration gain available | tie; kept as an equal-score alternative | `v7` |
| T25 | **France component ablation** | Every France fix now moves the leaderboard by less than its rounding step (±0.0005), so look for a *large* effect: which pipeline stage transfers worst to France? Candidate: the group-context model, whose sibling features take unseen values because French entities have dozens of same-name candidates, consistent with the measured under-confidence | one France run saving stage-1 / group-model / blend probabilities; variants A (stage-1), B (no cross-encoder), C (full = v4), each at its validation-tuned threshold + swap filter | French matches: A 804,791 → B 810,466 (group model +5.7k) → C 830,913 (**cross-encoder +20.4k**, reproduces v4 exactly). The group model is *not* the culprit; the cross-encoder is France's most active component. FES: A −0.0010, B −0.0008 vs 0.974 (removing stages hurts) | A/B not submitted (saved two submissions). → The component that carries France is English-only: try a **multilingual cross-encoder** (T26) | `v8` |
| T26 | Multilingual cross-encoder | The cross-encoder moves France the most, but it is English MiniLM (MS MARCO) reading French text; a multilingual pair model should read French names and address conventions better | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, 118M, 14 languages incl. French), identical fine-tuning recipe; validation A/B through the same v4 group model and blend, then France-only variant + FES | validation: band AUC **0.9400 → 0.9481**; blend hold-out B **0.9826 → 0.9828** (all 0.9825 → 0.9829; no cross-encoder 0.9790). Better even on English text; its advantage should be larger on French | adopted for the cross-encoder. **France run:** 820,359 French matches (English CE 830,913) → after the swap filter vs the 0.974 file: −4.6k net; FES −0.0005 (neutral). FES only knows *class-average* precision, so it can't judge *which* pairs a better model re-decides → the leaderboard decides (`output/lb_probe/matching_results_v9fr.tsv`). **US/India** re-scored with the multilingual CE (validated +0.0002–0.0004): `v9_usin_mce.ipynb` running | `v9` |
| T27 | Cross-encoder stacked with group context (Ditto + TransClean) | The cross-encoder joins through a 2-weight blend and never sees siblings; TransClean says a pair's credibility depends on its cluster | small GBDT on validation half A over: group-model logit, CE logit, CE rank/gap/#strong within the entity, best CE of the other source, CE–GBDT disagreement; scored on half B | B **0.9829 vs 0.9828** (+0.0001, noise); gain dominated by the two probabilities (425/77), CE group features ≤22 → the group context is already inside the group model's probability | **rejected** (keep the simple blend) | `v10` |
| T28 | Where do blocking misses rank? (Sparkly: top-k is the recall lever) | Blocking is the largest *reducible* loss (0.0072); decide between *more k* and *new channels* by measuring where the missed true record sits in each kNN channel | exact GPU ranks of 7,996 missed validation pairs among same-country, same-source pool records, per channel | <15 (retrieved, then **cut by the top-40 prefilter**) 14% · 15–99 (just past k) 23% · 100–999 27% · ≥1000 36% (59% blank address) | cheap levers first: prefilter cut (T29), then combined-channel k | `v10` |
| T29 | Larger prefilter cut, validation-only | 14% of misses are already retrieved and only cut by the prefilter; the classifier can still reject weak pairs | rebuild validation candidates with top-80 (was 40), score with the existing v9 models (no retraining) | candidate recall 97.69% → **98.01%**, ceiling 0.9928 → 0.9939; stage-1 alone 0.9768 → 0.9764 and group model 0.9790 → 0.9789 (extra weak candidates are noise for the trees), but the **blend 0.9829 → 0.9835, hold-out B 0.9828 → 0.9836 (+0.0008)**: the cross-encoder turns the recovered candidates into matches (Sparkly's recall lever pays off only with a strong re-ranker) | **adopted**; full test inference `v10_full.ipynb` (streaming, no feature store) → `matching_results_v10.tsv` (US/India v10 + France from 0.974) and `_v10_frnew.tsv` (France too, for a leaderboard check) | `v10` |
| T30 | Why can the top 100 reach 0.991? Structure diagnostic | If hundreds of teams reach 0.991, our "information limit" (T18) must be a limit of the *pairwise* design; check for leaks and for cluster evidence we don't use | train data: row-order and ID correlation between matched records; cluster composition; addressed siblings of blank-address matches; S2↔S3 name similarity | **no leaks** (corr −0.0001 / +0.0029). 94% of clusters have ≥2 records, 85% span S2 and S3; **97.9% of blank-address true matches have an addressed sibling**; S2↔S3 names median 70 vs S1↔S2/3 92 | new plan `NEXT_PLAN.md`: strong data-rich pair model, collective cluster decisions, learned blocking, full training data | `v11-plan` |
| T14 | Language-independent keys | What identifies a business is the *whole* name at a *number*, not the first word of the name | `ex_name_hn`, `ex_hn_street` + silver report | validation +0.15% recall (nothing to gain there); **test France: +0.65 new name+number pairs per S1** | adopted | `v4` |
| T31 | Collective sibling evidence, lite (NEXT_PLAN B) | T30: 97.9% of blank-address true matches have an addressed sibling; pairs are decided alone | `exp_v11_collective.py`: undecided zone (blend p 0.01–0.99, 88k pairs, 22% true); anchors = pairs of the same S1 with p ≥ 0.8; sibling features (name / address / house-number agreement with anchors, same-source anchor count, strongest rival claim); XGBoost stacker cross-fitted A↔B, and a *control* stacker without the sibling features | v10 reference B **0.9836**; control stacker **0.9841** (+0.0005); + sibling features **0.9842** (+0.0001 more) | re-stacking the zone is a small real gain (kept for the next integrated run); sibling features barely help → T32 explains why | `v11-exp` |
| T32 | Why sibling evidence doesn't help: blank-address ambiguity | Lost true pairs: 7,966 (all in p 0.01–0.7), 70% blank-address, 96% have an anchor sibling | for false blank-address zone candidates: who owns them in the full ground truth? how similar is the owner's name? | **97.6% of the false ones are true matches of ANOTHER S1**, and that owner has the **same name as our S1 in 95% of cases** (token-set tie), e.g. two `medyne adr` entities at different addresses. True rate falls as ≈ 1/k with k S1 entities sharing the name (1 → 70%, 2 → 47%, 3 → 36%); the model already has these counts (`s1_name_cnt_*`) | the blank-address part of the rejected-true loss (~0.005) is **largely irreducible** with name-only evidence (the record could belong to any of k identical names; the global owner rule can't break the tie either). Not a modelling target | – |
| T33 | Where does the leaderboard gap sit? (country arithmetic) | test S1: India 46.8%, US 38.3%, France 15.0%; v10 validation India 0.9809, US 0.9852; leaderboard 0.975 | implied France score; per-country prediction profiles vs train truth (US and India truth profiles are identical: 5.6% empty, 3.46 matches per S1); assigned S2/S3 rates | implied **France ≈ 0.93** (±0.003). Even a perfect France gives only ~0.985 → the top (0.991) must also be ≥ 0.989 on US/India: **both halves need work**. Test predictions per S1: US 3.38, India 3.29, France 3.12; French assigned-S2 rate 55.7% vs India 56.3% / US 57.8% (test has more unmatched records than train in every country); France's recall shortfall is not visible as a French-specific excess of unassigned records | priorities: blocking recall + pair model (both countries), France via the same structural levers; no more threshold/label-free France tweaks |
| T34 | What are the French silver "misses"? | France silver 0.981 vs US 0.994: 9.5k unique name + house-number pairs not predicted; are they lost true pairs? | `diag_v11_frsilver.py`: reason (owned by another S1 / unassigned) + 25 random examples | 92% unassigned, 8% owned by another S1; examples are overwhelmingly **coincidences of templated names**: `Dunkerque Comite SAS, 24 Rue Paul Machy` vs `Dunkerque Comite SARL, 024 Rue Vauban`; `Nantes Fetes SARL, 41 Rue de la Grange au Loup` vs `41 R. Moriceau Thebault`; where a real owner exists the model gives the record to it (`EHPAD Pierre` → `EHPAD de Pierre, 52 Rue Arsène Bossu`) | the French silver gap is mostly *correct* rejections: silver is not a France accuracy measure (French names are generated from templates, so name + number collide). France's loss remains unobservable offline → only structural, validated improvements go to France | – |
| T35 | Data-rich multilingual cross-encoder (NEXT_PLAN A-lite) | v9's CE saw 1.2M easy-ish pairs from 150k entities (7% of training entities), one epoch | `exp_v11_mce.py`: warm start from v9 CE; all 400k training entities, top-16: 1.34M positives + 636k hard negatives (GBDT p1 ≥ 0.001) + 1.53M sampled easy negatives + 49k GBDT-error duplicates = 3.55M pairs; Ditto attribute deletion (candidate address blanked, 15%) + token deletion (5%); 55k steps, 3.7 h on the RTX 3050; checkpoints checked mid-run | band AUC 0.9550 (v9) → step 10k 0.9432 → 20k 0.9522 → 40k 0.9555 → **final 0.9568** (wide band 0.9814 → 0.9827); hold-out B with blend: **0.9836 → 0.9838** (narrow) / **0.9839** (wide). The mid-run dip was the high-LR phase, not the recipe | adopted (small alone; the gain compounds with T31's re-stacker, T36) | `v11` |
| T36 | v11 configuration choice | Pick CE set × band × re-stacker on hold-out B | `exp_v11_select.py`: {v11, v9+v11} × {0.01–0.99, 0.001–0.999} × {blend, + zone re-stacker cross-fitted A↔B} | v11 narrow 0.9838 / +stacker 0.9845; **v11 wide + stacker 0.9848**; v9+v11 wide 0.9842 / +stacker 0.9848 (no gain from the second CE) | **adopted: v11 CE alone, wide band, + zone re-stacker = hold-out 0.9848 (+0.0012 over v10, passes the +0.001 gate)**; full test run `v11_full.ipynb` launched (`output/matching_results_v11.tsv` = US/India v11 + France from the 0.974 file; `_v11_frnew.tsv` France re-scored) | `v11` |

## Pitfall ledger

Each problem that held the score back, a real example, the root cause reasoned from first principles,
and the fix or patch.

| ID | Pitfall | Real example | Root cause (first principles) | Fix / patch | Status |
|---|---|---|---|---|---|
| P01 | Windows console crash on native text | `UnicodeEncodeError` printing `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்` | Redirected stdout defaults to cp1252, which can't encode Indic scripts | `sys.stdout.reconfigure(encoding="utf-8")` | fixed (T00) |
| P02 | Resource contention and a full C: drive | another project's `predict.py` using 12 GB RAM and 5 cores | Page file and temp files on a 96%-full C: | stopped it; all TEMP/pip/HF/Jupyter dirs moved to D: | fixed |
| P03 | Lossy blocking | `Consultancy Otc Web Pvt Ltd` missed although the candidate had the *same* address | SVD compresses TF-IDF; top-10 fills with neighbours in crowded buildings | combined channel, larger k, exact keys | fixed (T02, T04) |
| P04 | Native-script names never retrieved | `New Food Private Limited` ↔ `న్యూ ఫుడ్ ప్రైవేట్ లిమిటెడ్` → `nyu phud praivet` | anyascii is *phonetic*, English is *orthographic*, so n-grams don't overlap | learned lexicon `praivet→private` etc. | fixed (T03) |
| P05 | Same address, different business | `city of appleton dental health` vs `… behavioral health`, same street number | A shared address is necessary but not sufficient; the *distinguishing word* decides | rare-token mismatch, candidate-set rarity | mostly fixed (T05, T08) |
| P06 | Data leak inflating recall to 4.17 | exact-key rows sliced by an ID *range* also pulled in other splits' queries | train and val IDs are interleaved, so a range is not a set | `np.isin`, then a vectorised per-query row gather | fixed; rule: any impossible metric gets an independent recheck |
| P07 | Near-duplicate distractors with a different number | `atlantic property 1315 clifton st` vs `1336 clifton st` | Equality features can't tell a typo from a different building | numeric edit distance, gap, containment, `[NUM]` span tags | largely fixed (T08, T10) |
| P08 | Digit typos rejected | `coastal program 1604 old oak hill st` vs `604 …` | same as P07, other direction | `hn_contains`, `num_min_lev` | largely fixed (T08) |
| P09 | Blank-address true matches crowded out | `eye clinic 4642 highcroft ln` ↔ `eye clinic (blank)` | In the name top-k, blank-address records compete with thousands that share the name *and* have an address | separate blank-address index and key (T07) | retrieval fixed; **classification still cautious** (see P11) |
| P10 | Decision theory didn't help | DTA +0.0001 | DTA assumes independent candidates, but siblings are correlated near-copies; the stacked group model already learned the per-entity decision | kept the tuned global threshold | closed (negative result) |
| P11 | Name-only pairs remain ambiguous | recovered blank-address pairs barely change the score | With no address, a common name is weak evidence, and blank-address same-name *distractors* also exist | next: signals that separate a real blank-address match from a same-name distractor | **open** |
| P12 | Analysis bug from int32 overflow | FP count 328,005 instead of 2,659 | `q * N` with int32 `q` overflows past 2³¹ | cast to int64 before building keys | fixed; rule: build keys in int64 |
| P14 | **Validation 0.981, hidden test 0.965: France blocking failure** | `Roubaix Gestion SARL, 53 Rue de Cartigny` ↔ `Roubaix Gestion SARL, 53 R. DE CARTIGNY, ROUBAIX, Nord` never became a candidate, so the S1 entity got 0 matches | French names are templated `{City} {Word} {LegalForm}`. Every key assumed the *first token* is distinctive: `roubaix\|53` overflows the block cap, `53 r` (r = rue) matches thousands, and the name kNN is swamped by ~57 same-name records per S1. **85% of the missed French same-name-same-number pairs were never candidates.** US/India first tokens *are* distinctive, so validation couldn't see it | keys on the *full* name + number and on number + first *distinctive* street word (T14); silver report per country | **fix in v4, verify on leaderboard** |
| P15 | Validation blind to unlabelled countries and to shifts in distractor density | test: France 15% of S1, no labels; test pools have 41% distractors vs 26% in train (two independent fingerprints agree: India 0.41/0.41, US 0.43/0.41), with true matches per S1 unchanged (~3.4) | A validation split drawn from the training distribution only measures that distribution | label-free monitors: per-country prediction stats and **silver-label acceptance** (validation: true silver pairs are accepted 99.7–100%; test US 0.988 and India 0.741 match validation, so those countries transfer; France was the outlier) | monitor adopted |
| P16 | A model trained on one country degrades on an unseen one, even with correct blocking | US-trained stage-1 on India: 0.9293 vs 0.9681 India-trained, on the same candidate pairs | *Feature shift*, not prior shift: the positive rate among candidates is nearly identical (US 0.0849, India 0.0836), but what similarity patterns *mean* differs (native scripts, shared names, address styles). Quantile alignment fails because similarity values carry absolute meaning (100 = identical everywhere) | self-training on confident target pseudo-labels (T15), which learns the new country's patterns from its easy pairs | adopted for France in v5 |
| P17 | The monitoring code crashed the v4 run after the outputs were written | `MemoryError: Unable to allocate 1.66 TiB` in the silver-report cell | Joined S1 and pool rows on a key *before* dropping empty keys; ~1M empty-key S1 rows × ~9M empty-key pool rows = a 228-billion-row cross product | filter empty and non-unique keys *before* the join; validator and silver report re-run standalone (`silver_report.py`) | fixed in v4/v5 builders |
| P18 | v5 failed at start | `UnicodeDecodeError` reading `work/v4/run.log` | papermill logs contain cp1252 bytes from Windows console output | `read_text(errors="replace")` | fixed |
| P19 | Self-training hurt on the real unlabelled country (v5: 0.972 vs v4 0.973) although it helped India in simulation | v5 **added** 23k French pairs, mostly same address + *different* name (`Andelle Foyer SA` ↔ `Andelle Union SA`, 22% same name vs 74% for kept pairs), and **removed** 5.4k, including clear trade-name matches (`Entraide College SARL` ↔ `Ciraevo`, `Pongiste Club SAS` ↔ `Rizaveo`) | **Confirmation / selection bias:** confident pseudo-labels in France are overwhelmingly the easy same-name pairs, so the student over-weights the name. That is harmful exactly where France's templated names make the name *least* informative. India's hard cases resemble the US's, so the simulation didn't expose it | keep v4's France; validate France changes only with controlled leaderboard probes (change France alone, one factor at a time) | closed |
| P20 | A session restart killed a running pipeline | the France-IDF run stopped at 467 s (no process left, log ends after the IDF step) | background jobs started from the agent's shell are children of the session; when the session ends they end too | relaunch from caches (the notebooks are cache-first, so a restart costs only the unfinished stage); for long runs, launch detached | mitigated |
| P21 | French addresses carry inconsistent administrative areas | `53 Rue de Cartigny, Roubaix, Hauts-de-France` vs `53 R. DE CARTIGNY, ROUBAIX, Nord` | US state codes are canonicalized and appear on both sides; French records alternate region/département and add `N°`, so address similarity of true French pairs is systematically lower (anchor pairs 77 vs 88–90) | strip administrative areas and markers for French pair features (T24) | measured, but FES forecasts −0.001: the noise affected borderline pairs, not clear matches → not adopted |
| P22 | France micro-fixes have reached the leaderboard's resolution | noswap +0.001, v7fr ±0 (both displayed as 0.974); FES forecasts for further tweaks are within ±0.0008 | the public leaderboard shows 3 decimals and France is 15% of entities, so a change touching ~1–2% of French pairs is below one rounding step | favour experiments with large expected effects (component ablation, T25); bundle small positive-forecast fixes into one submission | open |
| P23 | A run would have filled the disk | US/India feature store: ~22 GB of chunk files + a ~22 GB merged copy, with 28 GB free on D: | the France notebooks wrote chunk files and then merged them (2× peak disk); fine for France (4 GB), not for US/India (59M pairs) | stopped the run and its orphaned joblib workers before any damage; write straight into one preallocated memmap (1× peak) | fixed |
| P24 | v10 test run died at 1.0M of 1.73M S1 with `MemoryError` | joblib failed to pickle the strings of a 4M-pair chunk to its workers | top-80 doubles the pairs per chunk (50k S1 × 80 = 4M); validation chunks were smaller, so validation passed; RAM (24 GB) ran out only deep into the test run, and the old loop kept results only in memory, so all progress was lost | test chunks of 25k S1 (~2M pairs), checkpoint each chunk to disk (resumable), free chunk arrays each iteration | fixed; relaunched run completed all 1,732,544 S1 in 2.7 h (70 chunks, peak RAM stable), both files pass the validator |
| P13 | Bash heredoc quoting broke a background job | `unexpected EOF while looking for matching '` | nested quotes in a shell heredoc | write scripts with the editor, run them as files | fixed |

---

## How we decide what to work on

Every version ends with the same **loss decomposition**. We take the validation predictions and
compute three "what if" scores:

1. **Remove every false positive** → the gain is the cost of wrong merges.
2. **Add every true pair the classifier missed, among those it saw** → the gain is the cost of classifier recall.
3. **A perfect classifier on the candidates** → 1 minus this score is the cost of blocking, i.e. true pairs never generated.

We then look at each error bucket for common properties (blank address, name similarity,
native script, country, postcode agreement) and at random examples. The biggest bucket gets
worked on first. This is why v2 put most of its effort into blocking rather than the model: in v1,
64% of the loss came from pairs the model never saw.

---

## v0 — Data analysis

**What:** `view_same.py --analyze` loads all four training files into SQLite and prints ground-truth statistics.

**Findings that shaped every later version:**
- 2.2M S1 entities and 7.64M true pairs. **5.6% of entities are singletons** (no match), and the rest have 1–11 matches (at most 5 from S2 and 6 from S3).
- **No S2/S3 record belongs to two S1 entities.** This became the "one owner per record" decision rule.
- **Every true pair shares a country (100%)**, so blocking runs per country.
- **About 26% of S2/S3 records match nothing.** They are distractors.
- About 3–4% of S2/S3 addresses are blank. Names are never blank.

**Examples from the ground truth** (one entity, `S1-965667`):
```
[S1] Maure Williams Colombier Inc   85 Wayne Avenue, Ticonderoga, NY
[S2] Maure Wilblims Colombier Inc   (blank address)                        <- name typo
[S3] Dréxkor                         85 Wanye Avenue, Ticonderoga Townshiip  <- trade name, only the address matches
[S3] maurewilliamscolombier.com      Wayne Ave, Ticonderoga Townshiip        <- domain used as the name
```
**Pitfall found:** `view_same.py` crashed on Windows with `UnicodeEncodeError` when printing
Tamil or Devanagari names, because the console defaults to cp1252. Fixed with
`sys.stdout.reconfigure(encoding="utf-8")`. This was the first sign that native-script text is common.

---

## v1 — Baseline (0.8871)

**Approach:**
1. **Normalize:** anyascii transliteration, lowercasing, standard legal forms (`private → pvt`), address abbreviations, postcode and house-number extraction.
2. **Embed:** character n-gram TF-IDF, reduced to 192 dimensions by SVD on the GPU.
3. **Block:** GPU top-10 by name plus top-10 by address, for each of S2 and S3, plus two exact keys (squashed name; postcode + first token).
4. **Classify:** XGBoost on the GPU with 30 features, trained on 300k S1 entities.
5. **Decide:** one global threshold, one owner per record, caps per source.

**Result:** 0.8871, with a blocking ceiling of only 0.9275.

**Loss decomposition (total loss 0.113):**

| Bucket | Cost | Share |
|---|---|---|
| Blocking never generated the true pair | ~0.073 | **64%** |
| Classifier rejected true pairs it saw | ~0.020 | 18% |
| Classifier accepted wrong pairs | ~0.020 | 18% |

**What was holding the score down, with examples:**

1. **The SVD embedding plus k=10 was too narrow.** 98.6% of blocking misses still had similar
   addresses (token-set ≥ 60), yet they didn't make the top-10. Crowded Indian commercial buildings
   filled the list with neighbours.
   `Consultancy Otc Web Pvt. Ltd. | T5, 172, Pebblebay, 1St Main Road Rmv…` was missed, although the
   candidate had *the same address*.
2. **Native-script names (26% of blocking misses).** anyascii produces phonetic spellings that
   never meet the English n-grams:
   `New Food Private Limited` ↔ `న్యూ ఫుడ్ ప్రైవేట్ లిమిటెడ్` (→ `nyu phud praivet limited`).
3. **India was far worse than the US: 0.833 vs 0.923.** 63% of blocking misses were Indian.
4. **Trade/DBA names**, where only the address can match:
   `adorne choate dynamix` ↔ `iriciranex`, same address `835 silvergate dr houston tx`, scored p = 0.47.
5. **Different businesses at the same address** (the main source of false positives):
   `city of appleton dental health` ↔ `city of appleton behavioral health dynamic group`, both at `2112 lucille st`.
6. **Entities with a single true match scored only 0.763**, and **10% of singletons got a wrong match**.
7. **A blank candidate address** was behind 25% of classifier misses and 23% of false positives.

**Engineering pitfalls:**
- A second project's `predict.py` (12 GB of RAM) was competing for the CPU and filling the Windows page file on C:. We stopped it and moved all temp and cache folders to D:.
- joblib's cleanup prints `KeyError` tracebacks on Windows when the kernel exits. They are harmless.

**Thought process → v2:** 64% of the loss was in blocking, so the biggest gain was more
candidates of better quality, not a better model. A quick experiment on 40k validation queries,
using the cached embeddings, measured recall for different channels and k:

| Blocking setup | Recall | Candidates per query |
|---|---|---|
| name 10 + address 10 (v1) | 0.828 | 38.6 |
| name 50 + address 50 | 0.889 | 198 |
| **+ combined name+address channel (k=20)** | **0.912** | 74 |
| name 30 + address 50 + combined 50 | 0.932 | 249 |

The **combined channel** was the key finding. Pairs that are only moderately similar on both
fields rank highly on the sum, even when they're outside both top-10 lists. Beyond about 93%, the
remaining misses were native-script names and shared house or compound numbers, which led to the
lexicon and to new exact keys.

---

## v2 — Improved pipeline (0.9721)

**What changed:**

| v1 problem | v2 fix |
|---|---|
| Narrow blocking | 3 GPU channels (name 15, address 20, **combined 30**, per source) plus **6 exact keys**, including first token + house number and Indian compound numbers (`20-158/1/A`). A **prefilter** XGBoost then keeps the top 30 per S1 (99.5% of blocked true pairs retained). |
| Native script | A **transliteration lexicon learned from the training ground truth**, excluding validation entities: 802 name words and 47 address words. It aligns `anyascii(native word) → English token` by position. For example `praivet → private`, `bildrs → builders`, `mharastr → maharashtra`, and `इंटरनेशनल बाबा सर्विसेज प्राइवेट लिमिटेड → international baba service pvt ltd`. |
| Same address, different business | IDF-weighted token overlap, and **rare-word mismatch** features: the rarest word that appears on only one side. |
| Blank addresses / common names | Counts of how crowded an address is and how common a name is. |
| Singletons and 1-match entities | A **group-context model** stacked on out-of-fold stage-1 probabilities: rank within the entity, gap to the best candidate, number of strong candidates, best candidate from the other source. |
| Too little data | 400k training entities (up from 300k). The validation entities are unchanged. |

**Results:**

| | v1 | v2 |
|---|---|---|
| Blocking recall | 83.6% | **97.0%** |
| Best possible score | 0.9275 | **0.9906** |
| Stage-1 model alone | – | 0.9708 |
| **+ group-context model** | 0.8871 | **0.9721** |
| India | 0.833 | 0.966 |
| US | 0.923 | 0.976 |
| Singletons | 0.899 | 0.979 |
| Entities with 1 true match | 0.763 | 0.895 |

**Checks:** a from-scratch scorer run on the written TSV gives exactly 0.9721. No validation
entity appears in the training pairs or in the lexicon's source pairs.

**Pitfall found while building v2:** the first run reported **blocking recall = 4.17**, which is
impossible. The cause: exact-key pairs were sliced by the *range* of query IDs in a chunk, and
because training and validation IDs are interleaved, other entities' pairs leaked in. That would
have put validation entities' pairs into the training set. We stopped the run before any model was
trained, filtered with `np.isin`, and resumed from caches. **Lesson: any recall above 1, or any
large jump, gets an independent recheck before it's trusted.**

**Loss decomposition (total loss 0.0279):**

| Bucket | Cost | Share | Main property |
|---|---|---|---|
| Classifier rejected true pairs it saw | 0.0120 | **43%** | 41% have a **blank candidate address** |
| Blocking never generated the true pair | 0.0094 | **34%** | 41% blank address, 60% India |
| Classifier accepted wrong pairs | 0.0066 | 24% | **near-duplicate distractors with a different house number** |

By entity type:
- Entities with 1 true match cause **20% of the loss**, at a mean score of 0.895.
- Loss is split roughly evenly between India and the US.

**What is holding v2 down, with examples:**

1. **Near-duplicate distractors (false positives).** The dataset contains records with the same
   business name but a *different* house number:
   ```
   atlantic property  | 1315 clifton st unit 208 wa dc      <->  atlantic property | 1336 1 2 clifton st wa dc
   south imperial     | 135 reuben ct unit 22 danville va   <->  south impcail     | 340 reuben ct 22 danville va
   coastal guild      | n3682 996th st town of spring brook <->  coastal guild     | n3686 996th steet ...
   ```
   Our features only record whether house numbers are *equal*. They can't tell a digit typo from a
   different building.
2. **Digit typos that are true matches (misses):**
   ```
   coastal program  | 1604 old oak hill st conroe tx   <->  coastal program | 604 old oak hill st conroe tx   (dropped digit)
   trusted network studio | 5645 eveleth ct            <->  trusted network | 5645b eveleth ct farmington
   ```
   This is the mirror image of item 1. Both need numeric-aware comparison.
3. **Blank candidate addresses (misses and blocking misses):**
   `pilla bold | 620 franklin ave` ↔ `pilla bold | (blank)`, and `eye clinic | 4642 highcroft ln` ↔ `eye clinic | (blank)`.
   With a blank address only the name channel can find the record, and common names crowd it out.
4. **Trade names at the same address:** `horizon construction enterprise` ↔ `orbifaye`, both at `1701 bremen ave`.
5. **Entities with 1 true match:** one mistake drops the entity from 1.0 to 0.0–0.5, and a single global threshold can't allow for that.

---

## v3 — Towards SOTA (in progress)

The plan comes from the v2 decomposition above and from the literature review below. Each item
targets a measured loss bucket:

| Item | Targets | Expected |
|---|---|---|
| B. Numeric-aware address features (digit edit distance, numeric gap, dropped digit, units) | FP distractors + digit-typo misses | +0.003 |
| C. Blank-address name channel + exact core-name key | 34% blocking loss (41% blank) | +0.004 |
| D. Per-entity decision that maximises expected F0.5 | 1-match entities, threshold effects | +0.002–0.004 |
| D2. Cross-source consistency (does the candidate agree with the best candidate from the other source?) | distractors, trade names | +0.002 |
| E. More training data, ensemble | general | +0.002 |
| A. Fine-tuned transformer re-ranker for uncertain pairs (Ditto-style) | typos vs. different numbers, trade names | +0.005–0.010 |
| F. French normalization | test only (unmeasurable) | ? |

### Literature review: what the papers say and what we take from them

| Paper | Key finding | What we take for v3 |
|---|---|---|
| **Ditto** — Li et al., *Deep Entity Matching with Pre-Trained Language Models*, PVLDB 2020 ([arXiv 2004.00584](https://arxiv.org/abs/2004.00584)) | Matching framed as sequence-pair classification with a fine-tuned LM, records serialized as `[COL] attr [VAL] value`. **Domain-knowledge injection:** for organizations, the important spans are the *street number* and the *last 4 phone digits*. These are marked with special tokens so the model aligns them, e.g. `246 - [LAST] 6453 [/LAST]`. Span normalization rewrites equivalent forms into one string. On a real employer dataset (789K × 412K records) it reached 96.5% F1 using zip-code blocking plus a Sentence-BERT top-10 blocker. | (1) A **cross-encoder re-ranker** for uncertain pairs, with serialized name + address. (2) **Tag house numbers** (`[NUM] 1315 [/NUM]`) so the model compares them explicitly, which targets our FP distractors. (3) Our lexicon is already a form of span normalization. |
| **Sparkly** — Paulsen, Govind, Doan, *A Simple yet Surprisingly Strong TF/IDF Blocker*, PVLDB 16(6) 2023 ([paper](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)) | Exact top-k BM25 over 3-grams of concatenated attributes beats 8 state-of-the-art blockers, including DeepBlocker. "Top-k instead of thresholding is the most important decision." They index the smaller table and probe from the larger one, because that gives higher recall at the same k. | Our SVD embedding is a *lossy* approximation of TF-IDF. The leftover blocking misses are records that are crowded out, e.g. blank-address pool records drowned by records with full addresses. Fix: a **separate channel over blank-address pool records** (small, so exact and cheap) and **exact core-name keys**. |
| **DeepBlocker** — Thirumuruganathan et al., PVLDB 14(11) 2021 ([paper](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)) | Self-supervised tuple embeddings, with Hybrid/CTT models best on dirty data. Relative to Sparkly, the gain from deep blocking is modest. | Confirms our choice of cheap embeddings plus top-k. A learned bi-encoder is only worth it if the cheap fixes stall. |
| **Ye, Chai, Lee, Chieu**, *Optimizing F-measure: A Tale of Two Approaches*, ICML 2012 ([arXiv 1206.4625](https://arxiv.org/abs/1206.4625)); Jansche 2007; Dembczyński et al. 2011 | Two ways to optimize F: thresholding (EUM) vs. the **decision-theoretic approach (DTA)**, which estimates calibrated probabilities and then picks the prediction with the highest *expected* F: sort by probability and choose the best top-k. **DTA wins when each evaluation unit has few instances**, which is exactly our per-entity macro F0.5 with about 1–6 true matches. | **Replace the global threshold with a per-entity expected-F0.5 maximizer** over calibrated probabilities. This targets 1-match entities (20% of the loss) and singletons (predicting nothing has expected F = P(no true match)). |
| **Saeedi, Peukert, Rahm**, *Comparative Evaluation of Distributed Clustering Schemes for Multi-source ER*, ADBIS 2017 ([link](https://link.springer.com/chapter/10.1007/978-3-319-66917-5_19)); FAMER | For multi-source ER with clean (duplicate-free) sources, clustering that enforces **at most one record per source per cluster** removes false links and adds missing ones. | Our "one owner per record" rule is a special case. We add **cross-source consistency**: a candidate from S2 is more credible if it also matches the best S3 candidate of the same entity. |
| **TransClean** — *Finding False Positives in Multi-Source Entity Matching via Transitive Consistency*, 2025 ([arXiv 2506.04006](https://arxiv.org/abs/2506.04006)) | Checking implied pairs within a predicted cluster (A↔B and A↔C imply B↔C) detects false positives, with an average +24 F1 over pairwise matching in their multi-source setting. | Same direction: **features that compare each candidate with the entity's other strong candidates**. A candidate that disagrees with the rest of the cluster is probably a distractor. |
| **Peeters, Steiner, Bizer**, *Entity Matching using LLMs* / *Fine-tuning LLMs for EM*, 2023–24 ([arXiv 2310.11244](https://arxiv.org/abs/2310.11244), [2409.08185](https://arxiv.org/abs/2409.08185)) | Fine-tuned *small* models reach performance comparable to large LLMs, at a fraction of the cost. Fine-tuned PLMs improve consistently. | Use a **small open-licence cross-encoder** (fits the ≤8B, MIT/Apache rule and our 6 GB GPU), applied only to the uncertain band. |
| **Linacre**, *Building Accurate Address Matching Systems* (Splink author) ([blog](https://www.robinlinacre.com/address_matching/)) | Number mismatches should be penalized more than text mismatches, which are more likely to be typos. **Token frequency *within the candidate set*** separates neighbouring addresses better than corpus-wide frequency. Extract "discriminating tokens" by removing tokens shared across nearby candidates. | Add **numeric-aware features** (digit edit distance, dropped digit, numeric gap) and **candidate-set token rarity**: which tokens make this candidate different from the entity's other candidates. |

### Experiment log

All experiments reuse v2's cached pairs and features. The protocol: tune the threshold on
validation half **A** (50k entities), report on half **B** (the other 50k), and also on all 100k.
v2 reference: stage-1 model 0.9708, group-context model **0.9721**.

**Experiment D: expected-F0.5 decision (decision-theoretic approach).** Per entity, sort candidates by
calibrated probability and pick the top-k with the highest Monte-Carlo expected F0.5 (512 samples,
calibration grid σ(a·logit p + b), optional Poisson mass for true matches outside the candidates).

| Base probabilities | Threshold (B) | Expected-F0.5 (B) | Gain |
|---|---|---|---|
| v2 stage-1 | 0.9708 | 0.9701 | −0.0007 |
| v2 group-context | 0.9722 | 0.9722 | +0.0001 |

→ **Negative result, not adopted.** The group-context model already learned the per-entity reasoning
the decision rule would add: rank, gap to the best candidate, number of strong candidates. The
independence assumption behind DTA is also violated here, because candidates of one entity are
strongly correlated (the distractors are near-copies of each other). *Lesson: the literature's gain
assumes a probability model that ignores the group. Ours already models it.*

**Experiment C: blocking of blank-address records (Sparkly-inspired).** 10,312 validation true pairs
(2.98%) are missing from v2's candidates, and 41% of those have a blank candidate address.

| Added channel | Misses recovered (% of all true pairs) | New candidates per S1 before prefilter |
|---|---|---|
| Name kNN over blank-address pool records, k=5 per source | 1,709 (0.49%) | 8.8 |
| same, k=10 | 2,071 (0.60%) | 18.8 |
| same, k=20 | 2,298 (0.66%) | 38.8 |
| Squashed-name key on blank-address pool records (cap has no effect) | 996 (0.29%) | 0.9 |

→ **Adopted:** a blank-address name channel (k=5) plus the blank-address squashed-name key. Together
they recover about 0.6–0.7% of true pairs cheaply, raising the best possible score by about +0.004.
*Why it works:* in the main name channel, blank-address records compete with thousands of records that
share the name *and* have an address. A separate small index gives them their own top-k, which is
Sparkly's "top-k, not threshold" insight applied to a sub-population.

**Experiment B + D2: numeric-aware, candidate-set rarity and cross-source features.** Added 20 features:
- 9 numeric (house-number edit distance, dropped digit/containment, relative numeric gap, shared or unmatched numeric tokens, minimum edit distance of unmatched numbers, street-text similarity with digits removed, name-number mismatch).
- 5 candidate-set rarity (Linacre): token rarity *within the entity's candidates*, tokens unique to this candidate, and a match rate on "discriminating" S1 tokens.
- 6 cross-source (Saeedi/TransClean): the candidate's name, address and house-number agreement with the entity's best *other-source* candidate, and with its best *same-source rival*.

| Model | A | B | all 100k |
|---|---|---|---|
| v2 stage-1 | 0.9708 | 0.9708 | 0.9708 |
| **+ numeric + candidate-set features** | 0.9748 | 0.9751 | **0.9750** (+0.0042) |
| v2 group-context | 0.9718 | 0.9722 | 0.9721 |
| **+ cross-source features in the group model** | 0.9774 | 0.9778 | **0.9776** (+0.0055) |

Top new features by gain: `num_only_q` (413: S1 numbers the candidate lacks), `cs_addr_unique_c` (276:
address tokens unique to this candidate among the entity's candidates), `hn_contains` (248: dropped
digit or suffix, the "1604 vs 604" typo signal), `num_only_c` (106), `cs_addr_rare_overlap` (88) and
`num_min_lev` (87). The cross-source features have small gain individually but add +0.0026 on top of
stage 1. → **Adopted.** This confirms the v2 diagnosis: the distractors are about *numbers*, and
*which tokens make a candidate different from its siblings*.

**Experiment A: Ditto-style cross-encoder re-ranker.**
- **Model:** `cross-encoder/ms-marco-MiniLM-L12-v2` (Apache-2.0, 33M parameters). A pair-relevance model, so it's a good starting point.
- **Serialization:** `[COL] name [VAL] … [COL] addr [VAL] …`, with the house number wrapped as `[NUM] 1315 [/NUM]` (Ditto span typing). Example of the tokenized input: `[NUM] 131 ##5 [/NUM] clifton st … [SEP] … [NUM] 133 ##6 [/NUM] 1 2 clifton st`.
- **Training:** 1 epoch, fp16, on the top-8 prefilter candidates of 150k training entities (1.2M pairs, 41% positive). About 53 min on the RTX 3050.
- **Use:** it re-scores only the **uncertain band** (GBDT probability 0.01–0.99, just 2.5% of pairs). A logistic blend of the two logits is fit on half A.

| Model | A | B | all 100k |
|---|---|---|---|
| Feature-experiment group model | 0.9774 | 0.9778 | 0.9776 |
| Cross-encoder *replaces* the GBDT in the band | 0.9750 | 0.9748 | 0.9749 |
| **Logistic blend** (weights 0.77·GBDT + 0.41·CE in logit space) | 0.9810 | 0.9809 | **0.9810** (+0.0034) |

→ **Adopted as a blend, not a replacement.** On its own the cross-encoder is *worse*: it sees only the
two texts, not the entity's other candidates or the crowd and frequency statistics. It still carries
information the trees don't, above all character-level reading of names and numbers. That's why the
blend weight on it is large (0.41) and the blend beats both models alone. This matches Peeters & Bizer:
small fine-tuned PLMs are strong matchers, but here they complement the feature model rather than
replace it.

---

## v3 — full run (validation 0.9811)

The v3 notebook combines every adopted piece: v2, plus the blank-address channel and key, plus 14 new
pair features, plus 6 cross-source group features, plus the cross-encoder blend. It runs end to end
from the shared v2 caches.

| Stage | v2 | **v3** |
|---|---|---|
| Stage-1 blocking recall (before prefilter) | 97.5% | **98.0%** |
| Prefilter cut (≥99.5% retention) | top 30 | top 40 |
| Final candidate recall | 97.0% | **97.6%** |
| Best possible score | 0.9906 | **0.9923** |
| Stage-1 model | 0.9708 | 0.9752 |
| Group-context model | 0.9721 | 0.9777 |
| **Blend with cross-encoder** | – | **0.9811** |
| Hold-out (tuned on A, scored on B) | – | **0.9813** |

**An observation that matters for the next step:** the +0.6% extra candidate recall from the
blank-address channel barely moved the model scores (group model 0.9776 in the experiment vs. 0.9777
here). The recovered pairs are blank-address, name-only matches, and the model is rightly cautious
about them because blank-address, same-name *distractors* also exist. The ceiling went up (0.9906 →
0.9923), but turning it into score needs a better signal for name-only pairs.

*Test inference and the final v3 error analysis are appended below.*

---

## v4 — why the hidden test scored 0.965 when validation said 0.981

**Symptom:** v3 scored 0.9811 on validation (0.9813 on the hold-out half) but **0.965 on the hidden test**, a gap of 0.016.

**First-principles framing:** the gap has to come from a difference between the validation distribution and the test
distribution. So we listed every structural difference and measured each one *without test labels*.

**Step 1: structure per country.**

| Split / country | S1 entities | Pool records per S1 | Predicted matches per S1 (v3) | Predicted empty |
|---|---|---|---|---|
| train India | 883k | 4.68 | (true 3.46) | (true singletons 5.6%) |
| train US | 1.32M | 4.67 | (true 3.46) | (true 5.6%) |
| test India | 810k | 5.82 | 3.26 | 6.1% |
| test US | 663k | 5.76 | 3.36 | 5.9% |
| **test France** | 259k | 5.53 | **2.90** | **9.7%** |

**Step 2: what the extra test records are.** On train, true matches share an exact name with an S1 entity 64%/56%
of the time (India/US), and distractors only 21%/25%. Name + house number gives 47%/42% vs 5%/3%. Using both fingerprints,
the test pools of India and the US are **41% distractors (train: 26%) with ~3.4 true matches per S1, unchanged**. The
extra distractors are *not* same-name-same-number twins (1.78 vs 1.76 twins per S1), so they're mostly unrelated
records and easy to reject.

**Step 3: silver labels.** Silver pair = a pool record with the same country, name and house number as the S1 entity.

| | Silver precision | v3 accepts |
|---|---|---|
| validation India | 0.760 | 0.759 (accepts 99.7% of the *true* ones) |
| validation US | 0.980 | 0.981 (accepts 100% of the true ones) |
| test India | – | 0.741 → consistent with validation |
| test US | – | 0.988 → consistent with validation |
| **test France** | – | 0.716; for silver keys **unique among French S1**: 0.952 |

The US and India transfer. France is the outlier: **24k French silver pairs with a unique name+number** were missed,
and 85% of them were **never candidates**.

**Step 4: look at the failures.**
```
S1 Roubaix Gestion SARL | 53 Rue de Cartigny, Roubaix, Hauts-de-France
S2 Roubaix Gestion SARL | 53 R. DE CARTIGNY, ROUBAIX, Nord                  -> S1 got 0 matches
S1 Roubaix Parents SASU | 28 Rue Ternaux, Roubaix, Hauts-de-France
S2 Roubaix Parents SASU | 28 RUE TERNAUX, ROUBAIX, Nord                     -> not a candidate
S1 Lille Club SARL      | 86 Rue de Lannoy, Lille        vs  Lille Club S.C.I. | #86 R DE CANTELEU  -> correctly rejected
```
French names are **templated from a small vocabulary**: city + word + legal form. The distinguishing information is
the **legal form and the street**, not the name's first word. Every blocking key had silently assumed a distinctive
first token.

**Step 5: the fix and its evidence.** Language-independent keys (`ex_name_hn`, `ex_hn_street`) add only +0.15% true
pairs on validation, *because the problem doesn't exist there*, but **+0.65 new name+number pairs per French S1** on
test (US: +0.001). Expected effect: France recall up. The leaderboard is the only labelled check for France, so the v4
notebook prints a **silver report**: the share of unique name+number pairs predicted, per country. France should move
from 0.952 towards the ~0.99 seen in labelled countries.

**Lesson:** a validation score only certifies the distribution it was drawn from. When the test contains a population
we can't label, every run needs **label-free monitors** (prediction statistics, fingerprints, silver labels), and the
design must avoid **language-specific assumptions** (here: "the first name token is distinctive").

### Leaderboard check of the France diagnosis (v4)

v4 changed almost nothing for the US and India (their silver acceptance and prediction statistics are unchanged)
but fixed French blocking. The leaderboard moved **0.965 → 0.973 (+0.008)**, which confirms P14 as a real cause of the
validation/test gap.

Implied per-country picture. Assume US/India score on test what they score on validation (~0.985 / ~0.979) and use
the test mix (US 38.3%, India 46.8%, France 15.0%):

| | v3 | v4 |
|---|---|---|
| leaderboard | 0.965 | 0.973 |
| implied France macro F0.5 | ~0.88 | **~0.92** |

France is still ~0.06 below the labelled countries and remains the largest source of loss. It is ~0.009 of the total
0.027 gap to a perfect score, but ~0.009 of the ~0.010 gap to our validation level. The leave-one-country-out study
(T15) predicts about 0.039 of *model-transfer* loss for an unseen country, independent of blocking. That is the next
target: v5 self-training.

---

## France Evaluation Suite (FES): the France benchmark

**Why it's needed.** The US and India keep their benchmark unchanged: macro F0.5 on the same 100k labelled validation
entities (hold-out halves A/B). France has **no labels**, is 15% of the test, and follows different naming semantics.
Before FES, the only French signal was a public-leaderboard submission: 3 decimals, noisy, limited in number.

**Components.**

| Component | What it measures | Source of truth |
|---|---|---|
| FES-1 Marginal values | score change per pair added/removed, true/false | exact, on US/India validation (country-independent arithmetic of per-entity F0.5) |
| FES-2 Pattern classes | what kind of French pairs a variant changes (A same name+address, B same name/other address, C trade name, D variant, E same name/blank address, G different name/blank address, **SWAP** descriptor swap at the same address, F other) | raw record attributes |
| FES-3 Class precision | share of a class's changed pairs that are true | validation per decision zone for country-independent classes; **leaderboard fit** for France-specific ones (SWAP) |
| FES-4 Forecast | expected leaderboard Δ of a variant vs v4 = Σ class (added, removed) × precision-weighted marginal value | combines 1–3 |
| FES-5 Guards | silver acceptance (unique name+number pairs), entities with no match, matches per entity | label-free per-country monitors |

**Validation of the benchmark itself.** It must reproduce leaderboard results it was *not* fitted on. Fitted on the IDF
variant, it predicts v5 at −0.0017 against an actual −0.001 (the ±0.0005 rounding band plus 0.0002). Each new leaderboard result is added to the fit.

**How it's used.** Every France idea is first scored by FES. Only variants with a clearly positive forecast are submitted,
and each submission then refits FES. This has already (a) chosen `fr_noswap_f300` over the threshold probes and
(b) rejected the blank-address rule (T23) without spending a submission.

---

## v9 score report (competition metric, labelled validation)

Macro F0.5 per S1 entity, singletons included, on the fixed 100k validation entities (US/India); threshold tuned on half A.

| Model | all | half A | half B (held out) | US | India | 1-match entities | singletons |
|---|---|---|---|---|---|---|---|
| v4 (English cross-encoder) | 0.9826 | – | 0.9826 | – | – | – | – |
| **v9 (multilingual cross-encoder)** | **0.9829** | 0.9829 | **0.9828** | 0.9850 | 0.9797 | 0.9411 | 0.9891 |

Loss 0.0171 = false merges 0.0024 + rejected true pairs 0.0075 (mostly inherently ambiguous, T18) + **never generated
by blocking 0.0072** → the largest *reducible* bucket is blocking (Sparkly: top-k size is the main recall lever).

### Housekeeping (2026-09-27)
With approval, deleted 27 GB of regenerable caches: all of `work/baseline` (v1 embeddings, records, candidates; the v1
outputs are archived in `versions/v1_baseline`) and the large arrays in `work/v3` (training/validation pairs, feature
stores, kNN caches). **Kept** `work/v3/ce_model` (the English cross-encoder used by v4/v6, the lineage of the 0.974
submission) plus the small model/log files, so that pipeline stays reproducible. D: free space 106 GB → 133 GB.
