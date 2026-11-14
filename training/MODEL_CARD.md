# Model Card — Parkinson's Early Detection (OpenNeuro ds005892)

**Date:** 2026-11-10 (rounds 2–3) · **Task:** 3-class HC vs PD-NC vs PD-MCI
(primary) + binary HC vs PD (secondary) · **n = 55** (22 HC, 18 PD-NC,
15 PD-MCI) · **Protocol:** nested cross-validation — outer stratified
5-fold, seed 42, identical splits for every model; inner stratified
3-fold for all scaling, feature selection and hyper-parameter tuning, so
nothing leaks from a test fold into training · **Features:** Fisher-z
functional connectivity, Schaefer 100 parcels (4950-dim upper triangle;
full 100×100 matrix for GCN/GAT) and Schaefer 200 parcels (19,900-dim) ·
**Runs:** round 2 completed 2026-11-10; model weights saved
(`training/models_v2/`, `training/models_pooled/`).

## Results — round 2, 3-class, Schaefer-100 (chance: 0.333)

| Model | Accuracy | Balanced acc. | Macro-AUC | Macro-F1 |
|---|---|---|---|---|
| **Logistic regression (best)** | 0.527 ± 0.156 | **0.502 ± 0.161** | 0.579 ± 0.126 | 0.481 ± 0.174 |
| SVM (RBF) | 0.527 ± 0.068 | 0.481 ± 0.088 | 0.559 ± 0.040 | 0.453 ± 0.099 |
| Random forest | 0.455 ± 0.081 | 0.407 ± 0.078 | 0.569 ± 0.079 | 0.344 ± 0.108 |
| 1D-CNN | 0.382 ± 0.106 | 0.373 ± 0.058 | 0.525 ± 0.086 | 0.317 ± 0.061 |
| GAT | 0.345 ± 0.089 | 0.367 ± 0.054 | 0.514 ± 0.062 | 0.264 ± 0.123 |
| MLP | 0.327 ± 0.169 | 0.356 ± 0.179 | 0.562 ± 0.121 | 0.298 ± 0.170 |
| GCN | 0.364 ± 0.057 | 0.343 ± 0.012 | 0.558 ± 0.101 | 0.283 ± 0.095 |
| XGBoost | 0.345 ± 0.134 | 0.326 ± 0.126 | 0.467 ± 0.105 | 0.307 ± 0.109 |

Best model **logistic regression** confusion matrix (summed over folds,
rows=true): `[[15, 4, 3], [6, 10, 2], [6, 5, 4]]` — PD-NC is now
recognised far better than in round 1 (10/18 correct, up from 2/18).
Cleaning helped the 3-class task: logistic regression went from 0.319
to 0.502 balanced accuracy.

## Results — round 2, 3-class, Schaefer-200 (chance: 0.333)

| Model | Balanced acc. | Macro-AUC |
|---|---|---|
| Random forest (best) | 0.442 ± 0.080 | 0.585 |
| SVM (RBF) | 0.416 ± 0.072 | 0.608 |
| MLP | 0.400 ± 0.098 | 0.570 |
| Logistic regression | 0.391 ± 0.121 | 0.589 |
| XGBoost | 0.382 ± 0.152 | 0.586 |
| 1D-CNN / GCN / GAT | skipped (see below) | — |

The finer atlas does not beat Schaefer-100. The 1D-CNN, GCN and GAT runs
on Schaefer-200 were stopped by the operator: about 30 minutes per fold
on 19,900 features with inner-config scores at chance level (0.37–0.49),
so they were not going to beat the classical models. They are recorded
as skipped, with the reason, in `metrics_v2.json` — not hidden, and not
run later as if they had been.

## Results — round 2, two-stage classifier

Stage 1 HC vs PD, stage 2 PD-NC vs PD-MCI, logistic regression:
Schaefer-100 combined balanced accuracy 0.403 (stage 1: 0.595, stage 2:
0.517); Schaefer-200 combined 0.431 (stage 1: 0.562, stage 2: 0.575).
XGBoost two-stage is worse (0.247 / 0.359). The two-stage route does not
beat the direct 3-class model (0.502).

## Results — round 2, binary HC vs PD (the honest surprise)

| Atlas | Model | Balanced acc. | AUC |
|---|---|---|---|
| Schaefer-100 | SVM (RBF) | 0.509 ± 0.077 | 0.438 ± 0.072 |
| Schaefer-100 | Random forest | 0.446 ± 0.096 | 0.488 ± 0.128 |
| Schaefer-100 | Logistic regression | 0.443 ± 0.094 | 0.475 ± 0.142 |
| Schaefer-100 | 1D-CNN / GAT / MLP | 0.376–0.455 | 0.338–0.359 |
| **Schaefer-200** | **XGBoost (best binary overall)** | **0.578 ± 0.189** | **0.689 ± 0.194** |
| Schaefer-200 | Logistic regression | 0.491 ± 0.129 | 0.502 ± 0.192 |

Round 1 reported binary logistic regression AUC 0.745. After proper
cleaning that number collapses on Schaefer-100 (AUC 0.475). A diagnostic
(`training/binary_diagnostic.py`, results in
`training/binary_diagnostic.json`) pins down why:

- **A — round-1 method on round-2 features: AUC 0.480.** The cleaned
  features no longer carry the signal.
- **B — round-2 method on round-1 features: AUC 0.689.** The method is
  not the cause.
- **C — round-1 method on round-1 features: AUC 0.745.** The round-1
  number reproduces exactly — no label or split bug.
- **D — round-2 method on round-2 features: AUC 0.447.** Matches the
  round-2 binary result.
- **E — mean head motion (framewise displacement) alone as a PD score:
  AUC 0.523** (HC mean 0.271 mm, PD mean 0.292 mm). Raw head movement is
  barely predictive.

So the round-1 binary signal lived in motion and physiological noise
inside the connectivity — not in raw head movement, and not in brain
activity that survives cleaning. The 3-class signal is the real one: it
improved with cleaning. The binary task's best remaining model is
XGBoost on Schaefer-200 (AUC 0.689), which partly recovers the signal
with a non-linear model on the finer atlas.

## Results — pooled data, binary HC vs PD (n = 264, four datasets)

ds005892 (55) + PPMI via BrainNet (195) + Neurocon (41) + Tao Wu (40),
all Schaefer-100 functional connectivity, nested CV, classical models
only. Mixed together: SVM balanced accuracy 0.620 / AUC 0.678,
logistic regression 0.600 / 0.696, XGBoost 0.549 / 0.651, random forest
0.529 / 0.690. **Leave-one-source-out (test on a whole dataset never
seen in training): logistic regression 0.502 / AUC 0.492, SVM 0.448 /
0.433, random forest 0.490 / 0.465, XGBoost 0.512 / 0.451 — chance.**
The pooled model learns each scanner's fingerprint, not the disease.
Site harmonisation is the next step, not more of the same data. Full
numbers: `training/metrics_pooled.json`.

## Results — round 3, site harmonisation on pooled data (n = 264)

**Date:** 2026-11-10 · **Question:** does ComBat site harmonisation lift
leave-one-source-out (LOSO) above chance? **Answer: no — plainly, with
the numbers.** Same pooled features and same nested protocol as the
round-2 pooled benchmark (outer stratified 5-fold seed 42 for the
pooled scheme, 4 LOSO folds, inner 3-fold search; LR, SVM-RBF, random
forest, XGBoost). Three harmonisation variants, all on Schaefer-100
Fisher-z connectivity, diagnosis + age + sex preserved as covariates
where the method takes them:

- **ComBat (strict):** empirical-Bayes ComBat fit **only on each
  outer-train fold**; a held-out fold is transformed with its own
  batch's fitted parameters, and a batch never seen in training (the
  LOSO held-out site) gets **no correction at all** — that is the
  leak-free rule. Cross-checked against neuroCombat 0.2.12 on the full
  data: max difference 1.82e-07, so the implementation is correct.
- **ComBat transductive (upper bound):** batch-only design, fit on all
  subjects before CV. No labels anywhere in the fit, but the held-out
  site's data is seen — so it is an upper bound, never the headline.
- **Per-site z-scoring:** each site's own training subjects set the
  mean/std; an unseen LOSO site falls back to the global training
  mean/std.

| Scheme | Best model | Balanced acc. | AUC |
|---|---|---|---|
| Raw pooled 5-fold (round-2 baseline) | SVM | 0.620 | 0.678 |
| ComBat pooled 5-fold | LR | 0.585 | 0.651 |
| z-score pooled 5-fold | LR | 0.514 | 0.537 |
| ComBat-trans pooled 5-fold | RF | 0.507 | 0.532 |
| Raw LOSO (round-2 baseline) | XGB | 0.512 | 0.451 |
| **ComBat LOSO** | **SVM** | **0.536** | **0.503** |
| z-score LOSO | SVM | 0.511 | 0.487 |
| ComBat-trans LOSO | SVM | 0.525 | 0.472 |

Full per-model table (balanced accuracy / AUC): ComBat LOSO — LR
0.495/0.457, SVM 0.536/0.503, RF 0.498/0.440, XGB 0.504/0.487.
ComBat pooled — LR 0.585/0.651, SVM 0.549/0.653, RF 0.500/0.637, XGB
0.544/0.622. z-score LOSO — LR 0.456/0.471, SVM 0.511/0.487, RF
0.486/0.486, XGB 0.487/0.496. ComBat-trans LOSO — LR 0.491/0.489, SVM
0.525/0.472, RF 0.498/0.446, XGB 0.504/0.468 (XGB folds
0.485/0.481/0.552/0.500). Per-source LOSO folds (ds005892, Neurocon,
PPMI, Tao Wu held out, in that order), ComBat LR: 0.561/0.451/0.495/
0.475; ComBat SVM: 0.553/0.509/0.581/0.500. So the harmonised pooled
model applied to ds005892 — the home dataset — scores 0.55–0.56
balanced accuracy with AUC 0.54–0.57: still a coin flip, not a lift.

What this means, in plain words:

- **Harmonisation removed the pooled signal instead of revealing a
  disease signal.** Pooled 5-fold fell from 0.600–0.620 (raw) to
  0.585 (ComBat) and 0.514 (z-score) — and the transductive variant,
  which sees the test site, fell further to 0.457–0.507. The pooled
  score really was the scanners' fingerprints.
- **Even the upper bound is at chance.** ComBat-trans LOSO best is
  0.525 with AUC 0.472. If harmonisation that has seen the held-out
  site still cannot beat a coin flip, the cross-site disease signal
  in these static functional-connectivity features is essentially
  absent — the bottleneck is the features, not the harmonisation
  method.
- **Random forest collapsed** to predicting every subject as PD in
  nearly every scheme (balanced accuracy exactly 0.500, confusion
  `[[0, 72], [0, 192]]` on pooled) — it found nothing to split on
  after harmonisation.
- **GAT/GCN not run in round 3, deliberately:** with every classical
  model at chance and even the transductive upper bound at chance, a
  graph model on the same harmonised features has no signal to find
  that LR/SVM/XGB missed. It would cost hours of CPU for a foregone
  conclusion. Recorded as skipped, with the reason, not hidden.

**Weights saved** (`training/models_round3/`): `round3_combat_logistic_
regression_full.pkl` (best pooled harmonised model, LR, C=10, k=1000),
`round3_combat_svm_rbf_full.pkl` (best LOSO harmonised model, SVM,
C=0.1, gamma=0.001, k=500), `round3_combat_fit.pkl` (fitted ComBat
parameters — the classifier pickles alone cannot harmonise a new scan;
the batch means/scales live in the fit file). **Neither is a deployment
candidate:** LOSO is at chance for every variant, so no round-3 model
replaces the service weights. The weights decision from round 2 stays
open. Full numbers: `training/metrics_round3.json`; confusion-matrix
figures: `training/figures_round3/`; code: `training/train_round3.py`,
`training/refit_round3_loso_best.py`.

**Run notes:** the run took 88.8 min of compute across two runtime
restarts (12:45 and ~14:47 UTC); per-model checkpointing was added after
the first restart so a kill costs at most one model, and the resumed
runs reproduced earlier numbers exactly (e.g. z-score LOSO XGB fold 0:
0.485 before and after the restart).

## Results — round 4, richer features on the enlarged pool (n = 395)

**Date:** 2026-11-14 · **Question:** do richer features — dynamic
connectivity, graph measures, ALFF/fALFF, ReHo, structural MRI —
plus a bigger multi-site pool lift leave-one-source-out above chance,
and can cognition staging (NC / MCI / PDD) work? **Answer: no, on
both counts, with the numbers.** Same nested protocol as rounds 2–3
(outer stratified 5-fold seed 42 for the pooled scheme, LOSO over
whole sources, inner 3-fold search; LR, SVM-RBF, random forest,
XGBoost, plus a soft-voting ensemble of LR+SVM+RF).

**Data:** 7 sources, n = 395 binary (HC 101 / PD 294): ds000245 (45),
ds001354 (28), ds004392 (58, all PD), ds005892 (55), Neurocon (41),
PPMI-195 via BrainNet (128), Tao Wu (40). Cognition staging pool n =
78 PD (NC 18, MCI 51, PDD 9 — all 9 PDD male). ds006224 (5 PPMI
subjects, 3 sessions each) is excluded from training by design — all
5 already appear through the BrainNet PPMI-195 matrices — and is used
only as a test-retest probe. Features are 12 blocks (S/G/Dmean/Dstd/
A/ReHo at Schaefer-100 and -200, 79,054 features on the raw subset),
plus a structural block (T) for the struct arm; the screening arms
also covered dynamic + ALFF + multi-atlas combinations on n = 209 /
264.

**Read the LOSO numbers with one correction.** The ds004392 LOSO fold
is all-PD, so its balanced accuracy is plain accuracy (= PD recall)
with no AUC, and it inflates the LOSO mean wherever the model predicts
PD for nearly everyone (RF 1.000, XGB 0.966, SVM 0.948 on that fold in
the ComBat arm). The table below gives the LOSO mean as reported and,
in brackets, without that fold. The honest best is the arm with no
single-class fold at all.

| Arm (n) | Pooled 5-fold best | LOSO best (as reported) | LOSO best (excl. all-PD fold) |
|---|---|---|---|
| SG_all (395) | LR 0.641 | XGB 0.520 | XGB 0.501 |
| common_all (395) | LR 0.611 | XGB 0.534 | LR 0.541 |
| common_combat (395) | LR 0.619 | LR 0.592 | LR 0.561 |
| fusion_derived (209) | SVM 0.706 | ensemble 0.564 | — (no single-class fold) |
| fusion_raw (186) | LR 0.621 | SVM 0.611 | LR 0.548 |
| fusion_raw_struct (186) | LR 0.610 | ensemble 0.603 | LR 0.554 |

Full per-model tables (balanced accuracy / AUC): SG_all pooled — LR
0.641/0.695, SVM 0.629/0.689, ensemble 0.597/0.695, XGB 0.556/0.677,
RF 0.518/0.641; SG_all LOSO — XGB 0.520/0.518, RF 0.519/0.506,
ensemble 0.509/0.539, LR 0.509/0.555, SVM 0.464/0.516. common_all
pooled — LR 0.611/0.679, ensemble 0.598/0.667, SVM 0.598/0.620, XGB
0.558/0.667, RF 0.526/0.634; common_all LOSO — XGB 0.534/0.518, LR
0.513/0.494, ensemble 0.500/0.502, RF 0.492/0.489, SVM 0.425/0.494.
common_combat pooled — LR 0.619/0.658, SVM 0.575/0.655, XGB
0.527/0.656, RF 0.498/0.667; common_combat LOSO — LR 0.592/0.554, RF
0.575/0.552, SVM 0.571/0.549, XGB 0.553/0.501. fusion_derived pooled
— SVM 0.706/0.705, LR 0.671/0.759, ensemble 0.655/0.752, XGB
0.558/0.722, RF 0.515/0.733; fusion_derived LOSO (3 sources) —
ensemble 0.564/0.541, LR 0.541/0.558, XGB 0.538/0.518, SVM
0.507/0.503, RF 0.500/0.547. fusion_raw pooled — LR 0.621/0.705, SVM
0.606/0.666, ensemble 0.565/0.689, RF 0.518/0.683, XGB 0.508/0.666;
fusion_raw LOSO — SVM 0.611/0.574, ensemble 0.603/0.602, XGB
0.589/0.552, RF 0.586/0.594, LR 0.575/0.580. fusion_raw_struct pooled
— LR 0.610/0.695, SVM 0.606/0.667, ensemble 0.599/0.686, XGB
0.532/0.674, RF 0.505/0.675; fusion_raw_struct LOSO — ensemble
0.603/0.582, SVM 0.603/0.561, RF 0.586/0.551, LR 0.583/0.578.
Screening (before the final run): arm209_SGDAM LR pooled 0.740 / AUC
0.773, LOSO 0.541 — the pooled number is the highest in the whole
project and still collapses to chance across sites.

**Cognition staging (NC / MCI / PDD, n = 78, chance 0.333):** pooled
5-fold LR 0.570 / AUC 0.720, ensemble 0.472 / 0.737, SVM 0.344 /
0.675, XGB 0.340 / 0.625, RF 0.313 / 0.697. LOSO has only 2 folds
(ds004392 and ds005892 are the only sources with staging labels):
SVM 0.450, ensemble 0.325, XGB 0.286, RF 0.261, LR 0.215. The LOSO
SVM's 0.450 is a majority-class artefact — its confusion matrix is
`[[0, 13, 5], [0, 48, 3], [0, 9, 0]]`: it calls almost every subject
MCI (recall 0.94 MCI, 0.00 NC, 0.00 PDD). It stages nothing.

**Sex-stratified accuracy (cognition, dedicated re-run of the
final_cog splits with out-of-fold predictions recorded,
`training/sex_stratified_cog.py`):** pool is 57 men / 21 women (NC
11M/7F, MCI 37M/14F, PDD 9M/0F — PDD is all male, so no female PDD
accuracy exists). Pooled LR: men 0.597, women 0.571; pooled SVM: men
0.561, women 0.524; LOSO SVM: men 0.614, women 0.619; LOSO LR: men
0.316, women 0.286. No meaningful sex gap — the models are equally
(un)informative for men and women, and the gap is within noise at n =
21 women. (The re-run's pooled out-of-fold balanced accuracies — LR
0.555, SVM 0.354 — differ slightly from the metrics file's
mean-of-fold values because they pool all out-of-fold predictions
instead of averaging per-fold values.)

**ds006224 test-retest probe (`training/probe_6224.json`):**
connectivity is stable within a person (mean pairwise r 0.592 across
15 session pairs) vs between people (r 0.418, 10 pairs) — the pipeline
itself is stable. But the best binary model's PD probabilities flip
across sessions for 3 of 5 subjects (sub-3124: 0.401 / 0.634 / 0.500;
sub-3125: 0.486 / 0.178 / 0.517; sub-3130: 0.738 / 0.800 / 0.212);
only sub-3120 (0.874 / 0.836 / 0.974) and sub-3134 (0.564 / 0.816 /
0.692) agree across all three sessions. And these five people are in
the training data via PPMI-195, so this is the optimistic case. The
model is not stable on the same brain scanned three times.

What this means, in plain words:

- **Richer features did not help either.** Dynamic connectivity,
  graph measures, ALFF/fALFF, ReHo and structural features lift the
  pooled score (best 0.706, and 0.740 in screening) but the
  leave-one-source-out number stays at 0.50–0.56 — the same place it
  has been since round 2. The pooled gain is still the scanners'
  fingerprints, now with more features to fingerprint with.
- **ComBat still does not help.** The final ComBat arm's LOSO LR is
  0.592 as reported, 0.561 without the all-PD fold — the same as the
  unharmonised arms. Two rounds of harmonisation, same answer.
- **Cognition staging does not work on this data.** The only
  above-chance LOSO number (SVM 0.450) comes from predicting MCI for
  everyone, and the pooled best (LR 0.570) is on n = 78 with 2 LOSO
  folds. There is no staging signal to stage with.
- **The 90% target was not reached on any honest test.** The best
  honest numbers are 0.56 binary LOSO and 0.57 3-class pooled. The
  bottleneck is the cross-site disease signal in resting-state
  connectivity at this sample size, not the model family, the
  harmonisation, or the feature list.

**Weights saved** (`training/models_round4/`): `final_binary_*_full.pkl`
(LR / SVM / RF refits of the picked arm, final_fusion_raw; SVM is the
picked model, C=1.0, gamma=scale, k=1000) and `final_cog_*_full.pkl`
(same for the cognition pick, final_cog SVM, C=1.0, gamma=0.001,
k=500), plus `best_binary.json` / `best_cog.json` records. **Neither
is a deployment candidate:** LOSO is at chance and the probe shows the
predictions flip across sessions, so no round-4 model replaces the
service weights. The weights decision from round 2 stays open. Full
numbers: `training/metrics_round4.json`; confusion-matrix figures:
`training/figures_round4/` (121 figures); probe:
`training/probe_6224.json`; sex-stratified re-run:
`training/sex_stratified_cog.json`; code:
`training/train_r4_final.py`, `training/finalize_r4.py`,
`training/probe_6224.py`, `training/sex_stratified_cog.py`.

**Run notes:** the final run took 12–14 Nov across 8 VM restarts
(#37–#44); per-fold checkpointing meant each restart cost at most one
fold, and resumed runs reproduced earlier numbers exactly. Two real
failures were fixed, not worked around: (1) the final_cog LOSO
XGBoost fold that holds out the source with all 18 NC subjects
failed all 48 fits because XGBoost saw labels [1, 2] with no class 0
— fixed in `train_r4.py` by remapping each fold's train labels to
contiguous 0-based ints and mapping predictions back. (2) The
finalize step could not finish: its full XGB grid (80 fits on 79,054
features) runs over 2 hours, longer than the VM's ~2-hour restart
window, so two attempts died in it — the XGB full refit is skipped
in `finalize_r4.py` with a log line (XGB is not the picked model for
either task; its CV numbers are already in the metrics). The probe
then failed on the first 200-parcel block because `probe_6224.py`
only built 100-parcel blocks — the 200-parcel branches were added
and the probe re-run. The picked binary model is the
final_fusion_raw SVM by LOSO balanced accuracy as reported (0.611);
the honest best is 0.564 (derived-fusion ensemble) — see the table
above — and the record keeps the as-picked model, with the correction
stated here and in `results.json`, not silently re-picked.

## Results — round 4B, time-series statistics on the unified blocks (n = 395)

**Date:** 2026-11-14 · **Question:** do per-parcel time-series
statistics and per-subject brain states — features the round-4 blocks
do not contain — lift leave-one-source-out above chance, do five extra
model families help, and does unifying ds005892's pipeline recover
the accuracy lost to the old v2 pipeline? **Answer: no, on all three,
with the numbers.** Same nested protocol as round 4 (outer stratified
5-fold seed 42 pooled, LOSO over whole sources, inner 3-fold search;
LR, SVM-RBF, random forest, XGBoost, plus a soft-voting ensemble and
five extra models: extra trees, Gaussian process, balanced bagging,
calibrated LR, stacking).

**Data:** same 7 sources, n = 395 binary (HC 101 / PD 294). The B
block is 614 features per subject: per-parcel time-series statistics
(std, skew, kurtosis, lag-1 autocorrelation, Hurst exponent, spectral
entropy at Schaefer-100) plus per-subject k-means brain states (state
occupancy, transition counts, dwell times). Built by
`training/build_r4b_blocks.py` from the unified features; the build
completed 395/395 subjects on 2026-11-14 (the 57 ds004392 subjects
missing since 12 Nov were filled in by the chain's B-block build).
Labels: `training/labels_r4b.csv`, complete.

**Read the LOSO numbers with the same correction as round 4.** The
ds004392 LOSO fold is all-PD, so its balanced accuracy is plain
accuracy (= PD recall) and it inflates the LOSO mean wherever the
model predicts PD for nearly everyone. The table below gives the LOSO
mean as reported and, in brackets, without that fold. Arms with no
ds004392 fold (b209, b264) need no correction.

| Arm (n) | Pooled 5-fold best | LOSO best (as reported) | LOSO best (excl. all-PD fold) |
|---|---|---|---|
| b186_raw_B (186) | GP 0.610 | SVM 0.617 | SVM 0.523 |
| b209_fusion_B (209) | stacking 0.718 | balanced_bagging 0.591 | — (no single-class fold) |
| b264_SG_unified (264) | LR 0.633 | XGB 0.517 | — (no single-class fold) |
| b395_common_B (395) | SVM 0.617 | RF 0.551 | RF 0.502 |
| b395_fusion_B (395) | SVM 0.620 | XGB 0.507 | RF 0.493 |
| bcog_B (78) | LR 0.427 | LR 0.332 | RF/XGB 0.433 (one fold only) |

Full per-model tables (balanced accuracy / AUC): b186 pooled —
GP 0.610/0.489, SVM 0.587/0.723, ensemble 0.582/0.696, XGB
0.561/0.692, LR 0.557/0.701, RF 0.546/0.699, extra_trees 0.535/0.709
(7 models — the manual full run was killed before balanced bagging,
calibrated LR and stacking finished); b186 LOSO, honest —
LR 0.528, SVM 0.523, XGB 0.501, RF 0.500 (4 fast models only).
b209 pooled — stacking 0.718/0.750, LR 0.671/0.760, balanced_bagging
0.666/0.767, ensemble 0.655/0.756, calibrated_lr 0.643/0.749, SVM
0.623/0.758, XGB 0.562/0.719, GP 0.561/0.483, RF 0.525/0.734,
extra_trees 0.509/0.739; b209 LOSO (3 sources, no correction needed)
— balanced_bagging 0.591/0.548, LR 0.584/0.560, GP 0.565/0.500,
stacking 0.564/0.558, calibrated_lr 0.554/0.534, SVM 0.535/0.559,
XGB 0.532/0.556, ensemble 0.520/0.557, RF 0.518/0.589, extra_trees
0.493/0.517. b264 pooled — LR 0.633/0.705, SVM 0.611/0.695, XGB
0.608/0.699, RF 0.511/0.678; b264 LOSO — XGB 0.517/0.507, SVM
0.495/0.480, LR 0.485/0.532, RF 0.468/0.459. b395_common pooled —
SVM 0.617/0.665, LR 0.573/0.646, XGB 0.562/0.655, RF 0.537/0.637;
b395_common LOSO, honest — XGB 0.503, RF 0.502, LR 0.498, SVM 0.472.
b395_fusion pooled — SVM 0.620/0.718, LR 0.610/0.694, RF 0.577/0.678,
XGB 0.553/0.700; b395_fusion LOSO, honest — RF 0.493, SVM 0.489,
XGB 0.479, LR 0.478. (LOSO AUC means are not quoted for the arms
with the all-PD fold: the stored mean is NaN because that fold has
no AUC.)

**The pipeline-mismatch test is clean — and negative.**
`b264_SG_unified` reruns the finisher's `arm264_SG` on the same 264
subjects and the same S+G blocks, with only ds005892 reprocessed
through the unified pipeline instead of the old v2 pipeline. Pooled
best 0.633 (LR) vs 0.601 (SVM), AUC 0.705 vs 0.668; LOSO best 0.517
(XGB) vs 0.498 (LR). On the ds005892 LOSO fold itself — the fold
that would show a pipeline recovery — the deltas are LR +0.015
(0.500 → 0.515), XGB +0.038 (0.485 → 0.523), SVM 0.000, RF 0.000.
Unification recovered a small amount on pooled CV and nothing that
matters on the honest test. The cross-site drop is a site effect,
not a pipeline effect.

**Cognition staging, B block (NC / MCI / PDD, n = 78, chance
0.333):** pooled LR 0.427 / AUC 0.632, XGB 0.393 / 0.619, SVM 0.384 /
0.677, RF 0.337 / 0.686. LOSO has 2 folds; without the ds004392 fold
only the ds005892 fold remains, so the "honest" numbers (RF 0.433,
XGB 0.433, LR 0.400, SVM 0.400) are a single fold of 55 subjects —
not a result. Staging still does not work.

**Run notes:** fast screening finished 2026-11-12 for all 6 arms (4
classical models each). The monolithic full run was killed by the
early-stop guard on 2026-11-14 at 14:47 UTC, 16 seconds after
screening finished — fast LOSO was at chance for 4 of 6 arms, so
only the two survivors (b209_fusion_B 0.584, b186_raw_B 0.617 as
reported) were given manual full runs. Those ran 14:52–16:04 UTC and
were killed by VM restart 45: b209 finished all 10 models in both
schemes; b186 finished 7 models pooled and 4 LOSO. Relaunching them
is blocked by `training/es_guard_4b.sh`, which kills any non-`--fast`
`train_r4b.py` process — stopping the guard is the coordinator's
call, and the numbers above are exactly what the partial runs
contain. S400 and 6224_not1w/5906 arms are not in this round: they
need a `train_r4b.py` extension that was never written.

What this means, in plain words:

- **Time-series statistics did not help.** The B block's best honest
  LOSO is 0.591 on the n = 209 derived-fusion subset — the same
  subset where the finisher's derived fusion got 0.564. On the full
  n = 395 pool the B arms are 0.49–0.50 honest. More features, same
  wall.
- **The extra model families did not help either.** Stacking wins
  pooled (0.718) and collapses LOSO (0.564); balanced bagging wins
  honest LOSO (0.591) on 3 folds with per-fold values 0.613 / 0.510 /
  0.650 — noisy at 3 folds, and its pooled number (0.666) is
  unremarkable. Gaussian process takes the b186 pooled best (0.610)
  with an AUC of 0.489, below chance — a balanced-accuracy win with
  no ranking signal behind it.
- **The 90% target was not reached.** Best honest binary LOSO in
  round 4B is 0.591; best honest full-pool LOSO is 0.502.

**Weights:** none saved for round 4B — no arm beats the finisher's
honest best by a margin worth refitting, and the best 4B number
(0.591, balanced bagging on 3 LOSO folds) is too thin to pick a
deployment candidate from. Full numbers:
`training/metrics_round4.json` (4B arms merged 2026-11-14) and
`training/metrics_round4b.json`; confusion-matrix figures:
`training/figures_round4/` (122 new 4B/4C figures); code:
`training/build_r4b_blocks.py`, `training/train_r4b.py`,
`training/update_results_round4bc.py`.

## Results — round 4C, site-invariant transforms (fast screening only, n = 395)

**Date:** 2026-11-14 · **Question:** do site-invariant transforms
of the common blocks — rank, stable-rank, rank-Gaussian,
age-residualised, per-site z-score, lag-1 precision and partial
correlation, a LOSO feature selector, a block stacker — remove the
scanner fingerprint and let the disease signal generalise?
**Answer: no.** Fast screening only (4 classical models per arm);
the full run was skipped by the early-stop rule because fast LOSO
was at chance. Same nested protocol as round 4.

| Arm (n = 395) | Pooled 5-fold best | LOSO best (as reported) | LOSO best (excl. all-PD fold) |
|---|---|---|---|
| c_ageresid_common | SVM 0.611 | XGB 0.514 | RF 0.524 |
| c_lagprec | LR 0.607 | RF 0.569 | SVM 0.533 |
| c_lagprec_S | SVM 0.592 | XGB 0.542 | SVM 0.522 |
| c_loso_select | — (LOSO only) | loso_selected 0.449 | 0.498 |
| c_rank_common | SVM 0.593 | RF 0.508 | SVM 0.505 |
| c_rankgauss_common | LR 0.538 | LR 0.592 | LR 0.523 |
| c_stable_common | SVM 0.622 | LR 0.507 | LR 0.520 |
| c_stack_loso | block_stack 0.477 | block_stack 0.476 | 0.492 |
| c_zscore_common | SVM 0.553 | SVM 0.588 | SVM 0.525 |

Full per-model tables (balanced accuracy / AUC): c_ageresid pooled
— SVM 0.611/0.653, LR 0.579/0.625, XGB 0.557/0.657, RF 0.554/0.616;
LOSO honest — RF 0.524, XGB 0.508, SVM 0.490, LR 0.472. c_lagprec
pooled — LR 0.607/0.703, SVM 0.575/0.710, XGB 0.549/0.685, RF
0.517/0.662; LOSO honest — SVM 0.533, LR 0.521, XGB 0.503, RF
0.497. c_lagprec_S pooled — SVM 0.592/0.701, LR 0.577/0.693, XGB
0.545/0.665, RF 0.532/0.652; LOSO honest — SVM 0.522, XGB 0.514,
RF 0.494, LR 0.490. c_rank pooled — SVM 0.593/0.651, LR 0.576/0.587,
XGB 0.540/0.585, RF 0.525/0.645; LOSO honest — SVM 0.505, LR 0.500,
RF 0.474, XGB 0.469. c_rankgauss pooled — LR 0.538/0.568, SVM
0.511/0.591, RF 0.503/0.616, XGB 0.497/0.607; LOSO honest — LR
0.523, SVM 0.500, RF 0.493, XGB 0.493. c_stable pooled — SVM
0.622/0.667, LR 0.576/0.627, XGB 0.573/0.558, RF 0.570/0.599; LOSO
honest — LR 0.520, SVM 0.501, RF 0.468, XGB 0.467. c_zscore pooled —
SVM 0.553/0.608, XGB 0.531/0.576, LR 0.528/0.564, RF 0.508/0.596;
LOSO honest — SVM 0.525, LR 0.515, XGB 0.506, RF 0.502.
c_loso_select LOSO — 0.449 as reported, 0.498 honest (the ds004392
fold was 0.155, below chance, so the correction raises it).
c_stack_loso — block_stack 0.477 pooled, 0.476 LOSO as reported,
0.492 honest.

Two things in this table are worth spelling out. First, the
rank-Gaussian and z-score arms look like they beat their own pooled
numbers on LOSO (rankgauss LR: pooled 0.538, LOSO 0.592 as reported;
zscore SVM: pooled 0.553, LOSO 0.588) — that inversion is entirely
the all-PD fold: rankgauss scores a perfect 1.000 on it with all
four models, zscore scores 1.000 (RF), 0.966 (SVM) and 0.948 (LR,
XGB). After the correction both arms sit at 0.52–0.53. Second, the
LOSO selector and the block stacker — the two arms that most
directly try to pick cross-site signal — are 0.498 and 0.492 honest.
There is no feature subset in these blocks that generalises across
sites.

**Run notes:** fast screening done 2026-11-14 16:36 UTC (the first
attempt failed at 16:16 on an atomic-replace race in the partial
metrics file and was relaunched). The full run was skipped by the
same early-stop rule as 4B. Full numbers:
`training/metrics_round4.json` (4C arms merged 2026-11-14) and
`training/metrics_round4c.json`; figures:
`training/figures_round4/`; code: `training/build_r4c_blocks.py`,
`training/train_r4c.py`, `training/update_results_round4bc.py`.

## Why logistic regression won (3-class, round 2)

With 55 subjects and 4950 features, the simplest regularised linear
model generalises best once the noise is cleaned out. The neural and
graph models have far more parameters than the data can support, and
their inner-config search shows it: on Schaefer-200 they score at chance
before training even finishes. Round 1's GAT win (0.401) was partly the
noise — on cleaned features GAT falls to 0.367 while logistic regression
rises to 0.502.

## Notable training observations

- **MLP barely trained:** early stopping fired at epochs 0–3 in several
  folds — validation loss never improved, so the MLP is effectively
  untrained in those folds. 1D-CNN was similar. Both underfit; they need
  different hyper-parameters, not more epochs.
- **Classical models are the honest baseline here** — with n=55, a tuned
  regularised linear model beats every neural model on the 3-class task.
- **Fold-to-fold variation is huge** (±0.08–0.19 on balanced accuracy).
  Any single number here is a noisy estimate, not a claim.
- **Determinism check:** every interrupted run reproduced its earlier
  numbers exactly on resume (same seed, same splits), which is how the
  restarts were caught and verified rather than silently re-run.

## Preprocessing (what the features actually are)

Round 2 (`training/preprocess_v2.py`, ~171 min for 55 subjects): raw BIDS
only — no fMRIPrep derivatives exist for this release. ANTs affine
normalisation to MNI152 2 mm (BOLD-mean → T1 rigid, T1 → MNI affine;
the spatial transform is applied in blocks of 25 volumes because the
full 4D transform spiked to ~3.7 GB and the VM's memory limit is ~6 GB)
→ **motion correction on** (Friston-24 motion regressors,
framewise-displacement scrubbing at 0.5 mm, high-variance confounds) →
Nilearn Schaefer-100 and Schaefer-200 parcel time series (6 mm smoothing,
bandpass 0.01–0.1 Hz, TR 2 s, z-scored) → Pearson functional connectivity.

**Remaining limitations of the preprocessing:**
- Affine-only normalisation (no nonlinear SyN) — less precise than
  fMRIPrep-style nonlinear warps.
- No fieldmap-based distortion correction (fieldmaps lack `IntendedFor`;
  sub-MJF050 is the sole phase-encoding outlier, direction `i` vs `j-`
  for the other 54).
- sub-MJF011's T1w is 256×256×191 vs 192 for all others (handled fine by
  the affine registration).
- Age/sex were NOT regressed out. Control skews female (15F/7M) while
  PD-MCI skews male (11M/4F); mean ages differ (Control 70.7, PD-NC 65.9,
  PD-MCI 69.9). Age/sex could leak into the signal.

## Round 1 results (archived 2026-11-09, `training/metrics_v1.json`)

Plain stratified 5-fold CV, seed 42, no motion correction, no confound
regressors, Schaefer-100 only. Best 3-class: GAT balanced accuracy
0.401 ± 0.173 (accuracy 0.418, AUC 0.620); logistic regression 0.319.
Binary HC vs PD: logistic regression AUC 0.745 ± 0.127 (balanced acc.
0.683 ± 0.103). The binary number is now understood as partly noise —
see the diagnostic above.

## Limitations (read before quoting any number)

- **n = 55 is tiny.** A 5-fold test fold holds ~11 subjects (~4/class), so
  every metric has a huge standard deviation (±0.10–0.19). Nothing here is
  a publishable accuracy claim.
- **No external validation cohort** inside this dataset (single site,
  single scanner, Siemens TrioTim 3 T). The pooled leave-one-source-out
  test shows a model trained on mixed sites does not transfer to an
  unseen site at all.
- **Confounds:** age/sex skew (above) was not regressed out.
- **Research demo only — not a medical diagnosis.**

## 3D CNN — not run

Skipped deliberately, not failed: a 3D CNN on raw 4D volumes with n=55,
no GPU, affine-only normalisation would overfit massively and take many
CPU-hours for a result worse than the graph models. If a GPU becomes
available, a 3D CNN on fMRIPrep-preprocessed data would be a fair
comparison — but it needs a bigger cohort first.

## What would actually improve this

1. ~~Site harmonisation (e.g. ComBat) before pooling~~ — **tried in
   round 3, did not close the gap.** LOSO stayed at chance for ComBat,
   per-site z-scoring, and even the transductive upper bound.
   ~~Richer features (dynamic connectivity, graph measures,
   structural + diffusion MRI alongside the functional data)~~ —
   **tried in round 4, did not close the gap either.** Pooled best
   rose to 0.706 (0.740 in screening) but LOSO stayed at 0.50–0.56,
   ComBat still did not help, and cognition staging did not work. The
   bottleneck is now the cross-site disease signal in resting-state
   connectivity at this sample size, not site alignment, the model
   family, or the feature list. What is left: a model that learns a
   site-invariant representation from a much larger multi-site cohort,
   or a different signal altogether (structural + diffusion MRI,
   task fMRI, or a longitudinal design).
2. More subjects from a single site/protocol, or PPMI access (needs the
   user to register and sign a data-use agreement — see
   `data/DATASETS.md` for the exact steps).
3. Regressing out age/sex, or stratifying folds by them.
4. fMRIPrep preprocessing (nonlinear warp, distortion correction).
5. Hyper-parameter tuning for MLP/CNN (they underfit here) and for GAT.

## Reproduce

```bash
cd training
./venv/bin/python preprocess_v2.py          # ~3 h, CPU-only
./venv/bin/python train_v2.py               # nested CV, both atlases
./venv/bin/python train_pooled.py --skip-torch   # pooled binary benchmark
./venv/bin/python binary_diagnostic.py      # the binary AUC diagnostic
./venv/bin/python train_round3.py           # ComBat / z-score, pooled + LOSO
./venv/bin/python refit_round3_loso_best.py # LOSO-best refit + ComBat fit
./venv/bin/python update_results_round3.py  # merge round 3 into results.json
./venv/bin/python train_r4_final.py         # round 4 final arms, pooled + LOSO
./venv/bin/python finalize_r4.py            # round 4 refits, figures, probe
./venv/bin/python sex_stratified_cog.py     # sex-stratified cognition re-run
./venv/bin/python update_results_round4.py  # merge round 4 into results.json
```

Full per-fold metrics: `training/metrics.json` (round 2) and
`training/metrics_v2.json` (per-atlas detail),
`training/metrics_pooled.json` (pooled),
`training/metrics_round3.json` (round 3, incl. saved-model records),
`training/metrics_round4.json` (round 4 final + screening arms),
`training/sex_stratified_cog.json` (sex-stratified cognition),
`training/probe_6224.json` (ds006224 test-retest probe),
`training/binary_diagnostic.json` (diagnostic). Logs:
`training/logs/preprocess_v2_a.log`, `training/logs/preprocess_v2_b.log`,
`training/logs/training_v2.log`, `training/logs/training_pooled.log`,
`training/logs/training_round3.log`,
`training/logs/training_round4_final.log`,
`training/logs/finalize_r4.log`.
