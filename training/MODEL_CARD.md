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
   per-site z-scoring, and even the transductive upper bound. The
   bottleneck is now the features/signal itself, not site alignment.
   What is left: richer features (dynamic connectivity, graph measures,
   structural + diffusion MRI alongside the functional data), or a
   model that learns a site-invariant representation from a much larger
   multi-site cohort.
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
```

Full per-fold metrics: `training/metrics.json` (round 2) and
`training/metrics_v2.json` (per-atlas detail),
`training/metrics_pooled.json` (pooled),
`training/metrics_round3.json` (round 3, incl. saved-model records),
`training/binary_diagnostic.json` (diagnostic). Logs:
`training/logs/preprocess_v2_a.log`, `training/logs/preprocess_v2_b.log`,
`training/logs/training_v2.log`, `training/logs/training_pooled.log`,
`training/logs/training_round3.log`.
