# Model Card — Parkinson's Early Detection (OpenNeuro ds005892)

**Date:** 2026-11-10 (round 2) · **Task:** 3-class HC vs PD-NC vs PD-MCI
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

1. Site harmonisation (e.g. ComBat) before pooling, or a model that is
   explicitly blind to site — the leave-one-source-out result says this
   is the bottleneck, not the model class.
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
```

Full per-fold metrics: `training/metrics.json` (round 2) and
`training/metrics_v2.json` (per-atlas detail),
`training/metrics_pooled.json` (pooled),
`training/binary_diagnostic.json` (diagnostic). Logs:
`training/logs/preprocess_v2_a.log`, `training/logs/preprocess_v2_b.log`,
`training/logs/training_v2.log`, `training/logs/training_pooled.log`.
