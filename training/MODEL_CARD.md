# Model Card — Parkinson's Early Detection (OpenNeuro ds005892)

**Date:** 2026-11-09 · **Task:** 3-class HC vs PD-NC vs PD-MCI (primary) +
binary HC vs PD (secondary) · **n = 55** (22 HC, 18 PD-NC, 15 PD-MCI) ·
**Protocol:** stratified 5-fold CV, seed 42, identical splits for every model ·
**Features:** Fisher-z functional connectivity, Schaefer 100 parcels (4950-dim
upper triangle; full 100×100 matrix for GCN/GAT) · **Runs:** training completed
2026-11-09 in 9.6 min on a CPU-only VM (2nd attempt; 1st attempt was killed by
a VM restart mid-binary — 3-class results were checkpointed to `metrics.json`
before the binary section on the re-run).

## Results — 3-class (chance: accuracy 0.333, balanced accuracy 0.333)

| Model | Accuracy | Balanced acc. | Macro-AUC | Macro-F1 |
|---|---|---|---|---|
| **GAT (best)** | 0.418 ± 0.187 | **0.401 ± 0.173** | 0.620 ± 0.183 | 0.362 ± 0.166 |
| GCN | 0.382 ± 0.176 | 0.357 ± 0.165 | 0.531 ± 0.131 | 0.315 ± 0.182 |
| Logistic regression | 0.345 ± 0.089 | 0.319 ± 0.097 | 0.540 ± 0.163 | 0.282 ± 0.101 |
| MLP | 0.327 ± 0.109 | 0.314 ± 0.103 | 0.495 ± 0.149 | 0.282 ± 0.109 |
| Random forest | 0.327 ± 0.073 | 0.291 ± 0.078 | 0.528 ± 0.084 | 0.247 ± 0.106 |
| XGBoost | 0.309 ± 0.073 | 0.280 ± 0.075 | 0.539 ± 0.094 | 0.249 ± 0.098 |
| 1D-CNN | 0.273 ± 0.100 | 0.259 ± 0.101 | 0.495 ± 0.142 | 0.238 ± 0.099 |
| SVM (RBF) | 0.273 ± 0.081 | 0.253 ± 0.089 | 0.594 ± 0.094 | 0.217 ± 0.078 |

Best model **GAT** confusion matrix (summed over folds, rows=true):
`[[15, 4, 3], [9, 2, 7], [5, 4, 6]]` — HC is recognised best (15/22);
PD-NC is the hardest class (only 2/18 correct, 9 confused as HC).

## Results — binary HC vs PD (top 3 models by 3-class balanced accuracy)

| Model | Accuracy | Balanced acc. | AUC | F1 |
|---|---|---|---|---|
| Logistic regression | 0.709 ± 0.106 | 0.683 ± 0.103 | **0.745 ± 0.127** | 0.680 ± 0.105 |
| GAT | 0.709 ± 0.068 | 0.681 ± 0.064 | 0.658 ± 0.122 | 0.668 ± 0.073 |
| GCN | 0.691 ± 0.109 | 0.673 ± 0.093 | 0.697 ± 0.135 | 0.659 ± 0.110 |

The binary task shows a real signal (AUC 0.66–0.75); the 3-class task is
near chance. Distinguishing PD-NC from PD-MCI from resting-state connectivity
alone is the hard part — cognitive staging is subtle in rs-fMRI.

## Why GAT won (3-class)

Graph attention over the brain-connectivity graph beat flat vector models:
attention weights let the model focus on the most informative connections
rather than treating all 4950 features equally. GCN is second for the same
reason. But the margin is small and the fold-to-fold std is large (±0.17),
so "best" is tentative at this sample size.

## Notable training observations

- **MLP barely trained:** early stopping fired at epochs 0–3 in every fold —
  validation loss never improved, so the MLP is effectively untrained.
  1D-CNN was similar (best epochs 2–29). Both underfit; they need different
  hyper-parameters, not more epochs.
- **GCN/GAT trained properly:** best epochs 33–154 (max 1000, patience 50).
- **Classical models are at/below chance on 3-class** — 4950 features vs 44
  training subjects per fold is a classic p ≫ n problem.

## Preprocessing (what the features actually are)

Raw BIDS only — no fMRIPrep derivatives exist for this release. Pipeline
(`training/preprocess.py`, 106.4 min for 55 subjects):
ANTs affine normalisation to MNI152 2 mm (BOLD-mean → T1 rigid, T1 → MNI
affine) → Nilearn Schaefer-100 parcel time series (6 mm smoothing, bandpass
0.01–0.1 Hz, TR 2 s, z-scored) → Pearson functional connectivity.

**Documented limitations of the preprocessing:**
- Motion correction skipped (measured 327 s/subject on this VM — ~5 h for
  the cohort before any registration). Head motion is NOT corrected.
- Affine-only normalisation (no nonlinear SyN) — less precise than
  fMRIPrep-style nonlinear warps.
- No confound regressors (no motion parameters in raw BIDS, no
  fMRIPrep confounds.tsv). No fieldmap-based distortion correction
  (fieldmaps lack `IntendedFor`; sub-MJF050 is the sole phase-encoding
  outlier, direction `i` vs `j-` for the other 54).
- sub-MJF011's T1w is 256×256×191 vs 192 for all others (handled fine by
  the affine registration).

## Limitations (read before quoting any number)

- **n = 55 is tiny.** A 5-fold test fold holds ~11 subjects (~4/class), so
  every metric has a huge standard deviation (±0.10–0.19). Nothing here is
  a publishable accuracy claim.
- **No external validation cohort** inside this dataset (single site,
  single scanner, Siemens TrioTim 3 T).
- **Confounds:** Control skews female (15F/7M) while PD-MCI skews male
  (11M/4F); mean ages differ (Control 70.7, PD-NC 65.9, PD-MCI 69.9).
  Age/sex could leak into the signal; they were NOT regressed out.
- **Research demo only — not a medical diagnosis.**

## 3D CNN — not run

Skipped deliberately, not failed: a 3D CNN on raw 4D volumes with n=55,
no GPU, affine-only normalisation and no motion correction would
overfit massively and take many CPU-hours for a result worse than the
graph models. If a GPU becomes available, a 3D CNN on fMRIPrep-preprocessed
data would be a fair comparison — but it needs a bigger cohort first.

## What would actually improve this

1. More subjects (PPMI or a second OpenNeuro cohort for external validation).
2. fMRIPrep preprocessing (motion correction, confounds, nonlinear warp,
   distortion correction).
3. Regressing out age/sex, or stratifying folds by them.
4. Hyper-parameter tuning for MLP/CNN (they underfit here) and for GAT.
5. Dynamic connectivity / graph features beyond static Pearson FC.

## Reproduce

```bash
cd training
./venv/bin/python preprocess.py --skip-motion   # ~2 h, CPU-only
./venv/bin/python train.py                     # ~10 min, CPU-only
```

Full per-fold metrics: `training/metrics.json`. Logs:
`training/logs/preprocessing.log`, `training/logs/training.log`.
