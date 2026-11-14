# Parkinson's Early Detection — rs-fMRI (OpenNeuro ds005892)

Early detection of Parkinson's disease from resting-state fMRI: 3-class classification
(HC vs PD-NC vs PD-MCI) on OpenNeuro ds005892 v1.0.0 (Kemp et al. 2025; 55 participants:
22 HC, 18 PD-NC, 15 PD-MCI).

## Website

Next.js 14 (App Router) static export, in `website/`. Deploy to Vercel from this repo.

```bash
cd website
npm install
npm run build   # outputs to out/
```

## Results

### Round 2 — completed 2026-11-10

Nested cross-validation (outer stratified 5-fold, seed 42; inner 3-fold
for all scaling, feature selection and tuning — nothing leaks across the
outer folds), motion correction + Friston-24 regressors + framewise
displacement scrubbing (FD > 0.5 mm) + high-variance confounds, Schaefer
100- and 200-parcel atlases, two-stage classifier, model weights saved
(`training/models_v2/`, `training/models_pooled/`). Full per-fold metrics:
`training/metrics.json` (v1 archived as `training/metrics_v1.json`); model
card: `training/MODEL_CARD.md`. Scores are means ± std across folds.

**3-class HC vs PD-NC vs PD-MCI, Schaefer-100** (chance: 0.333):

| Model | Accuracy | Balanced acc. | AUC | F1 |
|---|---|---|---|---|
| Logistic regression (best) | 0.527 ± 0.156 | 0.502 ± 0.161 | 0.579 ± 0.126 | 0.481 ± 0.174 |
| SVM (RBF) | 0.527 ± 0.068 | 0.481 ± 0.088 | 0.559 ± 0.040 | 0.453 ± 0.099 |
| Random forest | 0.455 ± 0.081 | 0.407 ± 0.078 | 0.569 ± 0.079 | 0.344 ± 0.108 |
| 1D-CNN | 0.382 ± 0.106 | 0.373 ± 0.058 | 0.525 ± 0.086 | 0.317 ± 0.061 |
| GAT | 0.345 ± 0.089 | 0.367 ± 0.054 | 0.514 ± 0.062 | 0.264 ± 0.123 |
| MLP | 0.327 ± 0.169 | 0.356 ± 0.179 | 0.562 ± 0.121 | 0.298 ± 0.170 |
| GCN | 0.364 ± 0.057 | 0.343 ± 0.012 | 0.558 ± 0.101 | 0.283 ± 0.095 |
| XGBoost | 0.345 ± 0.134 | 0.326 ± 0.126 | 0.467 ± 0.105 | 0.307 ± 0.109 |

Schaefer-200 3-class (best: random forest 0.442, SVM 0.416, MLP 0.400,
logistic regression 0.391, XGBoost 0.382) does not beat Schaefer-100.
The 1D-CNN/GCN/GAT runs on Schaefer-200 were stopped deliberately: about
30 minutes per fold on 19,900 features with inner-config scores at chance
level, so they were not going to beat the classical models (recorded as
skipped, with the reason, in the metrics file).

Two-stage (stage 1 HC vs PD, stage 2 PD-NC vs PD-MCI), logistic
regression: Schaefer-100 combined balanced accuracy 0.403 (stage 1 0.595,
stage 2 0.517); Schaefer-200 combined 0.431 (stage 1 0.562, stage 2 0.575).
The two-stage route is worse than the direct 3-class model (0.502).

**Binary HC vs PD — the honest surprise:** on Schaefer-100 every model
falls to chance after proper cleaning (best AUC: random forest 0.488,
logistic regression 0.475; SVM balanced accuracy 0.509, AUC 0.438).
Round 1 reported logistic regression AUC 0.745, but a diagnostic
(`training/binary_diagnostic.py`) shows that signal lived in the
uncleaned data: round-1 method on round-2 features gives AUC 0.480,
while round-2 method on round-1 features gives AUC 0.689, and the
round-1 number reproduces exactly (AUC 0.745) on the old features. Head
motion alone is barely predictive (mean framewise displacement as a
score: AUC 0.523), so the removed signal was motion/physiological noise
inside the connectivity, not raw head movement. The best binary model in
round 2 is XGBoost on Schaefer-200 (balanced accuracy 0.578, AUC 0.689).
Cleaning helped the 3-class task (logistic regression 0.319 → 0.502;
PD-NC correct 2/18 → 10/18) while exposing that the binary signal was
partly noise.

**Pooled data (n=264, four datasets: ds005892 + PPMI-195 + Neurocon-41 +
Tao Wu-40, binary HC vs PD):** mixed together, SVM balanced accuracy
0.620 / AUC 0.678 and logistic regression 0.600 / 0.696. But tested on a
whole dataset it has never seen (leave-one-source-out), every model drops
to chance: logistic regression 0.502 / AUC 0.492, SVM 0.448 / 0.433,
random forest 0.490 / 0.465, XGBoost 0.512 / 0.451. The pooled model
learns each scanner's fingerprint, not the disease — site harmonisation
is the next step, not more of the same data. Full pooled metrics:
`training/metrics_pooled.json`.

### Round 3 — completed 2026-11-10 (site harmonisation, pooled n=264)

The harmonisation round: ComBat (fit only on training folds, unseen
sites left uncorrected), per-site z-scoring, and a transductive
ComBat upper bound (batch-only fit on all data, no labels), same
nested protocol and models as the pooled benchmark. **It did not
work.** Best leave-one-source-out balanced accuracy after
harmonisation is 0.536 (ComBat SVM, AUC 0.503 — a coin flip) vs 0.512
unharmonised; z-score best is 0.511, transductive best 0.525. Pooled
5-fold actually fell after harmonisation (ComBat LR 0.585 vs raw
0.600–0.620; z-score LR 0.514; transductive LR 0.457), and random
forest collapsed to predicting every subject as PD in nearly every
scheme. Even harmonisation that has seen the held-out site cannot
beat chance — the cross-site disease signal in static functional
connectivity is essentially absent, and the pooled score was the
scanners' fingerprints. The harmonised pooled model applied to
ds005892 scores 0.55–0.56 balanced accuracy, AUC 0.54–0.57: still
chance. GAT/GCN were not run — with every classical model and even
the transductive upper bound at chance, a graph model on the same
features has no signal to find. Weights saved
(`training/models_round3/`: ComBat LR full refit, ComBat SVM full
refit, fitted ComBat parameters); **neither is a deployment
candidate** — no round-3 model replaces the service weights. Full
numbers: `training/metrics_round3.json`; model card:
`training/MODEL_CARD.md`. The next bottleneck is the features
themselves (dynamic connectivity, graph measures, structural +
diffusion data), not site alignment.

### Round 4 — completed 2026-11-14 (richer features, enlarged pool n=395)

The richer-features round: dynamic connectivity, graph measures,
ALFF/fALFF, ReHo and structural MRI, pooled across 7 sources
(n=395 binary, HC 101 / PD 294: ds000245 45, ds001354 28, ds004392 58,
ds005892 55, Neurocon 41, PPMI-195 128, Tao Wu 40), plus cognition
staging (NC/MCI/PDD, n=78). Same nested protocol and models as
before, plus a soft-voting ensemble. **It did not work either.**
Pooled 5-fold best is 0.706 (SVM, derived fusion, n=209; the screening
pass on the same feature family reached 0.740 with LR), but leave-one-source-out stays at
chance: the best honest LOSO balanced accuracy is 0.564 (ensemble,
derived fusion, n=209 — the only arm with no single-class fold). The
headline 0.611 (SVM, raw fusion) is inflated by the all-PD ds004392
fold, where balanced accuracy is just PD recall; without that fold it
is 0.522. ComBat still does not help (LOSO LR 0.592 as reported,
0.561 without the all-PD fold). Cognition staging: pooled LR 0.570,
LOSO SVM 0.450 — but the LOSO SVM calls almost every subject MCI
(recall 0.94 MCI, 0.00 NC, 0.00 PDD), so it stages nothing.
Sex-stratified accuracy shows no meaningful gap (pooled LR: men
0.60, women 0.57; PDD is all male, so no female PDD accuracy
exists). The ds006224 test-retest probe (5 subjects, 3 sessions
each, excluded from training): connectivity is stable within a
person (r 0.59) vs between people (r 0.42), but the best model's PD
probabilities flip across sessions for 3 of 5 subjects. **The 90%
target was not reached on any honest test.** The bottleneck is the
cross-site disease signal in resting-state connectivity at this
sample size — not the model, the harmonisation, or the feature list.
Full numbers: `training/metrics_round4.json`; model card:
`training/MODEL_CARD.md`; probe: `training/probe_6224.json`;
sex-stratified re-run: `training/sex_stratified_cog.json`. Weights
saved (`training/models_round4/`, picked SVM refits) — **neither is
a deployment candidate**.

### Round 4B — completed 2026-11-14 (time-series statistics, unified blocks n=395)

Per-parcel time-series statistics (std, skew, kurtosis, lag-1
autocorrelation, Hurst, spectral entropy) plus per-subject brain
states, on the same 7-source pool, with five extra model families
(extra trees, Gaussian process, balanced bagging, calibrated LR,
stacking). **It did not work either.** Pooled 5-fold best is 0.718
(stacking, derived-fusion subset, n=209), but the best honest
leave-one-source-out balanced accuracy is 0.591 (balanced bagging,
same n=209 subset, 3 folds) — and on the full n=395 pool the B arms
are 0.49–0.50 honest, i.e. chance. The clean pipeline-mismatch test
(b264_SG_unified vs the finisher's arm264_SG, same 264 subjects and
blocks, only ds005892 reprocessed through the unified pipeline):
pooled 0.633 vs 0.601, LOSO 0.517 vs 0.498, and on the ds005892 fold
itself +0.015 (LR) / +0.038 (XGB) / 0.000 (SVM, RF) — unification did
not recover accuracy, so the drop is a site effect, not a pipeline
effect. Cognition staging (B block, n=78): pooled LR 0.427, LOSO
0.433 on a single fold — still chance. The full run was
early-stopped (fast LOSO at chance for 4 of 6 arms); only the two
survivors got manual full runs, and those are partial (b209
complete, b186 pooled 7/10 models, LOSO 4/10) because VM restart 45
killed them and the early-stop guard blocks relaunching. **The 90%
target was not reached.** Full numbers:
`training/metrics_round4.json` (4B arms merged 2026-11-14); model
card: `training/MODEL_CARD.md`. No 4B weights were saved — no arm
beats the finisher's honest best by a margin worth refitting.

### Round 4C — completed 2026-11-14 (site-invariant transforms, fast screening only, n=395)

Rank, stable-rank, rank-Gaussian, age-residualised and per-site
z-score transforms of the common blocks, plus lag-1 precision and
partial correlation, a LOSO feature selector and a block stacker.
**None of them helped.** Pooled 5-fold best is 0.622 (SVM,
stable-rank), but the best honest leave-one-source-out balanced
accuracy is 0.533 (SVM, lag-1 precision) — chance, after removing
the all-PD ds004392 fold, which inflates the reported numbers badly
here (rank-Gaussian scores a perfect 1.000 on that fold with all
four models). The LOSO selector (0.498 honest) and the block stacker
(0.492 honest) also fail: there is no feature subset in these blocks
that generalises across sites. The full run was skipped by the
early-stop rule because fast LOSO was at chance. **The 90% target
was not reached.** Full numbers: `training/metrics_round4.json`
(4C arms merged 2026-11-14); model card:
`training/MODEL_CARD.md`.

### Round 1 — 2026-11-09 (archived)

Stratified 5-fold CV, seed 42, n=55, no motion correction, no confound
regressors, Schaefer-100 only. Best 3-class: GAT balanced accuracy
0.401 ± 0.173 (accuracy 0.418, AUC 0.620). Binary HC vs PD: logistic
regression AUC 0.745 ± 0.127 (balanced acc. 0.683 ± 0.103). The binary
number is now understood as partly noise (see the diagnostic above).
Archived metrics: `training/metrics_v1.json`.

Training status and scores also live in `website/public/results.json` and
are rendered live on the site.

## Pipeline

Round 2 (`training/preprocess_v2.py`, `training/train_v2.py`,
`training/train_pooled.py`): raw BIDS → ANTs affine normalisation to
MNI152 2 mm (BOLD-mean → T1 rigid, T1 → MNI affine; spatial transform
applied in blocks of 25 volumes to stay under the 6 GB memory limit) →
**motion correction on** (Friston-24 regressors, FD scrubbing at 0.5 mm,
high-variance confounds) → Nilearn Schaefer 100- and 200-parcel time
series (6 mm smoothing, bandpass 0.01–0.1 Hz, TR 2 s) → Pearson
functional-connectivity matrices → nested-CV benchmark (same 8 models
plus a two-stage classifier and tuned ensembles) → full-data refits
saved as model weights. Preprocessing v2 took ~171 min for 55 subjects;
training v2 and pooled training ran in stages with per-model
checkpoints, resuming after machine restarts.

Round 1 (`training/preprocess.py`, `training/train.py`): same pipeline
with motion correction skipped (327 s/subject on this VM) and no confound
regressors, Schaefer-100 only, plain 5-fold CV.

**Research demo only — not a medical diagnosis.**
