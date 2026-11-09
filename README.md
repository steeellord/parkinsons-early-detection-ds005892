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

Training completed 2026-11-09 (9.6 min, CPU-only, stratified 5-fold CV,
seed 42, n=55). Full per-fold metrics: `training/metrics.json`; model card:
`training/MODEL_CARD.md`. Scores are means ± std across folds.

**3-class HC vs PD-NC vs PD-MCI** (chance: accuracy 0.333):

| Model | Accuracy | Balanced acc. | AUC | F1 |
|---|---|---|---|---|
| GAT (best) | 0.418 ± 0.187 | 0.401 ± 0.173 | 0.620 ± 0.183 | 0.362 ± 0.166 |
| GCN | 0.382 ± 0.176 | 0.357 ± 0.165 | 0.531 ± 0.131 | 0.315 ± 0.182 |
| Logistic regression | 0.345 ± 0.089 | 0.319 ± 0.097 | 0.540 ± 0.163 | 0.282 ± 0.101 |
| MLP | 0.327 ± 0.109 | 0.314 ± 0.103 | 0.495 ± 0.149 | 0.282 ± 0.109 |
| Random forest | 0.327 ± 0.073 | 0.291 ± 0.078 | 0.528 ± 0.084 | 0.247 ± 0.106 |
| XGBoost | 0.309 ± 0.073 | 0.280 ± 0.075 | 0.539 ± 0.094 | 0.249 ± 0.098 |
| 1D-CNN | 0.273 ± 0.100 | 0.259 ± 0.101 | 0.495 ± 0.142 | 0.238 ± 0.099 |
| SVM (RBF) | 0.273 ± 0.081 | 0.253 ± 0.089 | 0.594 ± 0.094 | 0.217 ± 0.078 |

**Binary HC vs PD:** logistic regression AUC 0.745 ± 0.127 (balanced acc.
0.683 ± 0.103), GAT 0.709 ± 0.068 acc / AUC 0.658, GCN 0.691 ± 0.109 acc /
AUC 0.697. The 3-class task is near chance — distinguishing PD-NC from
PD-MCI from resting-state connectivity alone is the hard part, and n=55
makes every number noisy. 3D CNN was skipped deliberately (n=55, no GPU —
see MODEL_CARD.md). Training status and scores also live in
`website/public/results.json` and are rendered live on the site.

## Pipeline

Raw BIDS (no fMRIPrep derivatives exist for this release) → ANTs affine
normalisation to MNI152 2 mm (BOLD-mean → T1 rigid, T1 → MNI affine;
motion correction skipped — 327 s/subject on this VM — and no confound
regressors) → Nilearn Schaefer 100-parcel time series (6 mm smoothing,
bandpass 0.01–0.1 Hz, TR 2 s) → Pearson functional-connectivity matrices →
benchmark: GCN/GAT, MLP, 1D-CNN, SVM, Logistic Regression, Random
Forest/XGBoost. Max 1000 epochs with early stopping (patience 50);
stratified 5-fold cross-validation; metrics: accuracy, balanced accuracy,
AUC, F1, confusion matrix. Preprocessing took 106.4 min for 55 subjects;
training 9.6 min. Code: `training/preprocess.py`, `training/train.py`.

**Research demo only — not a medical diagnosis.**
