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

Training status and scores live in `website/public/results.json` and are rendered
live on the site. While training is in progress it reads
`status: "training_in_progress"` with an empty `models` list — no accuracy numbers
are claimed until a training run completes and updates that file.

## Pipeline (training, in progress)

fMRIPrep / Nilearn preprocessing → Schaefer 100-parcel atlas → Pearson
functional-connectivity matrices → benchmark: GCN/GAT, MLP, 1D-CNN, SVM,
Logistic Regression, Random Forest/XGBoost, 3D CNN (comparison). Max 1000 epochs
with early stopping; stratified cross-validation; metrics: accuracy, balanced
accuracy, AUC, F1, confusion matrix.

**Research demo only — not a medical diagnosis.**
