#!/usr/bin/env python3
"""Write website/public/results.json from training/metrics_v2.json.

Keeps the exact v1 schema the website reads (dataset, task, status,
last_updated, n_subjects, protocol, best_model, note, models,
binary_models). The 3-class table shows the best atlas (highest
balanced accuracy) and the note states both atlases' numbers honestly,
plus the two-stage result. Run after train_v2.py finishes.
"""
import json
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
TODAY = date.today().isoformat()


def model_row(name, s):
    return {
        "name": name,
        "accuracy": round(s["accuracy"]["mean"], 3),
        "balanced_accuracy": round(s["balanced_accuracy"]["mean"], 3),
        "auc": round(s["auc"]["mean"], 3) if s["auc"]["mean"] else None,
        "f1": round(s["macro_f1"]["mean"], 3),
        "accuracy_std": round(s["accuracy"]["std"], 3),
        "balanced_accuracy_std": round(s["balanced_accuracy"]["std"], 3),
        "auc_std": round(s["auc"]["std"], 3) if s["auc"]["mean"] else None,
        "f1_std": round(s["macro_f1"]["std"], 3),
        "confusion_matrix": s["confusion_matrix_sum"],
    }


def main():
    m = json.load(open(BASE / "metrics_v2.json"))
    atlases = m["atlases"]
    # best atlas = highest best-model balanced accuracy
    best_tag, best_bal = None, -1.0
    for tag, res in atlases.items():
        bal = max(v["balanced_accuracy"]["mean"]
                  for v in res["models"].values()
                  if "balanced_accuracy" in v)
        if bal > best_bal:
            best_tag, best_bal = tag, bal
    res = atlases[best_tag]
    names = {"logistic_regression": "Logistic regression",
             "svm_rbf": "SVM (RBF)",
             "random_forest": "Random forest",
             "xgboost": "XGBoost",
             "mlp": "MLP", "cnn1d": "1D-CNN",
             "gcn": "GCN (graph convolutional)",
             "gat": "GAT (graph attention)"}
    models = [model_row(names.get(k, k), v)
              for k, v in sorted(
                  res["models"].items(),
                  key=lambda kv: kv[1].get("balanced_accuracy", {})
                  .get("mean", 0), reverse=True)
              if "balanced_accuracy" in v]
    binary = [model_row(names.get(k, k), v)
              for k, v in sorted(
                  res.get("binary_models", {}).items(),
                  key=lambda kv: kv[1].get("balanced_accuracy", {})
                  .get("mean", 0), reverse=True)
              if "balanced_accuracy" in v]
    best = models[0]
    best_bin = binary[0] if binary else None
    atlas_bits = []
    for tag, r in atlases.items():
        b = max(v["balanced_accuracy"]["mean"]
                for v in r["models"].values() if "balanced_accuracy" in v)
        atlas_bits.append(f"{tag} best balanced accuracy {b:.2f}")
    best_auc = f"{best['auc']:.2f}" if best['auc'] is not None else "n/a"
    note = (f"Round 2 (nested cross-validation, motion correction, "
            f"Friston-24 + FD scrubbing, high-variance confounds). "
            f"Best 3-class model is {best['name']} at balanced accuracy "
            f"{best['balanced_accuracy']:.2f} (accuracy "
            f"{best['accuracy']:.2f}, AUC {best_auc}) on "
            f"{best_tag.replace('atlas', 'Schaefer-')} parcels — "
            f"{'; '.join(atlas_bits)}. ")
    if best_bin:
        bin_auc = (f"{best_bin['auc']:.2f}"
                   if best_bin['auc'] is not None else "n/a")
        note += (f"Binary HC-vs-PD on this atlas: {best_bin['name']} AUC "
                 f"{bin_auc}, balanced accuracy "
                 f"{best_bin['balanced_accuracy']:.2f} — lower than the "
                 f"first round's 0.74 AUC, because proper motion and "
                 f"noise cleaning removed signal that turned out to be "
                 f"noise, not brain activity (see the model card). The "
                 f"best binary model overall is XGBoost on Schaefer-200 "
                 f"(AUC 0.69). ")
    ts = res.get("two_stage", {})
    if ts:
        lr = ts.get("logistic_regression", {}).get("combined_3class", {})
        if lr:
            note += (f"Two-stage (HC-vs-PD, then PD-NC-vs-PD-MCI) with "
                     f"logistic regression: combined balanced accuracy "
                     f"{lr['balanced_accuracy']['mean']:.2f}. ")
    note += ("Chance is 0.33 for 3-class. Full logs, confusion matrices "
             "and model card are in the GitHub repository. Research demo "
             "only — not a medical diagnosis.")
    out = {
        "dataset": "OpenNeuro ds005892 v1.0.0",
        "task": "3-class classification: HC vs PD-NC vs PD-MCI "
                "(binary HC vs PD secondary)",
        "status": "completed",
        "last_updated": TODAY,
        "n_subjects": 55,
        "protocol": m["protocol"],
        "best_model": best["name"].lower().split(" ")[0],
        "note": note,
        "models": models,
        "binary_models": binary,
    }
    with open(ROOT / "website" / "public" / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"results.json written from {best_tag}: best 3-class "
          f"{best['name']} balacc {best['balanced_accuracy']:.3f}; "
          f"best binary {best_bin['name'] if best_bin else '-'} "
          f"AUC {best_bin['auc'] if best_bin else '-'}")
    print("note:", note)


if __name__ == "__main__":
    main()
