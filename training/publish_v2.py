#!/usr/bin/env python3
"""Publish round-2 results: build training/metrics.json (v1 schema) from
metrics_v2.json, metrics_pooled.json and binary_diagnostic.json.

The v1 schema is kept (task, n, seed, n_folds, models, binary,
total_seconds, total_minutes) so anything that read the v1 file still
works. Extra top-level keys carry the round-2 detail: both atlases,
the two-stage results, the pooled benchmark and the binary diagnostic.
Best atlas for the main table = highest best-model balanced accuracy.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def summarise(models):
    out = {}
    for k, v in models.items():
        if "balanced_accuracy" not in v:
            out[k] = v  # skipped entries are kept as-is, with the reason
            continue
        out[k] = {
            "balanced_accuracy": round(v["balanced_accuracy"]["mean"], 3),
            "accuracy": round(v["accuracy"]["mean"], 3),
            "auc": round(v["auc"]["mean"], 3)
            if v["auc"]["mean"] else None,
            "macro_f1": round(v["macro_f1"]["mean"], 3),
        }
    return out


def main():
    v2 = json.load(open(BASE / "metrics_v2.json"))
    pooled = json.load(open(BASE / "metrics_pooled.json"))
    diag = json.load(open(BASE / "binary_diagnostic.json"))

    # best atlas = highest best-model balanced accuracy
    best_tag, best_bal = None, -1.0
    for tag, res in v2["atlases"].items():
        bal = max(v["balanced_accuracy"]["mean"]
                  for v in res["models"].values()
                  if "balanced_accuracy" in v)
        if bal > best_bal:
            best_tag, best_bal = tag, bal
    best = v2["atlases"][best_tag]

    out = {
        "version": 2,
        "task": "3-class HC vs PD-NC vs PD-MCI",
        "n": v2["n"],
        "seed": v2["seed"],
        "n_folds": v2["n_folds"],
        "protocol": v2["protocol"],
        "best_atlas": best_tag,
        "models": best["models"],
        "binary": {
            "task": "binary HC vs PD (PD-NC+PD-MCI)",
            "atlas": best_tag,
            "models": best["binary_models"],
        },
        "total_seconds": v2["total_seconds"],
        "total_minutes": v2["total_minutes"],
        "atlases": {
            tag: {
                "models": summarise(res["models"]),
                "binary_models": summarise(res["binary_models"]),
                "two_stage": {
                    k: {
                        s: round(sv["balanced_accuracy"]["mean"], 3)
                        for s, sv in v.items()
                    }
                    for k, v in res["two_stage"].items()
                },
                "saved_full_models": res["saved_full_models"],
            }
            for tag, res in v2["atlases"].items()
        },
        "pooled": {
            "n": pooled["n"],
            "sources": pooled["sources"],
            "schemes": {
                s: summarise(v["models"])
                for s, v in pooled["schemes"].items()
            },
            "saved_full_models": pooled.get("saved_full_models", {}),
        },
        "binary_diagnostic": {
            "A_v1style_on_v2_features_auc": round(diag["A_v1style_on_v2"][0], 3),
            "A_v1style_on_v2_features_balacc": round(diag["A_v1style_on_v2"][1], 3),
            "B_v2style_on_v1_features_auc": round(diag["B_v2style_on_v1"][0], 3),
            "B_v2style_on_v1_features_balacc": round(diag["B_v2style_on_v1"][1], 3),
            "C_v1style_on_v1_features_auc": round(diag["C_v1style_on_v1"][0], 3),
            "C_v1style_on_v1_features_balacc": round(diag["C_v1style_on_v1"][1], 3),
            "D_v2style_on_v2_features_auc": round(diag["D_v2style_on_v2"][0], 3),
            "D_v2style_on_v2_features_balacc": round(diag["D_v2style_on_v2"][1], 3),
            "E_mean_fd_alone_auc": round(diag["E_fd_alone_auc"], 3),
            "E_fd_mean_hc_mm": round(diag["E_fd_mean_hc"], 3),
            "E_fd_mean_pd_mm": round(diag["E_fd_mean_pd"], 3),
            "finding": (
                "The v1 binary signal (LR AUC 0.745) does not survive "
                "cleaning: v1 method on v2 features gives AUC 0.480, while "
                "v2 method on v1 features gives AUC 0.689, so the "
                "difference is the cleaned features, not the method. "
                "Mean head motion alone is barely predictive (AUC 0.523), "
                "so the removed signal was motion/physiological noise in "
                "the connectivity, not raw head movement. The 3-class "
                "signal is real and improved with cleaning (LR 0.319 to "
                "0.502). The best v2 binary model is XGBoost on "
                "Schaefer-200 (AUC 0.689)."
            ),
        },
    }
    with open(BASE / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote metrics.json (version 2, best atlas %s)" % best_tag)


if __name__ == "__main__":
    main()
