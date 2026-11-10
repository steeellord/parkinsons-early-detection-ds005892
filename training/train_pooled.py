#!/usr/bin/env python3
"""Pooled binary (HC vs PD) benchmark.

Pools the Schaefer-100 FC features from every source we have:
  * BrainNet derived sets (features_pooled/): PPMI, Neurocon, Tao Wu
    — FCs recomputed here from their Schaefer-100 timeseries.
  * ds005892 v2 (features_v2/, when present): our own pipeline,
    binary label from label_bin.

Two honest evaluations, both nested (all scaling/selection/tuning
inside the inner fold):
  1. Pooled stratified 5-fold CV (seed 42).
  2. Leave-one-source-out CV: each fold holds out one whole dataset.
     This is the number that matters — it says whether the model
     works on a scanner/protocol it never saw, instead of just
     memorising site differences.

Models: same classical + torch runners as train_v2 (imported, not
copied), binary, n_classes=2. Full-data refits saved to
training/models_pooled/ (.pkl classical, .pt torch).

Outputs: training/metrics_pooled.json, training/figures_pooled/.
"""
import argparse, importlib.util, json, logging, sys, time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
POOL = BASE / "features_pooled"
V2 = BASE / "features_v2"
LOG = BASE / "logs" / "training_pooled.log"
FIG = BASE / "figures_pooled"
MODELS = BASE / "models_pooled"
SEED = 42
N_FOLDS = 5

spec = importlib.util.spec_from_file_location("tv2", BASE / "train_v2.py")
tv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv2)
# Point the shared runners' output dirs at the pooled folders.
tv2.MODELS = MODELS
tv2.FIG = FIG


def setup_logging():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log_stream = open(LOG, "a", buffering=1)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(log_stream),
                  logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("train_pooled")


def load_pooled(log):
    """Return X (upper-tri Fisher-z), mats, yb, sources, subjects."""
    import pandas as pd
    frames = []
    if (POOL / "labels.csv").exists():
        lab = pd.read_csv(POOL / "labels.csv")
        lab["feat_dir"] = str(POOL)
        frames.append(lab)
    else:
        log.warning("features_pooled/labels.csv missing — pooled-only run")
    if (V2 / "labels.csv").exists():
        lab = pd.read_csv(V2 / "labels.csv")
        lab["source"] = "ds005892"
        lab["feat_dir"] = str(V2)
        frames.append(lab)
    else:
        log.warning("features_v2/labels.csv missing — ds005892 not pooled")
    if not frames:
        raise SystemExit("no features found: run build_pooled_fc.py and "
                         "preprocess_v2 first")
    lab = pd.concat(frames, ignore_index=True)
    mats, yb, sources, subjects = [], [], [], []
    for _, r in lab.iterrows():
        m = np.load(Path(r["feat_dir"]) / f"{r['subject']}_fc100_fisherz.npy")
        mats.append(m)
        yb.append(int(r["label_bin"]))
        sources.append(r["source"])
        subjects.append(r["subject"])
    mats = np.stack(mats)
    yb = np.array(yb)
    iu = np.triu_indices(100, k=1)
    X = np.array([m[iu] for m in mats])
    log.info("pooled n=%d (HC=%d PD=%d) sources=%s",
             len(yb), int((yb == 0).sum()), int((yb == 1).sum()),
             {s: int((np.array(sources) == s).sum())
              for s in sorted(set(sources))})
    return X, mats, yb, np.array(sources), subjects


def save_binary_figures(results, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    for scheme, res in results.items():
        for name, r in res.get("models", {}).items():
            if "confusion_matrix_sum" not in r:
                continue
            cm = np.array(r["confusion_matrix_sum"])
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(2)); ax.set_yticks(range(2))
            ax.set_xticklabels(["HC", "PD"]); ax.set_yticklabels(["HC", "PD"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{scheme} {name} — summed CV confusion matrix")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2
                            else "black")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(FIG / f"cm_{scheme}_{name}.png", dpi=150)
            plt.close(fig)
    log.info("pooled figures saved to %s", FIG)


def run_scheme(X, mats, yb, splits, log, tag, skip_torch):
    from sklearn.model_selection import StratifiedKFold
    res = {"models": {}}
    classical, _ = tv2.run_classical_nested(X, yb, splits, 2, log, tag)
    res["models"].update(classical)
    if not skip_torch:
        ranked = sorted(
            ((k, v) for k, v in res["models"].items()
             if "balanced_accuracy" in v),
            key=lambda kv: kv[1]["balanced_accuracy"]["mean"], reverse=True)
        for k, _ in ranked[:3]:
            try:
                res["models"][k] = tv2.run_torch_nested(
                    k, X, mats, yb, splits, 2, log, tag)
            except Exception:
                log.exception("[%s %s] FAILED", tag, k)
                res["models"][k] = {"status": "failed", "kind": "neural"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--skip-full-refit", action="store_true")
    ap.add_argument("--loso-only", action="store_true",
                    help="only the leave-one-source-out scheme")
    args = ap.parse_args()
    log = setup_logging()
    t0 = time.time()
    log.info("=== Pooled binary benchmark: nested CV + leave-one-source-out ===")
    from sklearn.model_selection import StratifiedKFold
    X, mats, yb, sources, subjects = load_pooled(log)
    out = {"version": 2, "task": "binary HC vs PD (pooled)",
           "n": int(len(yb)), "seed": SEED, "n_folds": N_FOLDS,
           "sources": sorted(set(sources.tolist())),
           "protocol": "Nested CV: inner stratified 3-fold for all scaling, "
           "feature selection and hyperparameter search. Pooled scheme: "
           "outer stratified 5-fold (seed 42). LOSO scheme: each fold holds "
           "out one whole source dataset.",
           "schemes": {}}
    if args.loso_only:
        # resume from the pooled_5fold checkpoint written by an earlier run
        p = BASE / "metrics_pooled.json"
        if p.exists():
            with open(p) as f:
                out = json.load(f)
            log.info("resumed from %s (schemes: %s)",
                     p, list(out["schemes"]))
    if not args.loso_only:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=SEED)
        splits = list(skf.split(X, yb))
        out["schemes"]["pooled_5fold"] = run_scheme(
            X, mats, yb, splits, log, "pooled", args.skip_torch)
        with open(BASE / "metrics_pooled.json", "w") as f:
            json.dump(out, f, indent=2)
        log.info("Checkpoint metrics_pooled.json written (pooled_5fold done)")
    # Leave-one-source-out: one fold per source dataset.
    loso_splits = []
    for src in sorted(set(sources.tolist())):
        te = np.where(sources == src)[0]
        tr = np.where(sources != src)[0]
        loso_splits.append((tr, te))
    out["schemes"]["loso"] = run_scheme(
        X, mats, yb, loso_splits, log, "loso", args.skip_torch)
    with open(BASE / "metrics_pooled.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("Checkpoint metrics_pooled.json written (loso done)")
    if not args.skip_full_refit:
        out["saved_full_models"] = {}
        out["saved_full_models"]["classical"] = tv2.refit_full_classical(
            X, yb, 2, log, "pooled")
        if not args.skip_torch:
            out["saved_full_models"]["torch"] = {}
            for kind in ["mlp", "cnn1d", "gcn", "gat"]:
                try:
                    out["saved_full_models"]["torch"][kind] = \
                        tv2.refit_full_torch(kind, X, mats, yb, 2, log,
                                             "pooled")
                except Exception:
                    log.exception("[pooled full %s] FAILED", kind)
    save_binary_figures(out["schemes"], log)
    out["total_seconds"] = round(time.time() - t0, 1)
    out["total_minutes"] = round((time.time() - t0) / 60, 1)
    with open(BASE / "metrics_pooled.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("=== DONE pooled in %.1f min. metrics_pooled.json written ===",
             (time.time() - t0) / 60)
    for scheme, res in out["schemes"].items():
        ranked = sorted(
            ((k, v["balanced_accuracy"]["mean"])
             for k, v in res["models"].items() if "balanced_accuracy" in v),
            key=lambda kv: kv[1], reverse=True)
        log.info("[%s] ranking by balacc: %s", scheme, ranked)


if __name__ == "__main__":
    raise SystemExit(main())
