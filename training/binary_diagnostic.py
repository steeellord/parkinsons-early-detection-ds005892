#!/usr/bin/env python3
"""Diagnose the v1 -> v2 binary gap (LR AUC 0.745 -> 0.475).

v1 binary LR used all 4950 features, C=1.0, no selection, plain 5-fold.
v2 binary LR uses top-k selection + tuned C inside nested CV. The
3-class LR improved (0.319 -> 0.502) while binary collapsed, so the
cause is either the method (selection/regularisation) or the features
(motion correction removing a motion confound). This runs the four
corners on the same seed-42 splits and reports each honestly:

  A: v1-style (all features, C=1.0) on v2 features
  B: v2-style (selection + tuned C) on v1 features
  C: v1-style on v1 features (reproduces the v1 number)
  D: mean framewise displacement alone as a PD score (motion confound?)

Run after train_v2.py finishes so it does not compete for CPU.
"""
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

BASE = Path(__file__).resolve().parent
SEED = 42


def load(atlas_dir, n_parcels, fname_fmt):
    labels = list(csv.DictReader(open(atlas_dir / "labels.csv")))
    X, y3 = [], []
    for r in labels:
        fz = np.load(atlas_dir / fname_fmt.format(sub=r["subject"],
                                                  n=n_parcels))
        iu = np.triu_indices(n_parcels, k=1)
        X.append(fz[iu])
        y3.append(int(r["label3"]))
    y3 = np.array(y3)
    return np.array(X), y3, (y3 > 0).astype(int)


def plain_lr(X, y, splits):
    aucs, bals = [], []
    for tr, te in splits:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=5000,
                                             class_weight="balanced",
                                             random_state=SEED))
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        bals.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return float(np.mean(aucs)), float(np.mean(bals))


def nested_lr(X, y, splits):
    aucs, bals = [], []
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for tr, te in splits:
        pipe = make_pipeline(SelectKBest(f_classif),
                             StandardScaler(),
                             LogisticRegression(max_iter=5000,
                                                class_weight="balanced",
                                                random_state=SEED))
        grid = {"selectkbest__k": [500, 1000, 2000],
                "logisticregression__C": [0.01, 0.1, 1.0]}
        gs = GridSearchCV(pipe, grid, cv=inner,
                          scoring="balanced_accuracy", n_jobs=1)
        gs.fit(X[tr], y[tr])
        m = gs.best_estimator_
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        bals.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return float(np.mean(aucs)), float(np.mean(bals))


def main():
    X2, _, yb = load(BASE / "features_v2", 100, "{sub}_fc{n}_fisherz.npy")
    X1, _, _ = load(BASE / "features", 100, "{sub}_fc_fisherz.npy")
    assert (yb == (np.array([int(r["label3"]) for r in
                             csv.DictReader(open(BASE / "features_v2" /
                                                 "labels.csv"))]) > 0)
            .astype(int)).all()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(X2, yb))
    out = {}
    out["A_v1style_on_v2"] = plain_lr(X2, yb, splits)
    out["B_v2style_on_v1"] = nested_lr(X1, yb, splits)
    out["C_v1style_on_v1"] = plain_lr(X1, yb, splits)
    out["D_v2style_on_v2"] = nested_lr(X2, yb, splits)
    # motion confound: mean FD as a PD score. labels_qc.csv only lists
    # subjects with scrubbed volumes, so read mean FD from the per-subject
    # motion files instead (all 55 subjects, same order as labels.csv).
    fds = []
    for r in csv.DictReader(open(BASE / "features_v2" / "labels.csv")):
        m = pd.read_csv(BASE / "features_v2" / "motion" /
                        f"{r['subject']}_motion.csv")
        fds.append(float(m["fd_mm"].mean()))
    fds = np.array(fds)
    out["E_fd_alone_auc"] = float(roc_auc_score(yb, fds))
    out["E_fd_mean_hc"] = float(fds[yb == 0].mean())
    out["E_fd_mean_pd"] = float(fds[yb == 1].mean())
    print(json.dumps(out, indent=2))
    with open(BASE / "binary_diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote binary_diagnostic.json")


if __name__ == "__main__":
    main()
