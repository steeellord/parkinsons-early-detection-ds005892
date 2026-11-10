#!/usr/bin/env python3
"""Benchmark v2 for Parkinson's early detection (ds005892).

What changed vs v1 (see MODEL_CARD.md for the full story):
  * Features: preprocessing v2 — motion correction ON, Friston-24 motion
    regressors, framewise-displacement scrubbing (FD > 0.5 mm censored),
    high-variance confounds (nilearn CompCor-like), Schaefer-100 AND
    Schaefer-200 parcels. Both atlases are benchmarked and compared.
  * Protocol: NESTED cross-validation. Outer stratified 5-fold (seed 42,
    SAME outer splits as v1 so results are comparable). All scaling,
    feature selection and hyperparameter search happen INSIDE the inner
    fold (GridSearchCV, inner stratified 3-fold for classical models;
    inner 3-fold config search for torch models). No leakage by design.
  * Two-stage classifier: stage 1 HC vs PD, stage 2 PD-NC vs PD-MCI
    (PD-only), cascaded into a combined 3-class prediction.
  * Tuned ensembles: LR / SVM / RF / XGBoost with real grids; torch
    models with a small config search. Same 8-model benchmark as v1
    plus the new variants, so v1 vs v2 is apples-to-apples.
  * Weights SAVED: best full-data model per kind to training/models_v2/
    (.pt for torch, .pkl for sklearn pipelines) for future inference.

Outputs: training/metrics_v2.json, training/figures_v2/, training/models_v2/.
"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
FEAT = BASE / "features_v2"
LOG = BASE / "logs" / "training_v2.log"
FIG = BASE / "figures_v2"
MODELS = BASE / "models_v2"
SEED = 42
N_FOLDS = 5


def setup_logging():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered file stream: on this machine a plain FileHandler can
    # sit with records stuck in its buffer (flush calls fail here), so
    # records would only reach disk when the buffer fills. Line buffering
    # writes each record out at its newline instead.
    log_stream = open(LOG, "a", buffering=1)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(log_stream),
                  logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("train_v2")


def load_data(log, n_parcels):
    import pandas as pd
    labels = pd.read_csv(FEAT / "labels.csv")
    mats, ys3, ysb = [], [], []
    for _, r in labels.iterrows():
        m = np.load(FEAT / f"{r['subject']}_fc{n_parcels}_fisherz.npy")
        mats.append(m)
        ys3.append(int(r["label3"]))
        ysb.append(int(r["label_bin"]))
    mats = np.stack(mats)
    for i in range(mats.shape[0]):
        np.fill_diagonal(mats[i], 0.0)
    iu = np.triu_indices(n_parcels, k=1)
    X = mats[:, iu[0], iu[1]]
    log.info("Atlas-%d: loaded %d subjects: X %s, mats %s",
             n_parcels, len(labels), X.shape, mats.shape)
    return X, mats, np.array(ys3), np.array(ysb), labels["subject"].tolist()


def fold_metrics(y_true, y_pred, y_proba, n_classes):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, confusion_matrix, roc_auc_score)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }
    try:
        if n_classes == 2:
            m["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            m["auc"] = float(roc_auc_score(y_true, y_proba,
                                           multi_class="ovr", average="macro"))
    except Exception:
        m["auc"] = None
    return m


def summarize(per_fold, log, name):
    keys = ["accuracy", "balanced_accuracy", "auc", "macro_f1"]
    s = {}
    for k in keys:
        vals = [f[k] for f in per_fold if f.get(k) is not None]
        s[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "folds": [round(v, 4) for v in vals]}
    cm = np.sum([np.array(f["confusion_matrix"]) for f in per_fold], axis=0)
    s["confusion_matrix_sum"] = cm.tolist()
    log.info("[%s] acc=%.3f±%.3f balacc=%.3f±%.3f auc=%.3f±%.3f f1=%.3f±%.3f",
             name, s["accuracy"]["mean"], s["accuracy"]["std"],
             s["balanced_accuracy"]["mean"], s["balanced_accuracy"]["std"],
             s["auc"]["mean"], s["auc"]["std"],
             s["macro_f1"]["mean"], s["macro_f1"]["std"])
    log.info("[%s] confusion (rows=true, cols=pred):\n%s", name, cm)
    return s


# ---------------- classical models: nested CV ----------------
def classical_grids(n_feat):
    """(estimator, param_grid) per model. SelectKBest k is relative to
    feature count so the same code works for atlas-100 (4950) and
    atlas-200 (19900)."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    k_all = n_feat
    k_mid = min(1000, n_feat)
    k_small = min(500, n_feat)
    grids = {
        "logistic_regression": (
            Pipeline([("sc", StandardScaler()),
                      ("sel", SelectKBest(f_classif)),
                      ("clf", LogisticRegression(max_iter=5000,
                                                 class_weight="balanced",
                                                 random_state=SEED))]),
            {"sel__k": [k_small, k_mid, k_all], "clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "svm_rbf": (
            Pipeline([("sc", StandardScaler()),
                      ("sel", SelectKBest(f_classif)),
                      ("clf", SVC(kernel="rbf", probability=True,
                                  class_weight="balanced",
                                  random_state=SEED))]),
            {"sel__k": [k_small, k_mid], "clf__C": [0.1, 1.0, 10.0],
             "clf__gamma": ["scale", 0.001]}),
        "random_forest": (
            Pipeline([("sel", SelectKBest(f_classif)),
                      ("clf", RandomForestClassifier(
                          n_estimators=300, class_weight="balanced_subsample",
                          random_state=SEED, n_jobs=1))]),
            {"sel__k": [k_mid, k_all],
             "clf__max_depth": [None, 5, 10],
             "clf__min_samples_leaf": [1, 2]}),
    }
    try:
        from xgboost import XGBClassifier
        grids["xgboost"] = (
            Pipeline([("sel", SelectKBest(f_classif)),
                      ("clf", XGBClassifier(eval_metric="mlogloss",
                                            random_state=SEED, n_jobs=1))]),
            {"sel__k": [k_mid, k_all],
             "clf__n_estimators": [200, 300],
             "clf__max_depth": [3, 4],
             "clf__learning_rate": [0.03, 0.1],
             "clf__subsample": [0.8]})
    except Exception as e:
        pass
    return grids


def run_classical_nested(X, y, outer_splits, n_classes, log, tag):
    """Nested CV: outer folds given; inner GridSearchCV (3-fold) picks
    hyperparams + feature count on the outer-train only. Returns results
    dict + fitted best pipelines per fold (for weight saving)."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    grids = classical_grids(X.shape[1])
    results, fitted = {}, {}
    for name, (pipe, grid) in grids.items():
        t0 = time.time()
        per_fold, best_params, fold_models = [], [], []
        for fold, (tr, te) in enumerate(outer_splits):
            gs = GridSearchCV(pipe, grid, cv=inner,
                              scoring="balanced_accuracy", n_jobs=1)
            gs.fit(X[tr], y[tr])
            best = gs.best_estimator_
            fold_models.append(best)
            best_params.append({k: (v.tolist() if hasattr(v, "tolist") else v)
                                for k, v in gs.best_params_.items()})
            pred = best.predict(X[te])
            proba = (best.predict_proba(X[te])
                     if hasattr(best, "predict_proba")
                     else np.eye(n_classes)[pred])
            per_fold.append(fold_metrics(y[te], pred, proba, n_classes))
            log.info("[%s %s fold %d] best=%s balacc=%.3f",
                     tag, name, fold, gs.best_params_,
                     per_fold[-1]["balanced_accuracy"])
        res = summarize(per_fold, log, f"{tag}/{name}")
        res["seconds"] = round(time.time() - t0, 1)
        res["per_fold"] = per_fold
        res["kind"] = "classical"
        res["best_params_per_fold"] = best_params
        results[name] = res
        fitted[name] = fold_models
    return results, fitted


def refit_full_classical(X, y, n_classes, log, tag):
    """Refit each classical model on ALL data (inner GridSearchCV on the
    full set) and save the pipelines as .pkl for future inference."""
    import joblib
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    # inner folds capped by the smallest class so small datasets (or a
    # future pooled set with a rare class) cannot crash the refit
    n_inner = max(2, min(5, int(np.bincount(y).min())))
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True,
                           random_state=SEED)
    MODELS.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, (pipe, grid) in classical_grids(X.shape[1]).items():
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="balanced_accuracy",
                          n_jobs=1)
        gs.fit(X, y)
        path = MODELS / f"{tag}_{name}_full.pkl"
        joblib.dump(gs.best_estimator_, path)
        saved[name] = {"path": str(path), "best_params":
                       {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in gs.best_params_.items()}}
        log.info("[%s full %s] saved %s best=%s", tag, name, path,
                 gs.best_params_)
    return saved


# ---------------- torch models: nested config search ----------------
def build_adj(fc, thresh=0.2):
    A = (fc > thresh).astype(np.float32)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    return (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]).astype(np.float32)


def torch_configs(kind):
    """Small, honest config grids (4 each) for the inner-fold search."""
    if kind == "mlp":
        return [{"hidden": (256, 128), "lr": 1e-3, "dropout": 0.3, "wd": 1e-4},
                {"hidden": (128, 64), "lr": 1e-3, "dropout": 0.3, "wd": 1e-4},
                {"hidden": (256, 128), "lr": 5e-4, "dropout": 0.5, "wd": 1e-4},
                {"hidden": (128, 64), "lr": 5e-4, "dropout": 0.3, "wd": 1e-3}]
    if kind == "cnn1d":
        return [{"channels": (32, 64), "lr": 1e-3, "dropout": 0.3, "wd": 1e-4},
                {"channels": (16, 32), "lr": 1e-3, "dropout": 0.3, "wd": 1e-4},
                {"channels": (32, 64), "lr": 5e-4, "dropout": 0.5, "wd": 1e-4},
                {"channels": (16, 32), "lr": 5e-4, "dropout": 0.3, "wd": 1e-3}]
    if kind == "gcn":
        return [{"hidden": (64, 32), "lr": 1e-3, "dropout": 0.3, "wd": 5e-4},
                {"hidden": (32, 16), "lr": 1e-3, "dropout": 0.3, "wd": 5e-4},
                {"hidden": (64, 32), "lr": 5e-4, "dropout": 0.5, "wd": 5e-4},
                {"hidden": (64, 32), "lr": 1e-3, "dropout": 0.3, "wd": 1e-3}]
    if kind == "gat":
        return [{"hidden": 32, "lr": 1e-3, "dropout": 0.3, "wd": 5e-4},
                {"hidden": 16, "lr": 1e-3, "dropout": 0.3, "wd": 5e-4},
                {"hidden": 32, "lr": 5e-4, "dropout": 0.5, "wd": 5e-4},
                {"hidden": 32, "lr": 1e-3, "dropout": 0.5, "wd": 1e-3}]
    raise ValueError(kind)


def make_torch_model(kind, cfg, d_in, n_nodes, n_classes):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            h1, h2 = cfg["hidden"]
            self.net = nn.Sequential(
                nn.Linear(d_in, h1), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                nn.Linear(h2, n_classes))

        def forward(self, x):
            return self.net(x)

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            c1, c2 = cfg["channels"]
            self.conv = nn.Sequential(
                nn.Conv1d(1, c1, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(c1, c2, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
            self.fc = nn.Sequential(
                nn.Linear(c2 * (d_in // 4), 128), nn.ReLU(),
                nn.Dropout(cfg["dropout"]), nn.Linear(128, n_classes))

        def forward(self, x):
            return self.fc(self.conv(x.unsqueeze(1)).flatten(1))

    class GCNLayer(nn.Module):
        def __init__(self, d_in_, d_out):
            super().__init__()
            self.lin = nn.Linear(d_in_, d_out)

        def forward(self, h, a):
            return F.relu(a @ self.lin(h))

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            h1, h2 = cfg["hidden"]
            self.l1 = GCNLayer(n_nodes, h1)
            self.l2 = GCNLayer(h1, h2)
            self.out = nn.Linear(h2, n_classes)
            self.drop = nn.Dropout(cfg["dropout"])

        def forward(self, h, a):
            h = self.drop(self.l1(h, a))
            h = self.drop(self.l2(h, a))
            return self.out(h.mean(dim=1))

    class GATLayer(nn.Module):
        def __init__(self, d_in_, d_out):
            super().__init__()
            self.W = nn.Linear(d_in_, d_out, bias=False)
            self.a = nn.Parameter(torch.empty(2 * d_out))
            nn.init.xavier_uniform_(self.a.view(1, -1))
            self.leaky = nn.LeakyReLU(0.2)

        def forward(self, h, mask):
            Wh = self.W(h)
            B, N, D = Wh.shape
            e = self.leaky(torch.cat([Wh.unsqueeze(2).expand(B, N, N, D),
                                      Wh.unsqueeze(1).expand(B, N, N, D)],
                                     dim=-1) @ self.a)
            e = e.masked_fill(~mask, -1e9)
            return F.elu(torch.softmax(e, dim=2) @ Wh)

    class GAT(nn.Module):
        def __init__(self):
            super().__init__()
            hd = cfg["hidden"]
            self.l1 = GATLayer(n_nodes, hd)
            self.l2 = GATLayer(hd, hd)
            self.out = nn.Linear(hd, n_classes)
            self.drop = nn.Dropout(cfg["dropout"])

        def forward(self, h, a):
            mask = a > 0
            h = self.drop(self.l1(h, mask))
            h = self.drop(self.l2(h, mask))
            return self.out(h.mean(dim=1))

    return {"mlp": MLP, "cnn1d": CNN1D, "gcn": GCN, "gat": GAT}[kind]()


def train_torch_once(kind, cfg, X, mats, y, tr_idx, va_idx, n_classes,
                     seed, max_epochs=1000, patience=50):
    """Train one torch model; early-stop on va_idx. Returns
    (best_state_dict, best_val_balacc, best_epoch, history, val_proba)."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import balanced_accuracy_score
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    graph = kind in ("gcn", "gat")
    n_nodes = mats.shape[1]
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-8
    Xs = (X - mu) / sd
    counts = np.bincount(y[tr_idx], minlength=n_classes).astype(np.float32)
    w = len(y[tr_idx]) / (n_classes * np.maximum(counts, 1))
    model = make_torch_model(kind, cfg, X.shape[1], n_nodes, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["wd"])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w))
    if graph:
        A_tr = torch.tensor(np.stack([build_adj(m) for m in mats[tr_idx]]))
        A_va = torch.tensor(np.stack([build_adj(m) for m in mats[va_idx]]))
        H_tr = torch.tensor(mats[tr_idx], dtype=torch.float32)
        H_va = torch.tensor(mats[va_idx], dtype=torch.float32)
    else:
        A_tr = A_va = None
        H_tr = torch.tensor(Xs[tr_idx], dtype=torch.float32)
        H_va = torch.tensor(Xs[va_idx], dtype=torch.float32)
    y_tr = torch.tensor(y[tr_idx])
    y_va = torch.tensor(y[va_idx])

    def fwd(H, A, train):
        if train:
            model.train()
            opt.zero_grad()
            out = model(H, A) if graph else model(H)
            loss = loss_fn(out, y_tr)
            loss.backward()
            opt.step()
            return loss.item(), out.detach()
        model.eval()
        with torch.no_grad():
            out = model(H, A) if graph else model(H)
            return loss_fn(out, y_va).item(), out

    best_val, best_state, best_ep, wait = np.inf, None, 0, 0
    hist = {"train_loss": [], "val_loss": []}
    for epoch in range(max_epochs):
        tr_loss, _ = fwd(H_tr, A_tr, True)
        va_loss, va_out = fwd(H_va, A_va, False)
        hist["train_loss"].append(round(tr_loss, 4))
        hist["val_loss"].append(round(va_loss, 4))
        if va_loss < best_val - 1e-4:
            best_val, best_ep, wait = va_loss, epoch, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(H_va, A_va) if graph else model(H_va)
        proba = torch.softmax(out, dim=1).numpy()
    balacc = float(balanced_accuracy_score(y[va_idx], proba.argmax(1)))
    return best_state, balacc, best_ep, hist, proba


def predict_torch(kind, cfg, state, X, mats, y_dummy, idx, mu, sd, n_classes):
    import torch
    graph = kind in ("gcn", "gat")
    model = make_torch_model(kind, cfg, X.shape[1], mats.shape[1], n_classes)
    model.load_state_dict(state)
    model.eval()
    Xs = (X - mu) / sd
    if graph:
        A = torch.tensor(np.stack([build_adj(m) for m in mats[idx]]))
        H = torch.tensor(mats[idx], dtype=torch.float32)
    else:
        A, H = None, torch.tensor(Xs[idx], dtype=torch.float32)
    with torch.no_grad():
        out = model(H, A) if graph else model(H)
        return torch.softmax(out, dim=1).numpy()


def run_torch_nested(kind, X, mats, y, outer_splits, n_classes, log, tag):
    """Nested CV for one torch kind. Inner 3-fold picks the config;
    best config retrained on the outer-train (15% val split for early
    stopping) and tested on the outer test fold. Weights saved per fold."""
    import torch
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    MODELS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    per_fold, best_cfgs, histories, best_epochs = [], [], [], []
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(outer_splits):
        # --- inner config search ---
        cfg_scores = {}
        for cfg in torch_configs(kind):
            inner_ba = []
            for itr, iva in inner.split(X[tr], y[tr]):
                _, ba, _, _, _ = train_torch_once(
                    kind, cfg, X, mats, y, tr[itr], tr[iva], n_classes,
                    seed=SEED + fold)
                inner_ba.append(ba)
            cfg_scores[json.dumps(cfg, sort_keys=True)] = float(np.mean(inner_ba))
        best_cfg = json.loads(max(cfg_scores, key=cfg_scores.get))
        best_cfgs.append(best_cfg)
        log.info("[%s %s fold %d] inner config scores=%s -> best %s",
                 tag, kind, fold,
                 {k: round(v, 3) for k, v in cfg_scores.items()}, best_cfg)
        # --- retrain best config on outer-train, early-stop on 15% val ---
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15,
                                     random_state=SEED)
        itr, iva = next(sss.split(X[tr], y[tr]))
        state, _, best_ep, hist, _ = train_torch_once(
            kind, best_cfg, X, mats, y, tr[itr], tr[iva], n_classes,
            seed=SEED + fold)
        # tag can carry a slash (binary runs use "atlas100/binary");
        # flatten it so the checkpoint stays one file, not a subfolder.
        safe_tag = tag.replace("/", "_")
        torch.save(state, MODELS / f"{safe_tag}_{kind}_fold{fold}.pt")
        mu, sd = X[tr[itr]].mean(0), X[tr[itr]].std(0) + 1e-8
        proba = predict_torch(kind, best_cfg, state, X, mats, y, te,
                              mu, sd, n_classes)
        per_fold.append(fold_metrics(y[te], proba.argmax(1), proba, n_classes))
        histories.append(hist)
        best_epochs.append(best_ep)
        log.info("[%s %s fold %d] test acc=%.3f balacc=%.3f (best epoch %d)",
                 tag, kind, fold, per_fold[-1]["accuracy"],
                 per_fold[-1]["balanced_accuracy"], best_ep)
    res = summarize(per_fold, log, f"{tag}/{kind}")
    res["seconds"] = round(time.time() - t0, 1)
    res["per_fold"] = per_fold
    res["kind"] = "neural"
    res["best_configs_per_fold"] = best_cfgs
    res["best_epochs"] = best_epochs
    res["histories"] = histories
    return res


def refit_full_torch(kind, X, mats, y, n_classes, log, tag):
    """Train the most common best config on ALL data (15% val split for
    early stopping) and save the weights for future inference."""
    import torch
    from sklearn.model_selection import StratifiedShuffleSplit
    from collections import Counter
    # most common config across folds is picked by caller; here use config 0
    # unless the caller stored best configs — we re-search cheaply on a
    # single 3-fold inner split over all data.
    from sklearn.model_selection import StratifiedKFold
    n_inner = max(2, min(3, int(np.bincount(y).min())))
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True,
                           random_state=SEED)
    scores = {}
    for cfg in torch_configs(kind):
        bas = []
        for itr, iva in inner.split(X, y):
            _, ba, _, _, _ = train_torch_once(kind, cfg, X, mats, y,
                                              itr, iva, n_classes,
                                              seed=SEED)
            bas.append(ba)
        scores[json.dumps(cfg, sort_keys=True)] = float(np.mean(bas))
    best_cfg = json.loads(max(scores, key=scores.get))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15,
                                 random_state=SEED)
    itr, iva = next(sss.split(X, y))
    state, ba, best_ep, _, _ = train_torch_once(kind, best_cfg, X, mats, y,
                                                itr, iva, n_classes,
                                                seed=SEED)
    path = MODELS / f"{tag}_{kind}_full.pt"
    torch.save({"state_dict": state, "config": best_cfg,
                "n_parcels": mats.shape[1], "n_classes": n_classes,
                "val_balanced_accuracy": ba, "best_epoch": best_ep}, path)
    log.info("[%s full %s] saved %s config=%s val_balacc=%.3f",
             tag, kind, path, best_cfg, ba)
    return {"path": str(path), "config": best_cfg,
            "val_balanced_accuracy": ba}


# ---------------- two-stage classifier ----------------
def run_two_stage(X, y3, yb, outer_splits, log, tag):
    """Stage 1: HC vs PD (all subjects). Stage 2: PD-NC vs PD-MCI
    (PD subjects only). Cascaded into a combined 3-class prediction.
    Both stages use nested-CV tuned models (LR and XGBoost); the inner
    GridSearchCV is the same as run_classical_nested. Reported: stage-1
    metrics, stage-2 metrics (on true-PD test subjects), combined
    3-class metrics."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    grids = classical_grids(X.shape[1])
    out = {}
    for name in ["logistic_regression", "xgboost"]:
        if name not in grids:
            continue
        pipe, grid = grids[name]
        s1_folds, s2_folds, comb_folds = [], [], []
        for fold, (tr, te) in enumerate(outer_splits):
            # stage 1: HC vs PD on outer-train
            gs1 = GridSearchCV(pipe, grid, cv=inner,
                               scoring="balanced_accuracy", n_jobs=1)
            gs1.fit(X[tr], yb[tr])
            m1 = gs1.best_estimator_
            p1 = m1.predict(X[te])
            pr1 = m1.predict_proba(X[te])
            s1_folds.append(fold_metrics(yb[te], p1, pr1, 2))
            # stage 2: PD-NC (0) vs PD-MCI (1) on PD subjects of outer-train
            pd_tr = tr[yb[tr] == 1]
            z_tr = (y3[pd_tr] == 2).astype(int)
            gs2 = GridSearchCV(pipe, grid, cv=inner,
                               scoring="balanced_accuracy", n_jobs=1)
            gs2.fit(X[pd_tr], z_tr)
            m2 = gs2.best_estimator_
            # combined cascade on test subjects
            comb_pred = np.zeros(len(te), dtype=int)
            pd_te_mask = p1 == 1
            te_pd = te[pd_te_mask]
            if len(te_pd):
                p2 = m2.predict(X[te_pd])
                comb_pred[pd_te_mask] = np.where(p2 == 1, 2, 1)
            comb_folds.append(fold_metrics(y3[te], comb_pred,
                                           np.eye(3)[comb_pred], 3))
            # stage-2 metrics on TRUE PD test subjects (diagnostic)
            pd_te_true = te[yb[te] == 1]
            if len(pd_te_true):
                p2t = m2.predict(X[pd_te_true])
                pr2t = m2.predict_proba(X[pd_te_true])
                s2_folds.append(fold_metrics(
                    (y3[pd_te_true] == 2).astype(int), p2t, pr2t, 2))
            log.info("[%s two-stage/%s fold %d] s1 balacc=%.3f "
                     "s2 balacc=%.3f combined balacc=%.3f",
                     tag, name, fold, s1_folds[-1]["balanced_accuracy"],
                     s2_folds[-1]["balanced_accuracy"] if s2_folds else -1,
                     comb_folds[-1]["balanced_accuracy"])
        out[name] = {
            "stage1_hc_vs_pd": summarize(s1_folds, log, f"{tag}/two-stage/{name}/s1"),
            "stage2_pdnc_vs_pdmci": summarize(s2_folds, log, f"{tag}/two-stage/{name}/s2"),
            "combined_3class": summarize(comb_folds, log, f"{tag}/two-stage/{name}/combined"),
        }
        for k in out[name]:
            out[name][k]["per_fold"] = (s1_folds if k.startswith("stage1")
                                        else s2_folds if k.startswith("stage2")
                                        else comb_folds)
    return out


# ---------------- figures ----------------
def save_figures_v2(results_by_atlas, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    for tag, res in results_by_atlas.items():
        for name, r in res.get("models", {}).items():
            if "confusion_matrix_sum" not in r:
                continue
            cm = np.array(r["confusion_matrix_sum"])
            names = ["HC", "PD-NC", "PD-MCI"]
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels(names); ax.set_yticklabels(names)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{tag} {name} — summed CV confusion matrix (n=55)")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(FIG / f"cm_{tag}_{name}.png", dpi=150)
            plt.close(fig)
        for name, r in res.get("models", {}).items():
            if r.get("kind") != "neural" or "histories" not in r:
                continue
            hists = r["histories"]
            maxlen = max(len(h["val_loss"]) for h in hists)
            vl = np.full((len(hists), maxlen), np.nan)
            tl = np.full((len(hists), maxlen), np.nan)
            for i, h in enumerate(hists):
                vl[i, :len(h["val_loss"])] = h["val_loss"]
                tl[i, :len(h["train_loss"])] = h["train_loss"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.nanmean(vl, axis=0), label="val loss (mean over folds)")
            ax.plot(np.nanmean(tl, axis=0), label="train loss (mean over folds)")
            ax.set_xlabel("Epoch (early stopping, patience 50, max 1000)")
            ax.set_ylabel("Cross-entropy loss")
            ax.set_title(f"{tag} {name} — training curves")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIG / f"curves_{tag}_{name}.png", dpi=150)
            plt.close(fig)
    log.info("v2 figures saved to %s", FIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlases", nargs="*", type=int, default=[100, 200],
                    help="parcel counts to benchmark")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--skip-two-stage", action="store_true")
    ap.add_argument("--skip-full-refit", action="store_true")
    args = ap.parse_args()
    log = setup_logging()
    t0 = time.time()
    log.info("=== Benchmark v2: nested CV, two-stage, tuned, weights saved ===")
    from sklearn.model_selection import StratifiedKFold
    all_results = {"version": 2, "task": "3-class HC vs PD-NC vs PD-MCI",
                   "n": 55, "seed": SEED, "n_folds": N_FOLDS,
                   "protocol": "Nested CV: outer stratified 5-fold (seed 42, "
                   "same splits as v1); inner stratified 3-fold for all "
                   "scaling, feature selection and hyperparameter search. "
                   "Neural models: max 1000 epochs, early stopping "
                   "(patience 50). Preprocessing v2: motion correction, "
                   "Friston-24 regressors, FD scrubbing, high-variance "
                   "confounds, Schaefer-100 and Schaefer-200.",
                   "atlases": {}}
    # Resume: a service restart can kill this run mid-atlas, so results
    # are checkpointed after every step to metrics_v2_partial.json and
    # picked up again on the next start. A corrupt partial (killed
    # mid-write) falls back to the per-atlas metrics_v2.json.
    def save_partial():
        with open(BASE / "metrics_v2_partial.json", "w") as f:
            json.dump(all_results, f, indent=2)

    for src in ("metrics_v2_partial.json", "metrics_v2.json"):
        p = BASE / src
        if p.exists():
            try:
                saved = json.load(open(p))
                if isinstance(saved, dict) and saved.get("atlases"):
                    all_results["atlases"] = saved["atlases"]
                    log.info("resumed from %s: atlases present %s",
                             src, list(saved["atlases"].keys()))
                    break
            except Exception:
                log.warning("could not read %s, starting fresh", src)
    for n_parcels in args.atlases:
        tag = f"atlas{n_parcels}"
        X, mats, y3, yb, subjects = load_data(log, n_parcels)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=SEED)
        splits = list(skf.split(X, y3))
        res = all_results["atlases"].get(tag, {"models": {}})
        res.setdefault("models", {})
        classical_names = {"logistic_regression", "svm_rbf",
                           "random_forest", "xgboost"}
        if not classical_names <= set(res["models"]):
            classical, _ = run_classical_nested(X, y3, splits, 3, log, tag)
            res["models"].update(classical)
            all_results["atlases"][tag] = res
            save_partial()
        else:
            log.info("[%s] classical 3-class already done, skipping", tag)
        if not args.skip_torch:
            for kind in ["mlp", "cnn1d", "gcn", "gat"]:
                if kind in res["models"]:
                    log.info("[%s %s] already done, skipping", tag, kind)
                    continue
                try:
                    res["models"][kind] = run_torch_nested(
                        kind, X, mats, y3, splits, 3, log, tag)
                except Exception:
                    log.exception("[%s %s] FAILED", tag, kind)
                    res["models"][kind] = {"status": "failed", "kind": "neural"}
                all_results["atlases"][tag] = res
                save_partial()
        # binary HC vs PD (nested, all classical + torch best by 3-class)
        bin_splits = list(skf.split(X, yb))
        if "binary_models" not in res or not classical_names <= set(
                res.get("binary_models", {})):
            bin_classical, _ = run_classical_nested(
                X, yb, bin_splits, 2, log, tag + "/binary")
            res["binary_models"] = bin_classical
            all_results["atlases"][tag] = res
            save_partial()
        else:
            log.info("[%s] binary classical already done, skipping", tag)
        if not args.skip_torch:
            # binary versions of the best TORCH models only (classical
            # models already have binary results from the step above;
            # ranking across all models would pick classical names here
            # and skip them as "already done").
            torch_kinds = {"mlp", "cnn1d", "gcn", "gat"}
            ranked = sorted(
                ((k, v) for k, v in res["models"].items()
                 if "balanced_accuracy" in v and k in torch_kinds),
                key=lambda kv: kv[1]["balanced_accuracy"]["mean"],
                reverse=True)
            for k, _ in ranked[:3]:
                if k in res.get("binary_models", {}):
                    log.info("[%s binary %s] already done, skipping",
                             tag, k)
                    continue
                try:
                    res.setdefault("binary_models", {})[k] = \
                        run_torch_nested(k, X, mats, yb, bin_splits, 2,
                                         log, tag + "/binary")
                except Exception:
                    log.exception("[%s binary %s] FAILED", tag, k)
                all_results["atlases"][tag] = res
                save_partial()
        if not args.skip_two_stage:
            if "two_stage" not in res:
                res["two_stage"] = run_two_stage(X, y3, yb, splits, log, tag)
                all_results["atlases"][tag] = res
                save_partial()
            else:
                log.info("[%s] two-stage already done, skipping", tag)
        if not args.skip_full_refit:
            if "saved_full_models" not in res:
                res["saved_full_models"] = {}
                res["saved_full_models"]["classical"] = refit_full_classical(
                    X, y3, 3, log, tag)
                if not args.skip_torch:
                    res["saved_full_models"]["torch"] = {}
                    for kind in ["mlp", "cnn1d", "gcn", "gat"]:
                        if res["models"].get(kind, {}).get("status") \
                                == "skipped":
                            log.info("[%s full %s] skipped (model skipped "
                                     "in training)", tag, kind)
                            continue
                        try:
                            res["saved_full_models"]["torch"][kind] = \
                                refit_full_torch(kind, X, mats, y3, 3, log, tag)
                        except Exception:
                            log.exception("[%s full %s] FAILED", tag, kind)
                all_results["atlases"][tag] = res
                save_partial()
            else:
                log.info("[%s] full refits already done, skipping", tag)
        all_results["atlases"][tag] = res
        # checkpoint after each atlas so a restart never wipes results
        with open(BASE / "metrics_v2.json", "w") as f:
            json.dump(all_results, f, indent=2)
        log.info("Checkpoint metrics_v2.json written (%s done)", tag)

    save_figures_v2(all_results["atlases"], log)
    all_results["total_seconds"] = round(time.time() - t0, 1)
    all_results["total_minutes"] = round((time.time() - t0) / 60, 1)
    with open(BASE / "metrics_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("=== DONE v2 in %.1f min. metrics_v2.json written ===",
             (time.time() - t0) / 60)
    for tag, res in all_results["atlases"].items():
        ranked = sorted(
            ((k, v["balanced_accuracy"]["mean"])
             for k, v in res["models"].items() if "balanced_accuracy" in v),
            key=lambda kv: kv[1], reverse=True)
        log.info("[%s] ranking by balacc: %s", tag, ranked)


if __name__ == "__main__":
    raise SystemExit(main())
