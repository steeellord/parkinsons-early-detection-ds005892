#!/usr/bin/env python3
"""Benchmarking for Parkinson's early detection (ds005892).

Features: Fisher-z functional connectivity, Schaefer 100 parcels,
upper-triangle vector (4950) for vector models; full 100x100 matrix for GCN/GAT.
Protocol: stratified 5-fold CV, seed 42, SAME splits for every model.
Neural models: max 1000 epochs, early stopping (patience 50) on an inner
stratified validation split (15% of the training fold).
Metrics per fold: accuracy, balanced accuracy, macro-AUC (OvR), macro-F1,
plus summed confusion matrix. Binary HC-vs-PD rerun for the best 3 models.
"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
FEAT = BASE / "features"
LOG = BASE / "logs" / "training.log"
SEED = 42
N_FOLDS = 5


def setup_logging():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("train")


def load_data(log):
    import pandas as pd
    labels = pd.read_csv(FEAT / "labels.csv")
    mats, ys3, ysb = [], [], []
    for _, r in labels.iterrows():
        m = np.load(FEAT / f"{r['subject']}_fc_fisherz.npy")
        mats.append(m)
        ys3.append(int(r["label3"]))
        ysb.append(int(r["label_bin"]))
    mats = np.stack(mats)
    # Fisher-z diagonal is arctanh(0.9999)=4.95 (clipping artefact) — set to 0
    # so GCN node features are not dominated by the self-correlation.
    for i in range(mats.shape[0]):
        np.fill_diagonal(mats[i], 0.0)
    iu = np.triu_indices(100, k=1)
    X = mats[:, iu[0], iu[1]]
    log.info("Loaded %d subjects: X %s, mats %s", len(labels), X.shape, mats.shape)
    log.info("Class counts (3-class): %s",
             {int(k): int(v) for k, v in zip(*np.unique(ys3, return_counts=True))})
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
    auc_s = ("%.3f±%.3f" % (s["auc"]["mean"], s["auc"]["std"])
             if s["auc"]["folds"] else "n/a")
    log.info("[%s] acc=%.3f±%.3f balacc=%.3f±%.3f auc=%s f1=%.3f±%.3f",
             name, s["accuracy"]["mean"], s["accuracy"]["std"],
             s["balanced_accuracy"]["mean"], s["balanced_accuracy"]["std"],
             auc_s, s["macro_f1"]["mean"], s["macro_f1"]["std"])
    log.info("[%s] confusion (rows=true, cols=pred):\n%s", name, cm)
    return s


# ---------------- classical models ----------------
def run_classical(X, y, splits, n_classes, log):
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                               random_state=SEED)),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                class_weight="balanced", random_state=SEED)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=2),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=SEED, n_jobs=2)
        log.info("XGBoost available")
    except Exception as e:
        log.warning("XGBoost unavailable, skipped: %s", e)

    results = {}
    for name, model in models.items():
        t0 = time.time()
        per_fold = []
        for tr, te in splits:
            m = clone(model)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            proba = (m.predict_proba(X[te]) if hasattr(m, "predict_proba")
                     else np.eye(n_classes)[pred])
            per_fold.append(fold_metrics(y[te], pred, proba, n_classes))
        results[name] = summarize(per_fold, log, name)
        results[name]["seconds"] = round(time.time() - t0, 1)
        results[name]["per_fold"] = per_fold
        results[name]["kind"] = "classical"
    return results


# ---------------- torch models ----------------
def build_adj(fc, thresh=0.2):
    """Per-subject adjacency from Fisher-z FC: positive edges > thresh + self-loops,
    symmetrically normalised D^-1/2 A D^-1/2."""
    A = (fc > thresh).astype(np.float32)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    return (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]).astype(np.float32)


def train_torch_kind(kind, X, mats, y, splits, n_classes, log):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.model_selection import StratifiedShuffleSplit

    torch.set_num_threads(2)
    device = torch.device("cpu")

    class MLP(nn.Module):
        def __init__(self, d_in, n_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_out))

        def forward(self, x):
            return self.net(x)

    class CNN1D(nn.Module):
        def __init__(self, d_in, n_out):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
            n_feat = 64 * (d_in // 4)
            self.fc = nn.Sequential(
                nn.Linear(n_feat, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_out))

        def forward(self, x):
            h = self.conv(x.unsqueeze(1))
            return self.fc(h.flatten(1))

    class GCNLayer(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.lin = nn.Linear(d_in, d_out)

        def forward(self, h, a_norm):
            return F.relu(a_norm @ self.lin(h))

    class GCN(nn.Module):
        def __init__(self, n_out):
            super().__init__()
            self.l1 = GCNLayer(100, 64)
            self.l2 = GCNLayer(64, 32)
            self.out = nn.Linear(32, n_out)
            self.drop = nn.Dropout(0.3)

        def forward(self, h, a_norm):
            # h: (n_graphs, 100, 100) node features = FC rows
            h = self.drop(self.l1(h, a_norm))
            h = self.drop(self.l2(h, a_norm))
            return self.out(h.mean(dim=1))

    class GATLayer(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.W = nn.Linear(d_in, d_out, bias=False)
            self.a = nn.Parameter(torch.empty(2 * d_out))
            nn.init.xavier_uniform_(self.a.view(1, -1))
            self.leaky = nn.LeakyReLU(0.2)

        def forward(self, h, adj_mask):
            # h: (B, N, d_in); adj_mask: (B, N, N) bool
            Wh = self.W(h)                                   # (B,N,d_out)
            B, N, D = Wh.shape
            Wh_i = Wh.unsqueeze(2).expand(B, N, N, D)
            Wh_j = Wh.unsqueeze(1).expand(B, N, N, D)
            e = self.leaky(torch.cat([Wh_i, Wh_j], dim=-1) @ self.a)
            e = e.masked_fill(~adj_mask, -1e9)
            alpha = torch.softmax(e, dim=2)
            return F.elu(alpha @ Wh)

    class GAT(nn.Module):
        def __init__(self, n_out):
            super().__init__()
            self.l1 = GATLayer(100, 32)
            self.l2 = GATLayer(32, 32)
            self.out = nn.Linear(32, n_out)
            self.drop = nn.Dropout(0.3)

        def forward(self, h, a_norm):
            mask = a_norm > 0
            h = self.drop(self.l1(h, mask))
            h = self.drop(self.l2(h, mask))
            return self.out(h.mean(dim=1))

    makers = {"mlp": lambda: MLP(X.shape[1], n_classes),
              "cnn1d": lambda: CNN1D(X.shape[1], n_classes),
              "gcn": lambda: GCN(n_classes),
              "gat": lambda: GAT(n_classes)}
    graph_kind = kind in ("gcn", "gat")
    per_fold, histories, best_epochs = [], [], []
    t0 = time.time()
    for fold, (tr, te) in enumerate(splits):
        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)
        # inner stratified val split for early stopping
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15,
                                     random_state=SEED)
        itr, iva = next(sss.split(X[tr], y[tr]))
        tr_idx, va_idx = tr[itr], tr[iva]
        mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-8
        Xs = (X - mu) / sd
        # class weights from train fold
        counts = np.bincount(y[tr_idx], minlength=n_classes).astype(np.float32)
        w = len(y[tr_idx]) / (n_classes * np.maximum(counts, 1))
        cw = torch.tensor(w, dtype=torch.float32)
        model = makers[kind]().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                               weight_decay=1e-4 if not graph_kind else 5e-4)
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        if graph_kind:
            A_tr = torch.tensor(np.stack([build_adj(m) for m in mats[tr_idx]]))
            A_va = torch.tensor(np.stack([build_adj(m) for m in mats[va_idx]]))
            A_te = torch.tensor(np.stack([build_adj(m) for m in mats[te]]))
            H_tr = torch.tensor(mats[tr_idx], dtype=torch.float32)
            H_va = torch.tensor(mats[va_idx], dtype=torch.float32)
            H_te = torch.tensor(mats[te], dtype=torch.float32)
        else:
            H_tr = torch.tensor(Xs[tr_idx], dtype=torch.float32)
            H_va = torch.tensor(Xs[va_idx], dtype=torch.float32)
            H_te = torch.tensor(Xs[te], dtype=torch.float32)
        y_tr = torch.tensor(y[tr_idx])
        y_va = torch.tensor(y[va_idx])
        y_te = torch.tensor(y[te])

        def run_batch(model, H, A, yb, train, bs=16):
            n = H.shape[0]
            if graph_kind or n <= bs:
                if train:
                    model.train()
                    opt.zero_grad()
                    out = model(H, A) if graph_kind else model(H)
                    loss = loss_fn(out, yb)
                    loss.backward()
                    opt.step()
                    return loss.item(), out.detach()
                model.eval()
                with torch.no_grad():
                    out = model(H, A) if graph_kind else model(H)
                    return loss_fn(out, yb).item(), out
            # minibatches for vector models
            tot, outs = 0.0, []
            idx = torch.randperm(n) if train else torch.arange(n)
            model.train() if train else model.eval()
            for i in range(0, n, bs):
                b = idx[i:i + bs]
                if train:
                    opt.zero_grad()
                    out = model(H[b])
                    loss = loss_fn(out, yb[b])
                    loss.backward()
                    opt.step()
                else:
                    with torch.no_grad():
                        out = model(H[b])
                        loss = loss_fn(out, yb[b])
                tot += loss.item() * len(b)
                outs.append(out.detach())
            return tot / n, torch.cat(outs)

        best_val, best_state, best_ep, wait = np.inf, None, 0, 0
        hist = {"train_loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(1000):
            tr_loss, _ = run_batch(model, H_tr,
                                   A_tr if graph_kind else None, y_tr, True)
            va_loss, va_out = run_batch(model, H_va,
                                        A_va if graph_kind else None, y_va, False)
            va_acc = (va_out.argmax(1).numpy() == y[va_idx]).mean()
            hist["train_loss"].append(round(tr_loss, 4))
            hist["val_loss"].append(round(va_loss, 4))
            hist["val_acc"].append(round(float(va_acc), 4))
            if va_loss < best_val - 1e-4:
                best_val, best_ep, wait = va_loss, epoch, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 50:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(H_te, A_te) if graph_kind else model(H_te)
            proba = torch.softmax(out, dim=1).numpy()
        pred = proba.argmax(1)
        per_fold.append(fold_metrics(y[te], pred, proba, n_classes))
        histories.append(hist)
        best_epochs.append(best_ep)
        log.info("[%s fold %d] stopped at epoch %d (best %d), "
                 "test acc=%.3f balacc=%.3f",
                 kind, fold, epoch, best_ep,
                 per_fold[-1]["accuracy"], per_fold[-1]["balanced_accuracy"])
    res = summarize(per_fold, log, kind)
    res["seconds"] = round(time.time() - t0, 1)
    res["per_fold"] = per_fold
    res["kind"] = "neural"
    res["best_epochs"] = best_epochs
    res["histories"] = histories
    return res


# ---------------- figures ----------------
def save_figures(results, class_names, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = BASE / "figures"
    fig_dir.mkdir(exist_ok=True)
    for name, r in results.items():
        cm = np.array(r["confusion_matrix_sum"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name} — summed CV confusion matrix (n=55)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(fig_dir / f"cm_{name}.png", dpi=150)
        plt.close(fig)
    # training curves (mean val loss across folds) for neural models
    for name, r in results.items():
        if r.get("kind") != "neural":
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
        ax.set_title(f"{name} — training curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"curves_{name}.png", dpi=150)
        plt.close(fig)
    log.info("Figures saved to %s", fig_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset to run, e.g. svm_rbf mlp gcn")
    ap.add_argument("--skip-binary", action="store_true")
    args = ap.parse_args()
    log = setup_logging()
    t0 = time.time()
    log.info("=== Training benchmark ds005892: 5-fold stratified CV, seed 42 ===")
    X, mats, y3, yb, subjects = load_data(log)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X, y3))
    log.info("Splits (test sizes): %s", [len(te) for _, te in splits])

    results = {"task": "3-class HC vs PD-NC vs PD-MCI", "n": int(len(y3)),
               "seed": SEED, "n_folds": N_FOLDS, "models": {}}
    classical = run_classical(X, y3, splits, 3, log)
    results["models"].update(classical)

    torch_names = ["mlp", "cnn1d", "gcn", "gat"]
    wanted = args.models
    for kind in torch_names:
        if wanted and kind not in wanted:
            continue
        try:
            results["models"][kind] = train_torch_kind(
                kind, X, mats, y3, splits, 3, log)
        except Exception:
            log.exception("[%s] FAILED", kind)
            results["models"][kind] = {"status": "failed", "kind": "neural"}

    # Checkpoint: persist 3-class results NOW so a VM restart cannot wipe
    # them (a previous run was killed mid-binary with no metrics.json).
    with open(BASE / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Checkpoint metrics.json written (3-class only, binary pending)")

    # binary HC vs PD for the best 3 models by balanced accuracy
    if not args.skip_binary:
        ranked = sorted(
            ((k, v) for k, v in results["models"].items()
             if "balanced_accuracy" in v),
            key=lambda kv: kv[1]["balanced_accuracy"]["mean"], reverse=True)
        top3 = [k for k, _ in ranked[:3]]
        log.info("Binary HC-vs-PD rerun for best 3: %s", top3)
        bin_results = {}
        # classical top models: reuse run_classical on yb (full set, then filter)
        bin_all = run_classical(X, yb, list(skf.split(X, yb)), 2, log)
        for k in top3:
            if k in bin_all:
                bin_results[k] = bin_all[k]
        for k in top3:
            if k in torch_names:
                try:
                    bin_results[k] = train_torch_kind(k, X, mats, yb,
                                                      list(skf.split(X, yb)),
                                                      2, log)
                except Exception:
                    log.exception("[binary %s] FAILED", k)
        results["binary"] = {"task": "binary HC vs PD (PD-NC+PD-MCI)",
                             "models": bin_results}

    save_figures({k: v for k, v in results["models"].items()
                  if "confusion_matrix_sum" in v}, ["HC", "PD-NC", "PD-MCI"], log)
    if "binary" in results:
        save_figures({k: v for k, v in results["binary"]["models"].items()
                      if "confusion_matrix_sum" in v}, ["HC", "PD"], log)

    results["total_seconds"] = round(time.time() - t0, 1)
    results["total_minutes"] = round((time.time() - t0) / 60, 1)
    with open(BASE / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("=== DONE in %.1f min. metrics.json written ===",
             (time.time() - t0) / 60)
    # ranking summary
    ranked = sorted(
        ((k, v["balanced_accuracy"]["mean"]) for k, v in results["models"].items()
         if "balanced_accuracy" in v), key=lambda kv: kv[1], reverse=True)
    log.info("Ranking by balanced accuracy: %s", ranked)


if __name__ == "__main__":
    raise SystemExit(main())
