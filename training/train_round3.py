#!/usr/bin/env python3
"""Round 3: site harmonisation (ComBat) on the pooled data.

The question: does removing scanner/site fingerprints let a model
trained on pooled data work on a site it never saw (LOSO above
chance)? Round 2 pooled 5-fold CV scored 0.60-0.62 balanced
accuracy but LOSO sat at chance — the model was reading each
scanner's fingerprint, not the disease.

Methods, all under the SAME nested protocol as round 2 (outer
stratified 5-fold seed 42 + leave-one-source-out; inner 3-fold
GridSearchCV; same 4 classical models and grids):
  combat        ComBat (empirical Bayes, Fortin et al. 2018), design =
                diagnosis + age + sex. Fit ONLY on the outer-train
                fold; the held-out fold is transformed with its own
                batch's fitted parameters. A batch never seen in
                training (the LOSO held-out site) gets no correction
                at all — that is the honest leak-free rule, and it
                is the point of the test.
  combat_trans  ComBat with a batch-only design, fit on ALL subjects
                before CV. No labels anywhere in the fit, so nothing
                about the outcome leaks — but the test features do
                shape the batch estimates, so this is transductive.
                It can only show an upper bound and is reported
                separately, never as the headline.
  zscore        Per-site z-scoring: each site standardised on its
                own training subjects; an unseen site falls back to
                the global training mean/std.

The ComBat here is implemented from the paper (Fortin et al.,
NeuroImage 2018, building on Johnson & Li 2007) with an explicit
fit/transform split, because the published package only does
fit-and-transform in one call. It is cross-checked against
neuroCombat 0.2.12 on the full data — see validate_combat().

Outputs: training/metrics_round3.json (checkpointed after every
scheme, so a crash never wipes a finished scheme), training/
figures_round3/, training/models_round3/ (best harmonised model,
full-data refit).
"""
import argparse, importlib.util, json, logging, sys, time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
POOL = BASE / "features_pooled"
LOG = BASE / "logs" / "training_round3.log"
FIG = BASE / "figures_round3"
MODELS = BASE / "models_round3"
OUT = BASE / "metrics_round3.json"
SEED = 42
N_FOLDS = 5

# round-2 baseline, reused as-is (same splits, same grids, same code
# path for the classical runners — the harness check below proves it
# by reproducing the round-2 LR pooled numbers exactly).
BASELINE = BASE / "metrics_pooled.json"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tv2 = _load_module("tv2", BASE / "train_v2.py")
tp = _load_module("tp", BASE / "train_pooled.py")


def setup_logging():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stream = open(LOG, "a", buffering=1)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(stream),
                  logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("train_round3")


def load_covariates(subjects):
    """Age/sex per subject, aligned to `subjects` order."""
    import pandas as pd
    cov = pd.read_csv(POOL / "covariates.csv")
    cov = cov.set_index("subject").loc[subjects]
    age = cov["age"].to_numpy(dtype=float)
    sex = (cov["sex"].to_numpy() == "M").astype(float)
    return age, sex


# ---------------- ComBat (empirical Bayes) ----------------
# Follows Fortin et al. 2018 / Johnson & Li 2007, as implemented in
# sva::ComBat and neuroCombat. Y is features x samples; the model is
#   Y_g = alpha_g + X beta_g + gamma_batch + delta_batch * eps
# Standardisation fits OLS on [batch dummies, covariates] and
# subtracts the grand mean (mean of the batch coefficients) plus the
# covariate effects; the per-batch location/scale left in the
# standardised data are then shrunk with empirical-Bayes priors.
# A fitted object can transform new data; a batch never seen in
# fit is left untouched.

def _covariates(y, age, sex, use_covariates, age_mean=None):
    """Covariate block (no intercept — the batch dummies absorb it)."""
    if not use_covariates:
        return np.zeros((len(y), 0))
    mean = age.mean() if age_mean is None else age_mean
    return np.column_stack([y.astype(float), age - mean, sex])


def _standardize(Y, batch, B_cov):
    """Return Z, grand_mean, beta_cov, sigma (see header)."""
    uniq = sorted(set(batch.tolist()))
    n = Y.shape[1]
    D_batch = np.zeros((n, len(uniq)))
    for bi, b in enumerate(uniq):
        D_batch[batch == b, bi] = 1.0
    D = np.hstack([D_batch, B_cov]) if B_cov.shape[1] else D_batch
    beta_all = np.linalg.pinv(D.T @ D) @ D.T @ Y.T
    # grand mean = SAMPLE-WEIGHTED mean of the batch coefficients
    # (sva/neuroCombat weight by batch size; an unweighted mean is a
    # different number when batches differ in size, as here)
    counts = np.array([float((batch == b).sum()) for b in uniq])
    grand_mean = (counts / counts.sum()) @ beta_all[:len(uniq)]
    beta_cov = beta_all[len(uniq):]
    resid = Y - (D @ beta_all).T
    sigma = np.sqrt(np.maximum((resid ** 2).mean(axis=1), 1e-8))
    Z = (Y - grand_mean[:, None] - (B_cov @ beta_cov).T) / sigma[:, None]
    return Z, grand_mean, beta_cov, sigma


def _eb_priors(gamma_hat, delta_hat):
    """Method-of-moments priors, per batch (rows of the inputs)."""
    gamma_bar = gamma_hat.mean(axis=1)
    tau2 = gamma_hat.var(axis=1, ddof=1)
    tau2 = np.maximum(tau2, 1e-6)
    m = delta_hat.mean(axis=1)
    v = delta_hat.var(axis=1, ddof=1)
    v = np.maximum(v, 1e-6)
    lam = 2.0 + m ** 2 / v
    theta = m * (lam - 1.0)
    return gamma_bar, tau2, lam, theta


def _eb_posterior(Z, batches, uniq, gamma_bar, tau2, lam, theta, iters=50):
    """Iterative posterior location/scale per batch.

    Z: features x samples (standardised). Returns gamma_star and
    delta_star, each (n_batches, n_features)."""
    n_batches = len(uniq)
    n_feat = Z.shape[0]
    gamma_star = np.zeros((n_batches, n_feat))
    delta_star = np.zeros((n_batches, n_feat))
    for bi, b in enumerate(uniq):
        idx = np.where(batches == b)[0]
        Zb = Z[:, idx]
        n_i = len(idx)
        gh = Zb.mean(axis=1)
        dh = np.maximum(Zb.var(axis=1, ddof=1), 1e-6)
        g_star = gh.copy()
        d_star = dh.copy()
        for _ in range(iters):
            g_new = ((n_i * tau2[bi] * gh + d_star * gamma_bar[bi])
                     / (n_i * tau2[bi] + d_star))
            d_new = (theta[bi] + 0.5 * ((Zb - g_new[:, None]) ** 2).sum(axis=1)) \
                / (lam[bi] + 0.5 * n_i - 1.0)
            d_new = np.maximum(d_new, 1e-6)
            if np.max(np.abs(g_new - g_star)) < 1e-6 and \
                    np.max(np.abs(d_new - d_star)) < 1e-6:
                g_star, d_star = g_new, d_new
                break
            g_star, d_star = g_new, d_new
        gamma_star[bi] = g_star
        delta_star[bi] = d_star
    return gamma_star, delta_star


class CombatFit:
    """Fitted ComBat parameters; transform() applies them to new data."""

    def __init__(self, grand_mean, beta_cov, sigma, gamma_star,
                 delta_star, batches_seen, use_covariates, age_mean):
        self.grand_mean = grand_mean
        self.beta_cov = beta_cov
        self.sigma = sigma
        self.gamma_star = gamma_star
        self.delta_star = delta_star
        self.batches_seen = list(batches_seen)
        self.use_covariates = use_covariates
        self.age_mean = age_mean

    def transform(self, Y, batch, y, age, sex):
        """Y: features x samples. Batches never seen in fit are left
        untouched (identity) — no parameters exist for them and
        guessing would be fabrication."""
        B_cov = _covariates(y, age, sex, self.use_covariates,
                            age_mean=self.age_mean)
        fitted = self.grand_mean[:, None] + (B_cov @ self.beta_cov).T
        Z = (Y - fitted) / self.sigma[:, None]
        out = np.empty_like(Z)
        for j, b in enumerate(batch):
            if b in self.batches_seen:
                bi = self.batches_seen.index(b)
                out[:, j] = ((Z[:, j] - self.gamma_star[bi])
                             / np.sqrt(self.delta_star[bi]))
            else:
                out[:, j] = Z[:, j]
        return out * self.sigma[:, None] + fitted


def fit_combat(Y, batch, y, age, sex, use_covariates=True):
    """Fit ComBat on the given samples. Y: features x samples."""
    uniq = sorted(set(batch.tolist()))
    B_cov = _covariates(y, age, sex, use_covariates)
    Z, grand_mean, beta_cov, sigma = _standardize(Y, batch, B_cov)
    gamma_hat = np.zeros((len(uniq), Y.shape[0]))
    delta_hat = np.zeros((len(uniq), Y.shape[0]))
    for bi, b in enumerate(uniq):
        idx = np.where(batch == b)[0]
        gamma_hat[bi] = Z[:, idx].mean(axis=1)
        delta_hat[bi] = np.maximum(Z[:, idx].var(axis=1, ddof=1), 1e-6)
    gamma_bar, tau2, lam, theta = _eb_priors(gamma_hat, delta_hat)
    gamma_star, delta_star = _eb_posterior(Z, batch, uniq, gamma_bar,
                                          tau2, lam, theta)
    return CombatFit(grand_mean, beta_cov, sigma, gamma_star, delta_star,
                     uniq, use_covariates, float(age.mean()))


def validate_combat(X, batch, y, age, sex, log):
    """Cross-check my ComBat against neuroCombat on the full data.
    Both use design = diagnosis + age + sex. Reports the largest
    absolute difference between the two harmonised matrices."""
    from neuroCombat import neuroCombat
    import pandas as pd
    Y = X.T  # features x samples
    mine = fit_combat(Y, batch, y, age, sex, use_covariates=True)
    mine_h = mine.transform(Y, batch, y, age, sex).T
    covars = pd.DataFrame({"batch": batch, "dx": y.astype(int),
                           "age": age, "sex": sex})
    ref = neuroCombat(dat=Y, covars=covars, batch_col="batch",
                      categorical_cols=["dx", "sex"],
                      continuous_cols=["age"])["data"].T
    diff = np.abs(mine_h - ref)
    log.info("[validate] ComBat vs neuroCombat: max|diff|=%.2e "
             "mean|diff|=%.2e corr=%.6f",
             diff.max(), diff.mean(),
             np.corrcoef(mine_h.ravel(), ref.ravel())[0, 1])
    return float(diff.max())


# ---------------- per-site z-scoring ----------------
def zscore_fit_transform(Xtr, batch_tr, Xte, batch_te):
    """Standardise each site on its own training subjects. A test
    site with no training subjects falls back to the global
    training mean/std."""
    mu_g = Xtr.mean(axis=0)
    sd_g = Xtr.std(axis=0)
    sd_g = np.where(sd_g < 1e-8, 1.0, sd_g)
    out_tr = np.empty_like(Xtr)
    out_te = np.empty_like(Xte)
    for b in sorted(set(batch_tr.tolist())):
        idx = np.where(batch_tr == b)[0]
        mu = Xtr[idx].mean(axis=0)
        sd = Xtr[idx].std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        out_tr[idx] = (Xtr[idx] - mu) / sd
        te_idx = np.where(batch_te == b)[0]
        if len(te_idx):
            out_te[te_idx] = (Xte[te_idx] - mu) / sd
    unseen = np.where(~np.isin(batch_te, list(set(batch_tr.tolist()))))[0]
    if len(unseen):
        out_te[unseen] = (Xte[unseen] - mu_g) / sd_g
    return out_tr, out_te


# ---------------- nested-CV harness ----------------
def run_models_nested(X, y, outer_splits, log, tag, transform=None):
    """Same nested protocol as train_v2.run_classical_nested, except
    an optional transform(tr_idx, te_idx) -> (Xtr, Xte) is applied
    per outer fold first — that is where harmonisation lives, fit
    on the training fold only. Returns (results, fitted pipelines)."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    grids = tv2.classical_grids(X.shape[1])
    results, fitted = {}, {}
    for name, (pipe, grid) in grids.items():
        t0 = time.time()
        per_fold, best_params, fold_models = [], [], []
        for fold, (tr, te) in enumerate(outer_splits):
            if transform is not None:
                Xtr, Xte = transform(tr, te)
            else:
                Xtr, Xte = X[tr], X[te]
            gs = GridSearchCV(pipe, grid, cv=inner,
                              scoring="balanced_accuracy", n_jobs=1)
            gs.fit(Xtr, y[tr])
            best = gs.best_estimator_
            fold_models.append(best)
            best_params.append(
                {k: (v.tolist() if hasattr(v, "tolist") else v)
                 for k, v in gs.best_params_.items()})
            pred = best.predict(Xte)
            proba = (best.predict_proba(Xte)
                     if hasattr(best, "predict_proba")
                     else np.eye(2)[pred])
            per_fold.append(tv2.fold_metrics(y[te], pred, proba, 2))
            log.info("[%s %s fold %d] best=%s balacc=%.3f",
                     tag, name, fold, gs.best_params_,
                     per_fold[-1]["balanced_accuracy"])
        res = tv2.summarize(per_fold, log, f"{tag}/{name}")
        res["seconds"] = round(time.time() - t0, 1)
        res["per_fold"] = per_fold
        res["kind"] = "classical"
        res["best_params_per_fold"] = best_params
        results[name] = res
        fitted[name] = fold_models
    return results, fitted


def make_transform(method, X, batch, y, age, sex, log):
    """Return a per-outer-fold transform function for `method`."""
    if method == "raw":
        return None
    if method == "zscore":
        def trf(tr, te):
            return zscore_fit_transform(X[tr], batch[tr], X[te], batch[te])
        return trf
    if method == "combat":
        def trf(tr, te):
            fit = fit_combat(X[tr].T, batch[tr], y[tr], age[tr], sex[tr],
                             use_covariates=True)
            Xtr = fit.transform(X[tr].T, batch[tr], y[tr], age[tr],
                                sex[tr]).T
            Xte = fit.transform(X[te].T, batch[te], y[te], age[te],
                                sex[te]).T
            return Xtr, Xte
        return trf
    if method == "combat_trans":
        # batch-only ComBat fit on ALL subjects, before CV. No labels
        # anywhere in the fit; transductive, upper bound only.
        fit = fit_combat(X.T, batch, y, age, sex, use_covariates=False)
        Xh = fit.transform(X.T, batch, y, age, sex).T
        log.info("[combat_trans] pre-harmonised all %d subjects "
                 "(batch-only, no labels in fit)", len(y))
        def trf(tr, te):
            return Xh[tr], Xh[te]
        return trf
    raise ValueError(method)


def save_figures(schemes, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    for scheme, res in schemes.items():
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
    log.info("round-3 figures saved to %s", FIG)


def refit_best_full(X, y, batch, age, sex, method, model_name, log):
    """Full-data refit of the best model on harmonised data, saved
    for inference. Harmonisation is fit on all data here — the
    deployment artifact, not a CV estimate; the card says so."""
    import joblib
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    MODELS.mkdir(parents=True, exist_ok=True)
    trf = make_transform(method, X, batch, y, age, sex, log)
    if trf is None:
        Xh = X
        harm = {"method": "raw", "note": "no harmonisation"}
    elif method == "combat_trans":
        Xh, _ = trf(np.arange(len(y)), np.arange(len(y)))
        harm = {"method": "combat_trans",
                "note": "batch-only ComBat fit on all data"}
    elif method == "combat":
        fit = fit_combat(X.T, batch, y, age, sex, use_covariates=True)
        Xh = fit.transform(X.T, batch, y, age, sex).T
        harm = {"method": "combat", "batches": sorted(set(batch.tolist())),
                "note": "ComBat fit on all data for the deployment "
                        "artifact; CV estimates used train-only fits"}
    else:
        Xh, _ = trf(np.arange(len(y)), np.arange(len(y)))
        harm = {"method": "zscore",
                "note": "per-site z-score on all data for the "
                        "deployment artifact"}
    pipe, grid = tv2.classical_grids(X.shape[1])[model_name]
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grid, cv=inner, scoring="balanced_accuracy",
                      n_jobs=1)
    gs.fit(Xh, y)
    path = MODELS / f"round3_{method}_{model_name}_full.pkl"
    joblib.dump(gs.best_estimator_, path)
    log.info("[full %s %s] saved %s best=%s", method, model_name, path,
             gs.best_params_)
    return {"path": str(path), "model": model_name, "method": method,
            "harmonisation": harm,
            "best_params": {k: (v.tolist() if hasattr(v, "tolist") else v)
                            for k, v in gs.best_params_.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["combat", "zscore", "combat_trans"],
                    choices=["raw", "combat", "zscore", "combat_trans"])
    ap.add_argument("--schemes", nargs="+", default=["pooled", "loso"],
                    choices=["pooled", "loso"])
    ap.add_argument("--models", nargs="+",
                    default=["logistic_regression", "svm_rbf",
                             "random_forest", "xgboost"])
    ap.add_argument("--harness-check", action="store_true",
                    help="re-run raw LR pooled 5-fold and compare with "
                         "the round-2 baseline, then exit")
    ap.add_argument("--skip-full-refit", action="store_true")
    args = ap.parse_args()
    log = setup_logging()
    t0 = time.time()
    log.info("=== Round 3: ComBat harmonisation on pooled data ===")
    X, mats, yb, sources, subjects = tp.load_pooled(log)
    batch = np.array(sources)
    age, sex = load_covariates(subjects)
    log.info("covariates: age %.1f-%.1f, sex M=%d F=%d",
             age.min(), age.max(), int(sex.sum()), int((1 - sex).sum()))
    if args.harness_check:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=SEED)
        splits = list(skf.split(X, yb))
        # LR only: enough to prove the harness reproduces round 2.
        pipe, grid = tv2.classical_grids(X.shape[1])["logistic_regression"]
        res = {"logistic_regression": _run_one_model(
            X, yb, splits, log, "check/raw", None,
            "logistic_regression", pipe, grid)}
        got = res["logistic_regression"]["balanced_accuracy"]
        with open(BASELINE) as f:
            base = json.load(f)
        want = base["schemes"]["pooled_5fold"]["models"][
            "logistic_regression"]["balanced_accuracy"]
        log.info("[harness-check] raw LR pooled balacc mean=%.4f "
                 "(baseline %.4f) folds=%s (baseline %s)",
                 got["mean"], want["mean"], got["folds"], want["folds"])
        log.info("[harness-check] MATCH=%s",
                 got["folds"] == want["folds"])
        return 0
    # cross-check my ComBat against the published package once
    validate_combat(X, batch, yb, age, sex, log)
    out = {"version": 3,
           "task": "binary HC vs PD (pooled, harmonised)",
           "n": int(len(yb)), "seed": SEED, "n_folds": N_FOLDS,
           "sources": sorted(set(batch.tolist())),
           "protocol": "Nested CV: inner stratified 3-fold for all "
           "scaling, feature selection and hyperparameter search. "
           "Harmonisation is fit ONLY on each outer-train fold "
           "(combat/zscore); combat_trans is the labelled "
           "transductive exception (batch-only fit on all data, no "
           "labels). Pooled scheme: outer stratified 5-fold (seed "
           "42). LOSO scheme: each fold holds out one whole source.",
           "baseline": "round-2 unharmonised pooled numbers reused "
           "from metrics_pooled.json (same splits, same grids)",
           "schemes": {}}
    if OUT.exists():
        with open(OUT) as f:
            prev = json.load(f)
        out["schemes"] = prev.get("schemes", {})
        log.info("resumed from %s (schemes done: %s)", OUT,
                 list(out["schemes"]))
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=SEED)
    pooled_splits = list(skf.split(X, yb))
    loso_splits = []
    for src in sorted(set(batch.tolist())):
        te = np.where(batch == src)[0]
        tr = np.where(batch != src)[0]
        loso_splits.append((tr, te))
    splits = {"pooled": pooled_splits, "loso": loso_splits}
    for method in args.methods:
        trf = make_transform(method, X, batch, yb, age, sex, log)
        for scheme in args.schemes:
            key = f"{method}_{scheme}"
            if key not in out["schemes"]:
                out["schemes"][key] = {"models": {}}
            done = set(out["schemes"][key]["models"])
            log.info("=== [%s] %s scheme (done: %s) ===", method, scheme,
                     sorted(done) or "none")
            all_models = tv2.classical_grids(X.shape[1])
            wanted = {k: v for k, v in all_models.items()
                      if k in args.models and k not in done}
            for name, (pipe, grid) in wanted.items():
                res = _run_one_model(X, yb, splits[scheme], log, key,
                                     trf, name, pipe, grid)
                out["schemes"][key]["models"][name] = res
                with open(OUT, "w") as f:
                    json.dump(out, f, indent=2)
                log.info("[%s] checkpoint written (%s done)", key, name)
    # best harmonised model by LOSO balanced accuracy, then pooled
    best = None
    for key, sch in out["schemes"].items():
        if not key.startswith(("combat", "zscore")):
            continue
        for name, r in sch["models"].items():
            score = r["balanced_accuracy"]["mean"]
            if best is None or score > best[0]:
                best = (score, key, name)
    if best and not args.skip_full_refit:
        _, key, name = best
        if key.startswith("combat_trans"):
            method = "combat_trans"
        elif key.startswith("combat"):
            method = "combat"
        else:
            method = "zscore"
        saved = refit_best_full(X, yb, batch, age, sex, method, name, log)
        out["saved_full_model"] = saved
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2)
    save_figures(out["schemes"], log)
    out["total_seconds"] = round(time.time() - t0, 1)
    out["total_minutes"] = round((time.time() - t0) / 60, 1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log.info("=== DONE round 3 in %.1f min ===", (time.time() - t0) / 60)
    for key, sch in out["schemes"].items():
        ranked = sorted(
            ((k, v["balanced_accuracy"]["mean"])
             for k, v in sch["models"].items()),
            key=lambda kv: kv[1], reverse=True)
        log.info("[%s] ranking by balacc: %s", key, ranked)
    return 0


def _run_one_model(X, y, outer_splits, log, tag, transform, name, pipe,
                   grid):
    """One model through the nested protocol (see run_models_nested).
    Split out so the caller can checkpoint after every model — a
    runtime restart then costs at most one model, not a whole scheme."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    t0 = time.time()
    per_fold, best_params = [], []
    for fold, (tr, te) in enumerate(outer_splits):
        if transform is not None:
            Xtr, Xte = transform(tr, te)
        else:
            Xtr, Xte = X[tr], X[te]
        gs = GridSearchCV(pipe, grid, cv=inner,
                          scoring="balanced_accuracy", n_jobs=1)
        gs.fit(Xtr, y[tr])
        best = gs.best_estimator_
        best_params.append(
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in gs.best_params_.items()})
        pred = best.predict(Xte)
        proba = (best.predict_proba(Xte)
                 if hasattr(best, "predict_proba")
                 else np.eye(2)[pred])
        per_fold.append(tv2.fold_metrics(y[te], pred, proba, 2))
        log.info("[%s %s fold %d] best=%s balacc=%.3f",
                 tag, name, fold, gs.best_params_,
                 per_fold[-1]["balanced_accuracy"])
    res = tv2.summarize(per_fold, log, f"{tag}/{name}")
    res["seconds"] = round(time.time() - t0, 1)
    res["per_fold"] = per_fold
    res["kind"] = "classical"
    res["best_params_per_fold"] = best_params
    return res


if __name__ == "__main__":
    raise SystemExit(main())
