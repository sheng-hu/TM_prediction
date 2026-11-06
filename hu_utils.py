"""Utility functions for nested-CV classification of pancreatic cancer recurrence (used by train_eval.py)."""

import os
import warnings

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from scipy.stats import norm, rankdata

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def setup_gridsearch_model_clf(model, class_weight=None, cv=5, rs=42):
    """Set up GridSearchCV (inner CV, scoring="roc_auc") and parameter grids for a classifier.

    Args:
        model (str): "RFR", "ETR", "XGB", "LOGISTIC", "SVC" or "KNN".
        class_weight: passed to the classifier (not used by XGB / KNN).
        cv (int): number of inner folds (StratifiedKFold, shuffled).
        rs (int, optional): random state. Defaults to 42.

    Returns:
        GridSearchCV: object.
    """
    cv_object = None
    cvf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rs)
    if model == "RFR":
        cv_object = GridSearchCV(
            RandomForestClassifier(n_jobs=-1, n_estimators=200, 
                                   class_weight=class_weight ,random_state=rs),
            param_grid={
                "max_features": ["sqrt", 1.0, 'log2'],
                "min_samples_leaf": [1, 2, 5, 10],
                # "min_samples_split": [2, 5, 10],
                # "max_leaf_nodes": [None, 70, 200],
                "max_depth": [None, 5, 10, 20],
                # "max_depth": [ 5],
                "bootstrap": [True, False],
            },
            # scoring="accuracy",
            scoring="roc_auc",
            cv=cvf,
            n_jobs=-1,
        )
    elif model == "ETR":
        cv_object = GridSearchCV(
            ExtraTreesClassifier(n_jobs=-1, n_estimators=200, 
                                 class_weight=class_weight, random_state=rs),
            param_grid={
                "max_features": ["sqrt", 1.0],
                "min_samples_leaf": [1, 2, 5, 10],
                # "min_samples_split": [2, 5, 10],
                # "max_leaf_nodes": ["None", 70, 200],
                "max_depth": [None, 5, 10, 20],
                # "max_depth": [ 5],
                "bootstrap": [True, False],
            },
            # scoring="accuracy",
            scoring="roc_auc",
            cv=cvf,
            n_jobs=-1,
        )
    elif model == "XGB":
        cv_object = GridSearchCV(
            XGBClassifier(n_jobs=2, importance_type="total_gain", random_state=rs),
            param_grid={
                # "n_estimators": [200, 500, 1000, 1500],
                "n_estimators": [200],
                "max_depth": [6, 7, 8],
                # "max_depth": [6],
                # "learning_rate": [0.1, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6],
                "learning_rate": [0.1],
                # "subsample": [0.8, 0.9, 1],
                # "subsample": [1],
                # "colsample_bytree": [0.8, 0.9, 1],
                # "colsample_bytree": [0.9],
            },
            # scoring="accuracy",
            scoring="roc_auc",
            cv=cvf,
            n_jobs=-1,
        )
    elif model == "LOGISTIC":
        cv_object = GridSearchCV(
            Pipeline([
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression(class_weight=class_weight, 
                                           random_state=rs, max_iter=5000)),
            ]),
            param_grid=[
                {"clf__penalty": ["l1", "l2"],
                 "clf__solver": ["liblinear", "saga"],
                 "clf__C": np.logspace(-4, 4, 9)},
                {"clf__penalty": ["elasticnet"],
                 "clf__solver": ["saga"],
                 "clf__C": np.logspace(-4, 4, 9),
                 "clf__l1_ratio": np.linspace(0, 1, 5)},
                {"clf__penalty": [None],
                 "clf__solver": ["lbfgs"]},
            ],
            scoring="roc_auc",
            cv=cvf,
            n_jobs=-1,
        )
    elif model == "SVC":
        cv_object = GridSearchCV(
            # SVC(class_weight=class_weight, probability=True, random_state=rs),
            Pipeline([
                ("scaler", RobustScaler()),
                ("clf", SVC(class_weight=class_weight, probability=True, random_state=rs)),
            ]),
            param_grid={
                "clf__kernel": ['rbf'],
                "clf__C": np.logspace(-3, 3, 7),
                "clf__gamma": ['scale', 'auto', 1.0, 1e-1, 1e-2, 1e-3],
            },
            scoring="roc_auc",
            cv=cvf,
            n_jobs=-1,
        )
    elif model == "KNN":
        cv_object = GridSearchCV(
            Pipeline([
                ("scaler", RobustScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            param_grid={
                "clf__n_neighbors": [3, 5, 7, 9, 15, 21],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
            scoring="roc_auc",
            cv=cvf,
            n_jobs=-1,
        )
    else:
        warnings.warn("Machine Learning model cannot defined")
    return cv_object


def _fast_auc(y, score):
    """AUC via the Mann-Whitney U statistic (same value as roc_auc_score, much faster)."""
    n1 = y.sum()
    n0 = len(y) - n1
    r = rankdata(score)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rfecv_feature_stability(X, y, estimator=None, cv=5, rs=42, step=1, min_features_to_select=1, freq_threshold=0.6):
    """RFECV feature selection: once inside each outer fold (training part only) and once on the full cohort.

    The outer folds are StratifiedKFold(cv, shuffle=True, random_state=rs), identical to nested_cv_oof
    with the same cv/rs, so fold_features[k] can be passed to nested_cv_oof for fold k.

    Args:
        X (df): features.
        y (array-like): binary labels (0/1).
        estimator: must expose coef_ or feature_importances_. Defaults to RandomForest (500 trees, balanced).
        freq_threshold (float): features selected in >= this fraction of folds are reported as frequently selected.

    Returns:
        fold_features (list of lists): selected columns per outer fold.
        table (df): Table S4. Selection count/frequency per feature and whether kept in the full-cohort model.
        full_features (list): columns kept by RFECV refitted on the full cohort.
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=rs, n_jobs=1)
    y = np.asarray(y).astype(int)
    outer = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rs)
    inner = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rs)

    def _select(X_sub, y_sub):
        sel = RFECV(clone(estimator), step=step, cv=inner, scoring="roc_auc",
                    min_features_to_select=min_features_to_select, n_jobs=-1)
        sel.fit(X_sub, y_sub)
        return list(X_sub.columns[sel.support_])

    fold_features = [_select(X.iloc[trn], y[trn]) for trn, _ in outer.split(X, y)]
    full_features = _select(X, y)

    counts = pd.Series(0, index=X.columns)
    for feats in fold_features:
        counts[feats] += 1
    table = pd.DataFrame({
        "selected_folds": counts,
        "frequency": counts / cv,
        "in_full_cohort_model": X.columns.isin(full_features),
    })
    table.index.name = "feature"
    table = table.sort_values(["selected_folds", "in_full_cohort_model"], ascending=False)

    n_sel = [len(f) for f in fold_features]
    frequent = list(table.index[table["frequency"] >= freq_threshold])
    print(f"Retained features per outer fold: {n_sel}")
    print(f"The number of retained features ranged from {min(n_sel)} to {max(n_sel)}.")
    print(f"Selected in >= {freq_threshold:.0%} of folds: {frequent}")
    print(f"Full-cohort model retained {len(full_features)} features: {full_features}")
    return fold_features, table, full_features


def nested_cv_oof(model, X, y, class_weight=None, cv=5, rs=42, fold_features=None):
    """Nested CV. Outer StratifiedKFold for evaluation, inner GridSearchCV (setup_gridsearch_model_clf) for tuning.

    Every sample gets one out-of-fold (OOF) score from a model that never saw it, neither in
    fitting nor in hyperparameter selection. The OOF scores (positive-class probability; decision_function for SVC) feed bootstrap_fold_auc_ci and delong_vs_reference.

    Args:
        model (str): key of setup_gridsearch_model_clf, e.g. "RFR", "XGB", "ETR", "SVC", "KNN", "LOGISTIC".
        X (df): features.
        y (array-like): binary labels (0/1).
        fold_features (list of lists, optional): columns to use in each outer fold (from rfecv_feature_stability).

    Returns:
        dict: y, oof_score (rank-based use only: AUC, DeLong), oof_proba (predict_proba, used by soft_voting_oof),
              oof_pred (predicted labels from gs.predict), fold_id, fold_auc, best_params (per outer fold).
    """
    y = np.asarray(y).astype(int)
    outer = StratifiedKFold(n_splits=cv, shuffle=True, random_state=rs)
    oof_score = np.full(len(y), np.nan)
    oof_proba = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1)
    fold_id = np.full(len(y), -1)
    fold_auc, best_params = [], []
    for k, (trn, tst) in enumerate(outer.split(X, y)):
        cols = X.columns if fold_features is None else fold_features[k]
        gs = setup_gridsearch_model_clf(model, class_weight=class_weight, cv=cv, rs=rs)
        gs.fit(X.iloc[trn][cols], y[trn])
        est = gs.best_estimator_
        oof_proba[tst] = gs.predict_proba(X.iloc[tst][cols])[:, 1]
        if isinstance(est.steps[-1][1] if hasattr(est, "steps") else est, SVC):
            # Platt-scaled predict_proba can invert the ranking on small folds; decision_function is
            # what GridSearchCV's roc_auc scorer used for tuning.
            oof_score[tst] = gs.decision_function(X.iloc[tst][cols])
        else:
            oof_score[tst] = oof_proba[tst]
        oof_pred[tst] = gs.predict(X.iloc[tst][cols])
        fold_id[tst] = k
        fold_auc.append(roc_auc_score(y[tst], oof_score[tst]))
        best_params.append(gs.best_params_)
    fold_auc = np.array(fold_auc)
    print(f"{model}: fold AUC {np.round(fold_auc, 3)}, mean {fold_auc.mean():.3f} (std {fold_auc.std():.3f}), "
          f"pooled OOF AUC {roc_auc_score(y, oof_score):.3f}")
    return {"y": y, "oof_score": oof_score, "oof_proba": oof_proba, "oof_pred": oof_pred, "fold_id": fold_id,
            "fold_auc": fold_auc, "best_params": best_params}


def soft_voting_oof(res, members=None, weights=None):
    """Soft-voting ensemble from nested_cv_oof results: (weighted) mean of the members' OOF predict_proba.

    Within each outer fold every member was tuned and fitted on the outer-training part only, and the
    averaging has no fitted parameters, so the ensemble OOF predictions are still nested-CV estimates.
    Same output format as nested_cv_oof, so it can be added to res for summarize_nested_cv / DeLong.

    Args:
        res (dict): {model_name: output of nested_cv_oof}, all run with the same X, y, cv, rs.
        members (list, optional): models to average. Defaults to the keys of weights if it is a dict,
            otherwise all models in res.
        weights (dict or list, optional): non-negative weights, {model_name: w} or a list in the order of
            members. Normalized to sum to 1. None (default) gives equal weights.
            Weights must be fixed a priori; choosing them from these OOF results makes the ensemble optimistic.

    Returns:
        dict: y, oof_score (= oof_proba, averaged probability), oof_proba, oof_pred (averaged probability > 0.5),
              fold_id, fold_auc, best_params ({member: normalized weight}).
    """
    if members is None:
        members = list(weights) if isinstance(weights, dict) else list(res)
    members = list(members)
    if weights is None:
        w = np.ones(len(members))
    elif isinstance(weights, dict):
        missing = [m for m in members if m not in weights]
        if missing:
            raise ValueError(f"No weight for {missing}")
        w = np.array([weights[m] for m in members], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(members):
            raise ValueError(f"{len(w)} weights for {len(members)} members {members}")
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    w = w / w.sum()
    y, fold_id = res[members[0]]["y"], res[members[0]]["fold_id"]
    probas = []
    for m in members:
        r = res[m]
        if not (np.array_equal(r["y"], y) and np.array_equal(r["fold_id"], fold_id)):
            raise ValueError(f"{m}: different samples or outer folds; run all nested_cv_oof with the same X, y, cv, rs.")
        if "oof_proba" in r:
            proba = r["oof_proba"]
        elif m != "SVC":   # results from before oof_proba was stored: oof_score is the probability
            proba = r["oof_score"]
        else:
            raise ValueError("Re-run nested_cv_oof for SVC: its oof_score is decision_function, not a probability.")
        print(f"  {m}: pooled OOF AUC of predict_proba {roc_auc_score(y, proba):.3f}")
        probas.append(proba)
    oof_proba = np.average(probas, axis=0, weights=w)
    fold_auc = np.array([roc_auc_score(y[fold_id == k], oof_proba[fold_id == k]) for k in np.unique(fold_id)])
    print(f"Soft voting ({', '.join(f'{m} {wi:.2f}' for m, wi in zip(members, w))}): fold AUC {np.round(fold_auc, 3)}, mean {fold_auc.mean():.3f} "
          f"(std {fold_auc.std():.3f}), pooled OOF AUC {roc_auc_score(y, oof_proba):.3f}")
    return {"y": y, "oof_score": oof_proba, "oof_proba": oof_proba, "oof_pred": (oof_proba > 0.5).astype(int),
            "fold_id": fold_id, "fold_auc": fold_auc, "best_params": dict(zip(members, w))}


def bootstrap_fold_auc_ci(y, oof_score, fold_id, n_boot=2000, alpha=0.05, rs=42):
    """Percentile bootstrap CI for the mean of the per-fold AUCs.

    Each iteration resamples the test samples of every fold with replacement, separately for positives
    and negatives (so every fold keeps both classes), recomputes each fold AUC and averages them.
    The fitted fold models are held fixed, so the CI reflects test-sample variability only.

    Returns:
        (mean_auc, ci_low, ci_high)
    """
    rng = np.random.default_rng(rs)
    y = np.asarray(y).astype(int)
    oof_score = np.asarray(oof_score)
    folds = np.unique(fold_id)
    idx = {(k, c): np.where((fold_id == k) & (y == c))[0] for k in folds for c in (0, 1)}
    mean_auc = np.mean([_fast_auc(y[fold_id == k], oof_score[fold_id == k]) for k in folds])

    boot = np.empty(n_boot)
    for b in range(n_boot):
        aucs = []
        for k in folds:
            i = np.concatenate([rng.choice(idx[(k, c)], size=len(idx[(k, c)]), replace=True) for c in (0, 1)])
            aucs.append(_fast_auc(y[i], oof_score[i]))
        boot[b] = np.mean(aucs)
    ci_low, ci_high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean_auc, ci_low, ci_high


METRIC_NAMES = ["AUC", "SEN", "SPE", "ACC", "PRE", "F1"]


def _binary_metrics(y, score, pred):
    """AUC, sensitivity, specificity, accuracy, precision, F1 (0 when the denominator is 0)."""
    tp = np.sum((pred == 1) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    return np.array([
        _fast_auc(y, score),
        tp / (tp + fn) if tp + fn else 0.0,
        tn / (tn + fp) if tn + fp else 0.0,
        (tp + tn) / len(y),
        tp / (tp + fp) if tp + fp else 0.0,
        2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
    ])


def bootstrap_fold_metrics_ci(y, oof_score, oof_pred, fold_id, n_boot=2000, alpha=0.05, rs=42):
    """Mean +- SD over folds and percentile bootstrap CI of the fold mean for METRIC_NAMES.

    Same resampling scheme (and random draws) as bootstrap_fold_auc_ci: test samples of every fold are
    resampled with replacement within each class; the fold metrics are recomputed and averaged.

    Returns:
        df: index METRIC_NAMES, columns mean, std, ci_low, ci_high.
    """
    rng = np.random.default_rng(rs)
    y = np.asarray(y).astype(int)
    oof_score = np.asarray(oof_score)
    oof_pred = np.asarray(oof_pred).astype(int)
    folds = np.unique(fold_id)
    idx = {(k, c): np.where((fold_id == k) & (y == c))[0] for k in folds for c in (0, 1)}
    fold_vals = np.array([_binary_metrics(y[fold_id == k], oof_score[fold_id == k], oof_pred[fold_id == k])
                          for k in folds])

    boot = np.empty((n_boot, len(METRIC_NAMES)))
    for b in range(n_boot):
        vals = []
        for k in folds:
            i = np.concatenate([rng.choice(idx[(k, c)], size=len(idx[(k, c)]), replace=True) for c in (0, 1)])
            vals.append(_binary_metrics(y[i], oof_score[i], oof_pred[i]))
        boot[b] = np.mean(vals, axis=0)
    ci_low, ci_high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return pd.DataFrame({"mean": fold_vals.mean(axis=0), "std": fold_vals.std(axis=0),
                         "ci_low": ci_low, "ci_high": ci_high}, index=METRIC_NAMES)


def _with_soft_voting(res, soft_voting):
    """Return res plus a "SoftVoting" entry (see summarize_nested_cv for the meaning of soft_voting)."""
    if not soft_voting:
        return res
    if soft_voting is True:
        ens = soft_voting_oof(res)
    elif isinstance(soft_voting, dict):
        ens = soft_voting_oof(res, weights=soft_voting)
    else:
        ens = soft_voting_oof(res, members=soft_voting)
    return {**res, "SoftVoting": ens}


def _oof_pred(m, r):
    """OOF predicted labels; for results from before oof_pred was stored, the same rule as predict()."""
    if r.get("oof_pred") is not None:
        return np.asarray(r["oof_pred"]).astype(int)
    return (r["oof_score"] > (0 if m == "SVC" else 0.5)).astype(int)


def summarize_nested_cv(res, n_boot=2000, alpha=0.05, rs=42, digits=3, soft_voting=None):
    """Summary table of nested_cv_oof results for several models.

    Args:
        res (dict): {model_name: output of nested_cv_oof}.
        soft_voting (bool, list or dict, optional): True adds a "SoftVoting" row averaging all models in res
            with equal weights; a list averages only those models; a dict {model_name: weight} gives a weighted
            average of those models. None (default) adds no ensemble row.

    Returns:
        df: one row per model. AUC (mean +- SD over folds), AUC 95% CI, pooled OOF AUC, and
            SEN / SPE / ACC / PRE / F1 as fold mean (95% CI). Thresholded metrics use each model's own
            predict() (probability 0.5; decision_function 0 for SVC).
    """
    f = lambda v: f"{v:.{digits}f}"
    res = _with_soft_voting(res, soft_voting)
    rows = []
    for m, r in res.items():
        st = bootstrap_fold_metrics_ci(r["y"], r["oof_score"], _oof_pred(m, r), r["fold_id"],
                                       n_boot=n_boot, alpha=alpha, rs=rs)
        row = {"Model": m,
               "AUC (mean ± SD)": f"{f(st.at['AUC', 'mean'])} ± {f(st.at['AUC', 'std'])}",
               "AUC 95% CI": f"{f(st.at['AUC', 'ci_low'])}-{f(st.at['AUC', 'ci_high'])}",
               "Pooled OOF AUC": f(roc_auc_score(r["y"], r["oof_score"]))}
        for name in METRIC_NAMES[1:]:
            row[f"{name} (95% CI)"] = f"{f(st.at[name, 'mean'])} ({f(st.at[name, 'ci_low'])}-{f(st.at[name, 'ci_high'])})"
        rows.append(row)
    return pd.DataFrame(rows)


def oof_confusion_matrices(res, save_path="oof_confusion_matrices.csv", soft_voting=None, per_fold=False):
    """Confusion matrix of the OOF predictions of every model (labels 0 = negative, 1 = positive).

    Args:
        res (dict): {model_name: output of nested_cv_oof}.
        save_path (str, optional): CSV path. None skips saving.
        soft_voting (bool, list or dict, optional): add a "SoftVoting" row, same meaning as in summarize_nested_cv.
        per_fold (bool): also add one row per outer fold (the pooled row is the sum of the fold rows).

    Returns:
        df: Model, Fold ("pooled" or fold index), TN, FP, FN, TP, N.
    """
    res = _with_soft_voting(res, soft_voting)
    rows = []
    for m, r in res.items():
        y, pred, fold_id = np.asarray(r["y"]).astype(int), _oof_pred(m, r), np.asarray(r["fold_id"])
        parts = [("pooled", np.ones(len(y), dtype=bool))]
        if per_fold:
            parts += [(int(k), fold_id == k) for k in np.unique(fold_id)]
        for fold, mask in parts:
            tn, fp, fn, tp = confusion_matrix(y[mask], pred[mask], labels=[0, 1]).ravel()
            rows.append({"Model": m, "Fold": fold, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "N": int(mask.sum())})
    df = pd.DataFrame(rows)
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")
    return df


def _delong_auc_cov(y, preds):
    """Fast DeLong (Sun & Xu, 2014). preds: (k, n) scores of k models. Returns AUCs (k,) and covariance (k, k)."""
    pos = preds[:, y == 1]
    neg = preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([rankdata(p) for p in pos])
    ty = np.array([rankdata(q) for q in neg])
    tz = np.array([rankdata(np.concatenate([p, q])) for p, q in zip(pos, neg)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n          # structural components of positives
    v10 = 1 - (tz[:, m:] - ty) / m      # structural components of negatives
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov

def _holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def delong_vs_reference(y, proba_dict, reference="RFR", alternative="two-sided", alpha=0.05):
    """Paired DeLong tests of one reference model against every other model, Holm-adjusted over these comparisons.

    Args:
        y (array-like): binary labels (0/1).
        proba_dict (dict): {model_name: OOF score array}, all on the same samples in the same order.
        reference (str): key of the reference model.
        alternative (str): "two-sided" (AUC_ref != AUC_model) or "greater" (AUC_ref > AUC_model, one-sided).
        alpha (float): for the two-sided (1 - alpha) CI of the AUC difference.

    Returns:
        df: reference, model, auc_ref, auc_model, auc_diff (ref - model), diff_ci_low, diff_ci_high, z, p, p_holm.
    """
    if alternative not in ("two-sided", "greater"):
        raise ValueError("alternative must be 'two-sided' or 'greater'")
    y = np.asarray(y).astype(int)
    z_crit = norm.ppf(1 - alpha / 2)
    rows = []
    for m, score in proba_dict.items():
        if m == reference:
            continue
        aucs, cov = _delong_auc_cov(y, np.vstack([proba_dict[reference], score]))
        diff = aucs[0] - aucs[1]
        se = np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0))
        z = diff / se if se > 0 else 0.0   # identical rankings -> z = 0
        p = 2 * norm.sf(abs(z)) if alternative == "two-sided" else norm.sf(z)
        rows.append({"reference": reference, "model": m, "auc_ref": aucs[0], "auc_model": aucs[1],
                     "auc_diff": diff, "diff_ci_low": diff - z_crit * se, "diff_ci_high": diff + z_crit * se,
                     "z": z, "p": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = _holm(df["p"])
    return df


def plot_shap_summary(model, X, rename_list=None, save=False):
    """Plot the SHAP summary plots.

    Args:
        model (_type_): estimator.
        X (_type_): X with features
        rename_list (_type_, optional): If want to change the y axis name. Defaults to None.
        save (bool, optional): save. Defaults to False.
        Can change ticks and fnames.
    """
    if hasattr(model, "best_estimator_"):  # GridSearchCV -> the refitted best model
        model = model.best_estimator_
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):      # older shap: one array per class
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:      # newer shap: (samples, features, classes)
        shap_values = shap_values[:,:,1]
    if rename_list is not None:
        X = X.set_axis(rename_list, axis=1)  # rename for plotting only; the caller's X is unchanged
    
    # shap.summary_plot(shap_values, X, plot_type=plot_type) # Summary bar plot
    shap.summary_plot(shap_values, X, show=False, plot_size=(6,6)) # Summary bee-swarm plot
    # shap.summary_plot(shap_values, X, plot_type='layered_violin') # Summary bee-swarm plot
    # shap.dependence_plot(3, shap_values, X)
    # shap.force_plot(explainer.expected_value, shap_values, X)
    # shap.plots.waterfall(explainer)
    # plt.xticks([-50,-25,0,25,50])
    if save:
        os.makedirs("SHAP_summary", exist_ok=True)
        plt.savefig(f'SHAP_summary/{type(model).__name__}_{X.shape[1]}features.pdf', bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_shap_importance(model, X, topk=25, rename_list=None, save=False):
    """Plot the SHAP bar plots.

    Args:
        model (_type_): estimator.
        X (_type_): X with features
        rename_list (_type_, optional): If want to change the y axis name. Defaults to None.
        save (bool, optional): save. Defaults to False.
        Can change ticks and fnames.
    """
    if hasattr(model, "best_estimator_"):  # GridSearchCV -> the refitted best model
        model = model.best_estimator_
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):      # older shap: one array per class
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:      # newer shap: (samples, features, classes)
        shap_values = shap_values[:,:,1]
    if rename_list is not None:
        X = X.set_axis(rename_list, axis=1)  # rename for plotting only; the caller's X is unchanged

    if not save:
        shap.summary_plot(shap_values, X, plot_type='bar', plot_size=(10,8), max_display=topk) # Summary bar plot
    # shap.plots.bar(shap_values, max_display=15)
    # shap.summary_plot(shap_values, X, show=False, plot_size=(4,4)) # Summary bee-swarm plot
    # shap.summary_plot(shap_values, X, plot_type='layered_violin') # Summary bee-swarm plot
    # shap.dependence_plot(3, shap_values, X)
    # shap.force_plot(explainer.expected_value, shap_values, X)
    # shap.plots.waterfall(explainer)
    # plt.xticks([-50,-25,0,25,50])
    if save:
        shap.summary_plot(shap_values, X, plot_type='bar', plot_size=(10,8), max_display=topk, show=False)
        os.makedirs("SHAP_importance", exist_ok=True)
        plt.savefig(f'SHAP_importance/{type(model).__name__}_{X.shape[1]}features.pdf', bbox_inches="tight")
        plt.close()
