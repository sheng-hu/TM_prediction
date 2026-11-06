"""Usage:
    python train_eval.py                                   # time interpolation, all 30 features
    python train_eval.py --imputation nearest
    python train_eval.py --imputation MICE --features gradient
    python train_eval.py --features latest gradient        # union of preset feature sets
    python train_eval.py --features CA19-9_grad_mean CEA_grad_mean CEA_mean
"""
import argparse
import importlib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import hu_utils
hu_utils = importlib.reload(hu_utils)

MARKERS = ["CA19-9", "DUPAN-2", "CEA"]
# Preset feature lists (edit or add your own). Column names follow csv_order/*_X_*.csv.
FEATURE_SETS = {
    "latest": [f"{mk}_latest" for mk in MARKERS],
    "gradient": [f"{mk}_{s}" for s in ["grad_tail", "grad_mean", "grad_max", "grad_head"] for mk in MARKERS],
    "statistical": [f"{mk}_{s}" for s in ["max", "std", "mean", "m3", "m3_h"] for mk in MARKERS],
}

parser = argparse.ArgumentParser(description="Nested-CV evaluation of recurrence classifiers.")
parser.add_argument("--imputation", choices=["time", "nearest", "MICE"], default="time",
                    help="imputation method of the input CSV (default: time)")
parser.add_argument("--features", nargs="+", default=None,
                    help=f"preset names {list(FEATURE_SETS)} and/or column names; default: all columns")
args = parser.parse_args()

seed = 42

X_all = pd.read_csv(f'csv_order/3_month_trimed_X_early_{args.imputation}.csv')
y_all = pd.read_csv(f'csv_order/3_month_trimed_y_early_{args.imputation}.csv')

if args.features is None:
    features = list(X_all.columns)
else:
    features = []
    for name in args.features:
        for col in FEATURE_SETS.get(name, [name]):
            if col not in features:
                features.append(col)
    unknown = [c for c in features if c not in X_all.columns]
    if unknown:
        parser.error(f"unknown features {unknown}; available: {list(X_all.columns)}")
tag = f"{args.imputation}_{'all' if args.features is None else '+'.join(args.features)}"
print(f"Imputation: {args.imputation}, {len(features)} features: {features}")

X = X_all[features]
y = pd.DataFrame(y_all).to_numpy().reshape(-1)

MODELS = ["RFR", "XGB", "ETR", "SVC", "KNN", "LOGISTIC"]


# 1. Give the results for imputation results.
res, rows = {}, []
for m in MODELS:
    # res[m] = hu_utils.nested_cv_oof(m, X, y, class_weight="balanced", fold_features=fold_features)
    res[m] = hu_utils.nested_cv_oof(m, X, y, class_weight="balanced")
    auc, lo, hi = hu_utils.bootstrap_fold_auc_ci(res[m]["y"], res[m]["oof_score"], res[m]["fold_id"], n_boot=2000)
    rows.append({"model": m, "auc_mean": auc, "ci_low": lo, "ci_high": hi, "auc_std": res[m]["fold_auc"].std()})
auc_table_imputations_30_features = pd.DataFrame(rows)

print(auc_table_imputations_30_features)
# 2. Give the results for RFECV and detailed classify performance.
rfmodel = RandomForestClassifier(n_estimators=200,
                                    bootstrap=True,
                                    max_depth=None,
                                    max_features=1.0,
                                    min_samples_leaf=1,
                                    class_weight="balanced", random_state=42, n_jobs=1)

# # 2. RFECV

fold_features, table_s4, full_features = hu_utils.rfecv_feature_stability(X, y,estimator=rfmodel,
                                                                          cv=5, rs=42,freq_threshold=0.4)
res = {m: hu_utils.nested_cv_oof(m, X, y, class_weight="balanced", fold_features=fold_features)
       for m in MODELS}
voting = ["RFR", "ETR", "XGB", "KNN"]
auc_table_RFECV = hu_utils.summarize_nested_cv(res, soft_voting=voting)
print(auc_table_RFECV)
cm_table = hu_utils.oof_confusion_matrices(res, save_path=f"oof_confusion_matrices_{tag}.csv", soft_voting=voting)
print(cm_table)

# 3. DeLong: RFR vs. every other model (including the soft-voting ensemble).
res_all = hu_utils._with_soft_voting(res, voting)
scores = {m: r["oof_score"] for m, r in res_all.items()}
delong_table = hu_utils.delong_vs_reference(res["RFR"]["y"], scores, reference="RFR")
print(delong_table)
