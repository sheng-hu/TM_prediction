"""SHAP importance (Fig 6a) and summary (Fig 6b) plots of RF refitted on the full cohort.

Usage:
    python plot_SHAP.py                       # time interpolation
    python plot_SHAP.py --imputation nearest

Outputs: SHAP_importance/RandomForestClassifier_{n}features.pdf, SHAP_summary/RandomForestClassifier_{n}features.pdf
"""
import argparse
import importlib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import hu_utils
hu_utils = importlib.reload(hu_utils)
import plt_configs  # noqa: F401  (matplotlib rcParams)

parser = argparse.ArgumentParser(description="SHAP plots of the full-cohort RF refit.")
parser.add_argument("--imputation", choices=["time", "nearest", "MICE"], default="time",
                    help="imputation method of the input CSV (default: time)")
args = parser.parse_args()

X = pd.read_csv(f'csv_order/3_month_trimed_X_early_{args.imputation}.csv')
y = pd.read_csv(f'csv_order/3_month_trimed_y_early_{args.imputation}.csv').to_numpy().reshape(-1)

rfmodel = RandomForestClassifier(n_estimators=200,
                                    bootstrap=True,
                                    max_depth=None,
                                    max_features=1.0,
                                    min_samples_leaf=1,
                                    class_weight="balanced", random_state=42, n_jobs=1)
fold_features, table_s4, full_features = hu_utils.rfecv_feature_stability(X, y, estimator=rfmodel,
                                                                          cv=5, rs=42, freq_threshold=0.4)

# Descriptive full-cohort refit 
final_rf = hu_utils.setup_gridsearch_model_clf("RFR", class_weight="balanced").fit(X[full_features], y)
print(f"final_rf best params: {final_rf.best_params_}")

hu_utils.plot_shap_importance(final_rf, X[full_features], topk=len(full_features), save=True)
hu_utils.plot_shap_summary(final_rf, X[full_features], save=True)
