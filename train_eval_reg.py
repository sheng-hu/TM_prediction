"""Exploratory regression of days until recurrence after the largest CA19-9 elevation (Tables 13-14, Figs 8-9).

Usage:
    python train_eval_reg.py

Outputs: result_plot/recurrence_interval_hist.pdf, result_plot/hout_out_{rf,xgb}_oof.pdf
"""
import importlib
import os

import pandas as pd
from matplotlib import pyplot as plt

import hu_utils_reg
hu_utils_reg = importlib.reload(hu_utils_reg)

X = pd.read_csv('csv_reg/3_month_trimed_X_early_time_reg.csv')
y_all = pd.read_csv('csv_reg/3_month_trimed_y_early_time_reg.csv')

# "21 days" -> 21
y_days = [y_e.days for y_e in pd.to_timedelta(y_all.to_numpy().reshape(-1))]
y_days_df = pd.DataFrame(y_days)
print(y_days_df.describe().T)  # Table 13

# Fig 8: distribution of recurrence intervals
y_days_df.plot(kind='hist', bins=500, logy=True, xlim=(0, 150), xticks=range(0, 140, 10), legend=False)
os.makedirs("result_plot", exist_ok=True)
plt.savefig("result_plot/recurrence_interval_hist.pdf", bbox_inches="tight")
plt.close()

# Table 14 / Fig 9
for name, fname in [("RFR", "hout_out_rf_oof"), ("XGB", "hout_out_xgb_oof")]:
    gs = hu_utils_reg.setup_gridsearch_model(name)
    gs.fit(X, y_days)
    best_cv = hu_utils_reg.BestEstimatorCV(estimator=gs.best_estimator_, X=X, y=y_days, cv=5)
    print(name, gs.best_params_)
    best_cv.output_stats()
    best_cv.plot_hold_out_oof(fname, lim_range=[0, 300], save=True)
