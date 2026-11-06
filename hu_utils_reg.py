"""Utility functions for the exploratory regression of days until recurrence (used by train_eval_reg.py)."""

import os
import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


class BestEstimatorCV:
    """For containing the best_estimator and related stats, plots."""

    def __init__(self, cv, X, y, estimator, random_state=1126):
        """Prepare train, test data.

        Args:
            cv (int): cv folds
            X (df): features
            y (array-like): days until recurrence
            estimator (Regressor): best_estimator_
            random_state (int, optional): seed of the shuffled KFold. Defaults to 1126.
        """
        self.cv = cv
        self.X = X
        self.y = y
        self.estimator = estimator
        self.random_state = random_state

    def output_stats(self):
        """RMSE / MAE / R2 of a shuffled KFold CV (train and test), printed and returned.

        Returns:
            (dict, float): fold-mean stats (trn_/tes_ mean, std, r2, mae), test RMSE mean.
        """
        cvf = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        model = self.estimator
        err_trn = []
        err_tes = []
        r_2_tes = []
        r_2_trn = []
        mae_trn = []
        mae_tes = []
        X = self.X
        y = np.asarray(self.y)
        for train_index, test_index in cvf.split(X):
            x_trn, x_tes = X.iloc[train_index], X.iloc[test_index]
            y_trn, y_tes = y[train_index], y[test_index]
            model.fit(x_trn, y_trn)
            x_trn_pred = model.predict(x_trn)
            x_tes_pred = model.predict(x_tes)

            err_tes.append(mean_squared_error(x_tes_pred, y_tes))
            err_trn.append(mean_squared_error(x_trn_pred, y_trn))
            mae_tes.append(mean_absolute_error(x_tes_pred, y_tes))
            mae_trn.append(mean_absolute_error(x_trn_pred, y_trn))
            r_2_tes.append(r2_score(y_tes, x_tes_pred))
            r_2_trn.append(r2_score(y_trn, x_trn_pred))
        v_tes = np.sqrt(np.array(err_tes))  # 5 fold
        v_trn = np.sqrt(np.array(err_trn))
        print(
            "RMSE %1.3f (sd: %1.3f, min:%1.3f, max:%1.3f, det:%1.3f) ... train"
            % (v_trn.mean(), v_trn.std(), v_trn.min(), v_trn.max(), np.array(r_2_trn).mean())
        )
        print(
            "RMSE %1.3f (sd: %1.3f, min:%1.3f, max:%1.3f, det:%1.3f) ... test"
            % (v_tes.mean(), v_tes.std(), v_tes.min(), v_tes.max(), np.array(r_2_tes).mean())
        )
        ret = {}
        ret["trn_mean"] = v_trn.mean()  # 5 fold mean. It's different from the overal rmse.
        ret["trn_std"] = v_trn.std()
        ret["trn_r2"] = np.array(r_2_trn).mean()
        ret["tes_mean"] = v_tes.mean()
        ret["tes_std"] = v_tes.std()
        ret["tes_r2"] = np.array(r_2_tes).mean()
        ret["tes_mae"] = np.array(mae_tes).mean()
        ret["trn_mae"] = np.array(mae_trn).mean()
        return ret, v_tes.mean()

    def plot_hold_out_oof(self, fname, lim_range=[0, 150], save=True):
        """Observed vs. predicted days for the same KFold splits as output_stats (Fig 9).

        Train points (blue) and held-out test points (red) of every fold; saved to result_plot/{fname}.pdf.
        """
        feat = self.X
        target = np.asarray(self.y)
        model = self.estimator

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_aspect("equal")

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, test_idx) in enumerate(kf.split(feat)):
            x_train, x_test = feat.iloc[train_idx], feat.iloc[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]

            fold_model = clone(model)
            fold_model.fit(x_train, y_train)

            y_train_pred = fold_model.predict(x_train)
            y_test_pred = fold_model.predict(x_test)

            ax.plot(y_train, y_train_pred, "bo", alpha=0.2,
                    label="train" if fold == 0 else None)
            ax.plot(y_test, y_test_pred, "ro", alpha=0.7,
                    label="test" if fold == 0 else None)

        ax.plot(lim_range, lim_range, "--")
        ax.set_xlim(lim_range)
        ax.set_ylim(lim_range)
        ax.set_xlabel("Days Until Recurrence, Truth")
        ax.set_ylabel("Days Until Recurrence, Predicted")
        if save:
            os.makedirs("result_plot", exist_ok=True)
            plt.savefig(f"result_plot/{fname}.pdf", bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


def setup_gridsearch_model(model, rs=1126):
    """Set up GridSearchCV (5-fold, scoring = negative RMSE) and parameter grids for a regressor.

    Args:
        model (str): "RFR" or "XGB".
        rs (int, optional): random state. Defaults to 1126.

    Returns:
        GridSearchCV: object.
    """
    cv_object = None
    if model == "RFR":
        cv_object = GridSearchCV(
            RandomForestRegressor(n_jobs=-1, n_estimators=200, random_state=rs),
            param_grid={
                "max_features": ["sqrt", 1.0],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_leaf_nodes": ["None", 70],
            },
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
    elif model == "XGB":
        cv_object = GridSearchCV(
            # n_jobs=1: xgboost threads inside GridSearchCV's worker processes can segfault (macOS, xgboost 1.4).
            XGBRegressor(n_jobs=1, importance_type="total_gain", random_state=rs),
            param_grid={
                "n_estimators": [200, 500, 1000],
                "max_depth": [6, 7, 8],
                "learning_rate": [0.1, 0.05],
                "subsample": [0.8, 0.9, 1],
                "colsample_bytree": [0.8, 0.9, 1],
            },
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
    else:
        warnings.warn("Machine Learning model cannot defined")
    return cv_object
