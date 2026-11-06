# Pancreatic Cancer Recurrence Prediction from Longitudinal Tumor Markers

Code for the paper:

> **Explainable machine learning approaches for predicting pancreatic cancer recurrence using multiple longitudinal tumor markers**
> Sheng Hu, Hiroki Imamura, Taiga Kakoki, Tasuku Kimura, Yoshito Tomimaru, Shogo Kobayashi, Hirofumi Akita, Hidetoshi Eguchi, Yasuko Matsubara, Yasushi Sakurai.
> *PLOS ONE* (under review).

We predict postoperative recurrence of pancreatic ductal adenocarcinoma (PDAC) from the postoperative time series of three serum tumor markers: CA19-9, DUPAN-2, and CEA. Each patient's marker history is truncated 90 days before the reference date and imputed. It is then summarized into 30 interpretable features: latest values, gradient-based features, and other statistical features. These features are evaluated with six classifiers and a soft-voting ensemble under patient-level nested cross-validation. SHAP is used to describe how the fitted model uses the features.

## Repository structure

```
.
├── train_eval.py      # Main script: nested CV, RFECV, summary tables, confusion matrices, DeLong tests
├── plot_SHAP.py       # SHAP importance / summary plots of the full-cohort RF refit (Fig 6)
├── train_eval_reg.py  # Exploratory regression of days until recurrence 
├── hu_utils.py        # Model grids
├── hu_utils_reg.py    # Regressor grids
├── plt_configs.py     # 
├── csv_order/         # Feature matrices (X) and labels (y) for classification
│   ├── 3_month_trimed_X_early_{time,nearest,MICE}.csv
│   └── 3_month_trimed_y_early_{time,nearest,MICE}.csv
└── csv_reg/           # Feature matrix and recurrence intervals for the exploratory regression analysis
    ├── 3_month_trimed_X_early_time_reg.csv
    └── 3_month_trimed_y_early_time_reg.csv
```

### Input data

Each `X` file contains one row per patient (n = 105; 75 recurrent, 30 non-recurrent) and 30 feature columns. The columns follow the pattern `{marker}_{feature}`, where `{marker}` is `CA19-9`, `DUPAN-2` or `CEA`:

| Category | Feature suffixes |
|---|---|
| Latest value | `latest` |
| Gradient-based | `grad_tail`, `grad_mean`, `grad_max`, `grad_head` |
| Other statistical | `max`, `std`, `mean`, `m3` (max of last 3 points), `m3_h` (max of first 3 points) |

Each `y` file holds the matching binary labels (1 = recurrence, 0 = no recurrence). The file suffix gives the imputation method used before feature extraction:

- `time`: linear interpolation over time
- `nearest`: nearest-value filling
- `MICE`: chained-equation iterative imputation

The regression files in `csv_reg/` contain 83 recurrent patients. In these files, each marker history is truncated after the largest increase between adjacent CA19-9 values and then imputed by time interpolation. `X` has the same 30 feature columns as the classification files. `y` holds the number of days from the largest CA19-9 elevation to radiologically confirmed recurrence, stored as strings such as `21 days`.

## Requirements

Tested with Python 3.8 and:

| Package | Version |
|---|---|
| numpy | 1.23.5 |
| pandas | 1.5.3 |
| scikit-learn | 1.1.3 |
| xgboost | 1.4.1 |
| shap | 0.41.0 |
| scipy | 1.9.3 |
| matplotlib | 3.7.3 |

```bash
pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.1.3 xgboost==1.4.1 shap==0.41.0 scipy==1.9.3 matplotlib==3.7.3
```

## Usage

Run the scripts from the repository root.

### Classification (`train_eval.py`)

```bash
python train_eval.py                                    # time interpolation, all 30 features
python train_eval.py --imputation nearest               # time | nearest | MICE
python train_eval.py --features latest                  # latest-value-only comparator
python train_eval.py --features latest gradient         # union of preset feature sets
python train_eval.py --features CA19-9_grad_mean CEA_grad_mean CEA_mean   # explicit columns
```

| Option | Description |
|---|---|
| `--imputation` | Which `csv_order` files to load: `time` (default), `nearest`, or `MICE`. |
| `--features` | Preset names (`latest`, `gradient`, `statistical`) and/or column names. Default: all columns. Presets are defined in `FEATURE_SETS` at the top of the script. |

The script runs the following steps:

1. **Nested CV on the given features.** The outer loop is a stratified 5-fold split (`random_state=42`). The inner loop is a 5-fold `GridSearchCV` (scoring = ROC AUC). The six classifiers are RF, XGB, ET, SVM, KNN, and LR; SVM, KNN, and LR use `RobustScaler`. The script prints the mean outer-fold AUC ± SD and a 95% bootstrap CI.
2. **RFECV feature selection.** Selection is repeated within each outer training set, and the per-fold selection frequencies are printed. Nested CV is then run with the per-fold feature subsets. A soft-voting ensemble (RF, ET, XGB, KNN) is added. The script prints the following summary table:
   - mean outer-fold AUC ± SD, its 95% CI, and pooled out-of-fold (OOF) AUC
   - SEN, SPE, ACC, PRE, and F1, each with a 95% CI
3. **Confusion matrices** of the pooled OOF predictions are saved to `oof_confusion_matrices_{imputation}_{features}.csv`.
4. **Paired DeLong tests** compare RF with every other model, including the ensemble, on pooled OOF scores. The script reports the 95% CI of the AUC difference and Holm-adjusted p-values.

### SHAP plots (`plot_SHAP.py`)

```bash
python plot_SHAP.py --imputation time
```
### Exploratory regression (`train_eval_reg.py`)

```bash
python train_eval_reg.py
```

## Reproducing the paper

| Paper item | Command |
|---|---|
| Table 6 (imputation comparison, all 30 features) | `python train_eval.py --imputation {time,nearest,MICE}` (step 1 output) |
| Table 7 (single feature categories) | `python train_eval.py --features {latest,gradient,statistical}` (step 1 output) |
| Table 8 (pairwise category combinations) | `python train_eval.py --features latest gradient`, etc. (step 1 output) |
| Table 9, Table 10, Fig 5, Table 11 | `python train_eval.py` (steps 2–4) |
| S3 Table (RFECV selection frequencies) | `python train_eval.py` (step 2 printout / `table_s4`) |
| Fig 6 | `python plot_SHAP.py` |
| Table 13, Fig 8 (recurrence-interval distribution) | `python train_eval_reg.py` (step 1) |
| Table 14, Fig 9 (regression performance) | `python train_eval_reg.py` (steps 2–3) |

## Data availability

The feature files in `csv_order/` and `csv_reg/` are derived from anonymized clinical records of The University of Osaka Hospital. The study was approved by the hospital's ethics committee (approval number 23455-3). 

## License
MIT License
