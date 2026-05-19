# Togo Fiber Optics Uptake Prediction Challenge

Solution for the Zindi [Togo Fiber Optics Uptake Prediction Challenge](https://zindi.africa/competitions/togo-fiber-optics-uptake-prediction-challenge): predicting fiber-to-the-home (FTTH) internet adoption across commune-level segments in Togo using sociodemographic data (RGPH / INSEED) and geographic data (satellite imagery via the MOSAIKS API).

## Scores

| Metric            | Score        |
| ----------------- | ------------ |
| Public (ROC AUC)  | 0.923651326  |
| Private (ROC AUC) | 0.926604040  |

## Repository contents

```
.
├── solution.ipynb   # Single notebook containing the full solution
├── CLAUDE.md        # Guide for Claude Code
└── README.md
```

## Data

Data comes from three sources:

- **Togocom** and **GVA** — FTTH operators (`Connexion` field).
- **INSEED** — General Census of Population and Housing (RGPH).
- **MOSAIKS API** — 4000 features derived from satellite imagery.

Expected files in the input directory (`data_input`):

- `Train.csv`
- `train_2.csv`
- `Test.csv`

Encoding: `utf-8-sig`. Base administrative unit is the commune.

## Installation

```bash
pip install optuna xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn
```

Python 3.10+ recommended.

## Running

1. Open `solution.ipynb` (Jupyter / Kaggle / Colab).
2. Edit the path configuration cell:

   ```python
   data_input  = '/kaggle/input/zd-tg-ds'   # → local path
   data_output = '/kaggle/working/'         # → output directory
   ```

3. Set `WHAT_TO_DO` in the evaluation cell:

   | Value | Action                                                  |
   | ----- | ------------------------------------------------------- |
   | 1     | Evaluate a single model (RandomForest)                  |
   | 2     | Optuna hyperparameter search on the meta-estimator      |
   | 3     | Experiment with `n_estimators` / `max_depth`            |
   | 4     | 5-fold cross-validation of the full stack (default)     |
   | 5     | Evaluate all base models + stack                        |
   | 6     | No-op                                                   |

4. Run the notebook top to bottom.

Approximate runtime on Kaggle (4 vCPU): ~30 min for full CV, ~4 min for the MOSAIKS XGBoost fit.

## Approach

Two-stage pipeline:

1. **MOSAIKS reduction** — an `XGBClassifier` is trained on the 4000 MOSAIKS features to produce a single probability `MOSAIK_LABEL`, appended to the sociodemographic features. The 4000 raw columns are then dropped.
2. **Stacking** — `StackingClassifier` (stratified 5-fold CV) over the resulting ~36 features:

   - Base models: RandomForest, ExtraTrees, CatBoost, XGBoost, LightGBM, AdaBoost, KNeighbors, SVC.
   - Meta-estimator: CatBoost (hyperparameters tuned via Optuna).

Preprocessing: mojibake cleanup on `TypeLogmt_3`, `H08_Impute`, `H09_Impute`; `OrdinalEncoder` + `SimpleImputer(add_indicator=True)` for categoricals; `RobustScaler` for numerics.

## Output

CSV file `TestSubmission<timestamp>.csv` with columns `ID`, `Target` (probabilities rounded to 2 decimals).

## References

- Rolf et al., *A generalizable and accessible approach to machine learning with global satellite imagery*, Nature Communications (2021).
- Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019.
- [Stacking Classifiers — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html).
