# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

Single-notebook Zindi competition entry: **Togo Fiber Optics Uptake Prediction Challenge** (FTTH adoption prediction across Togo communes). Sole artifact: `solution.ipynb`. Final scores: Public 0.923651, Private 0.926604 (ROC AUC). Comments and markdown in the notebook are written in French — keep that language when editing existing cells.

Competition link: https://zindi.africa/competitions/togo-fiber-optics-uptake-prediction-challenge

## Running the Notebook

The notebook was authored on Kaggle and hardcodes Kaggle paths. When running locally, override the path cell:

```python
data_input = '/kaggle/input/zd-tg-ds'   # change to local dataset dir
data_output = '/kaggle/working/'        # change to local output dir
```

Expected input files in `data_input`: `Train.csv`, `train_2.csv`, `Test.csv` (UTF-8 with BOM, `encoding='utf-8-sig'`).

Dependencies (not pinned in repo — install manually): `optuna`, `xgboost`, `lightgbm`, `catboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`. The notebook contains commented `!pip install optuna` and `!pip install -r requirements.txt` lines; no `requirements.txt` is checked in.

Full cross-validation run of the stacking model takes ~30 min on Kaggle's 4 CPUs (`WHAT_TO_DO = 4` in the evaluation cell). The MOSAIKS XGBoost fit alone takes ~4 min wall time.

## Architecture

The solution is one linear pipeline executed top-to-bottom in `solution.ipynb`. Sections are numbered in markdown headers.

### Data shape and the two feature blocks

Each row = one Togolese commune segment. The training frame is ~4043 columns wide and splits into two semantically distinct blocks that the code treats very differently:

1. **MOSAIKS features** — columns at index `42:4042` (4000 float features from the MOSAIKS satellite-imagery API, named `MOSAIK_FEATURE` and `.0` … `.3999`). These are NEVER fed directly to the final model.
2. **Sociodemographic features** — the leftmost ~42 columns (RGPH census variables: `TypeLogmt_*`, `H08_Impute`, `H09_Impute`, `H17*`, `H18*`, `H20*`, `H21*`, `TAILLE_MENAGE`, `Connexion`, etc.) plus `ID`, `BoxLabel`, `Target`.

### The MOSAIKS dimensionality-reduction trick (critical to understand)

The 4000 MOSAIKS columns are not used as features in the final stack. Instead, an XGBoost classifier (`mosaik_model1`) is fit on those 4000 columns alone to predict `Target`, and its `predict_proba` output is appended as a single new column `MOSAIK_LABEL` to both train and test. The raw 4000 columns and the `' .*'`-prefixed columns are then dropped in `feature_engineering()`. This reduces the input to ~36 columns before the stacking model sees it. Any change to the feature pipeline must preserve this two-stage structure.

### Encoding cleanup

Three categorical columns (`TypeLogmt_3`, `H08_Impute`, `H09_Impute`) arrive with mojibake from inconsistent Latin/UTF encoding — the same value appears as both `Logement � un niveau` and `Logement ? un niveau`. The `tl3_map` / `h8_map` / `h9_map` dicts collapse both variants to one ASCII-folded label. Applied to **both** train and test. Skipping this step roughly doubles the cardinality of those columns and degrades the stack.

### Other preprocessing decisions encoded in code

- Train data is concatenated from `Train.csv` + `train_2.csv`, then deduped on `ID` (46733 → 30558 rows).
- `feature_engineering()` drops `ID`, `BoxLabel`, `UNKNOWN`, and an explicit list of low-signal H-columns (`H20Y, H17I, H17D, H21A, H17H, H17G`), plus everything starting with `' .'`.
- `Connexion` has ~34% missing values; the preprocessor handles this via `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` followed by `SimpleImputer(add_indicator=True)`. Do not drop the `add_indicator=True` — it adds a missingness flag the tree models use.
- Numeric features get `RobustScaler` (no imputation — trees tolerate NaN, KNN/SVC paths rely on the scaler).
- `Target` is renamed from `Accès internet`; `MOSAIK_FEATURE` is renamed from the blank-named first MOSAIKS column.

### Stacking model

`model_z` is a `StackingClassifier` (5-fold `StratifiedKFold`, `random_state=42`) with 8 base estimators wrapped in `make_pipeline(preprocessor, ...)`:

`RandomForestClassifier`, `ExtraTreesClassifier`, `CatBoostClassifier`, `XGBClassifier`, `LGBMClassifier`, `AdaBoostClassifier`, `KNeighborsClassifier`, `SVC`.

Final estimator: `CatBoostClassifier` (hyperparameters tuned via Optuna). The per-model hyperparameters in the notebook are the Optuna-tuned values — comments above each model record its standalone CV score.

### The `WHAT_TO_DO` dispatcher

Cell `35d57291` is a switch with 6 modes:

- `1` — single-model evaluation (`rfc_model`)
- `2` — run Optuna HPO over the final estimator's hyperparameters
- `3` — `try_n_estimators()` / `try_max_depth()` (defined elsewhere in the notebook)
- `4` — full 5-fold CV of the stack (the canonical "is it stable?" check)
- `5` — evaluate all 8 base models + the stack
- `6` — no-op

Always set `WHAT_TO_DO` before running the evaluation cell; default in the committed notebook is `4`.

### Output

Predicted probabilities are rounded to 2 decimal places before submission (`np.round(test_preds, 2)`). The notebook author flagged this rounding as a deliberate score-boosting choice — comment it out to emit raw probabilities. Submission filename embeds `time.time()` so each run produces a fresh file.
