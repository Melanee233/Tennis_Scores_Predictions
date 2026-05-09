# Tennis Scores Predictions

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Project Overview

This project predicts the outcome of professional women's tennis matches using historical WTA match data. It is built as a Kedro project and combines a reproducible data pipeline, feature engineering tailored to pre-match information, and two modeling approaches:

- a baseline `RandomForestClassifier`
- an AutoML workflow based on `AutoGluon Tabular`

The repository covers the full workflow from raw data ingestion to cleaned datasets, engineered features, trained models, and evaluation metrics.

## Main Goals

- build a reproducible end-to-end machine learning workflow for tennis match prediction
- transform winner/loser match records into a symmetric player-versus-player training format
- engineer informative pre-match features, including Elo-based strength estimates
- compare a manually configured Random Forest model with an AutoML approach
- optionally track experiments and artifacts with Weights & Biases

## Tech Stack

- Python 3.10+
- Kedro 1.2
- pandas, NumPy, scikit-learn
- AutoGluon Tabular
- Weights & Biases
- Jupyter notebooks for EDA and feature-design experiments

## Dataset

The project expects yearly CSV files with historical WTA match results. The helper script in [scripts/ingest_data.py](/scripts/ingest_data.py) downloads files for seasons `1968` through `2024`, concatenates them, and saves the combined dataset to:

`data/01_raw/combined_matches.csv`

The data dictionary for the most important raw columns is available in [notebooks/data_dictionary.md](/notebooks/data_dictionary.md).

## Pipeline Overview

The project contains three Kedro pipelines that can be run independently or together.

### 1. Data Processing

Source: [src/tennis_scores_predictions/pipelines/data_processing](/src/tennis_scores_predictions/pipelines/data_processing)

This pipeline prepares model-ready data from the raw combined match history. The main steps are:

- keep only columns available before a match starts
- keep matches involving players active since 2020
- standardize missing values
- impute selected missing fields such as entry type and player height
- remove rows with missing ranking or demographic data required downstream
- convert winner/loser records into a symmetric `player_A` vs `player_B` representation
- add a tournament round index
- compute Elo ratings over time
- build difference features such as ranking, age, height, ranking-points, and Elo gaps
- one-hot encode categorical variables

Key outputs:

- `data/02_intermediate/matches_cleaned.csv`
- `data/03_primary/matches_pre_encoded.csv`
- `data/04_feature/matches_features.csv`

### 2. Model Training

Source: [src/tennis_scores_predictions/pipelines/model_training](/src/tennis_scores_predictions/pipelines/model_training)

This pipeline trains a baseline `RandomForestClassifier` on the engineered feature table. It:

- splits features and target
- trains the model using parameters from `conf/base/parameters.yml`
- evaluates the model with `accuracy`, `precision`, `recall`, `f1`, and `roc_auc`
- stores metrics and, when configured, logs runs to Weights & Biases

Key outputs:

- `data/05_model_input/X_train.csv`
- `data/05_model_input/X_test.csv`
- `data/05_model_input/y_train.csv`
- `data/05_model_input/y_test.csv`
- `data/06_models/trained_model.pkl`
- `data/08_reporting/metrics.json`

### 3. AutoML Training

Source: [src/tennis_scores_predictions/pipelines/automl_training](/src/tennis_scores_predictions/pipelines/automl_training)

This pipeline reuses the prepared pre-encoded dataset and trains an `AutoGluon TabularPredictor`. It:

- uses the same train/test split logic as the baseline model
- fits an AutoML model under a configurable time limit
- evaluates the predictor with the same core classification metrics
- logs metrics, a leaderboard, and model artifacts to Weights & Biases when available

Key outputs:

- `data/06_models/autogluon_predictor/`
- `data/08_reporting/automl_metrics.json`

## Feature Engineering Logic

The project is designed to avoid post-match leakage by relying on information that is known before a match. The most important feature-engineering ideas are:

- symmetric player representation:
  each match is rewritten as `player_A` vs `player_B`, and the order is randomly swapped to reduce positional bias
- Elo ratings:
  historical match order is used to compute evolving player strength estimates before and after each match
- difference features:
  numeric attributes are transformed into player-to-player gaps such as `diff_rank`, `diff_rank_points`, `diff_age`, `diff_ht`, and `diff_elo_before`
- categorical encoding:
  categorical match and player attributes are one-hot encoded for modeling

## Repository Structure

```text
.
|-- conf/
|   `-- base/
|       |-- catalog.yml
|       `-- parameters.yml
|-- data/
|   |-- 01_raw/
|   |-- 02_intermediate/
|   |-- 03_primary/
|   |-- 04_feature/
|   |-- 05_model_input/
|   |-- 06_models/
|   `-- 08_reporting/
|-- notebooks/
|-- scripts/
|   `-- ingest_data.py
|-- src/tennis_scores_predictions/
|   |-- pipeline_registry.py
|   `-- pipelines/
|       |-- data_processing/
|       |-- model_training/
|       `-- automl_training/
|-- pyproject.toml
`-- requirements.txt
```

## Installation

Install project dependencies with:

```bash
pip install -r requirements.txt
```

If you want the package installed in editable mode as well:

```bash
pip install -e .
```

## Configuration

Core pipeline configuration lives in:

- [conf/base/parameters.yml](/conf/base/parameters.yml)
- [conf/base/catalog.yml](/conf/base/catalog.yml)

Important configurable values include:

- active-player cutoff date for filtering historical matches
- Random Forest hyperparameters
- Elo initialization and update settings
- AutoGluon model path, preset, evaluation metric, and time limit

Local secrets and machine-specific settings should be kept in `conf/local/` or in environment variables.

## Environment Variables

The repository uses environment variables for external integrations:

- `DATA_URL`:
  base URL used by [scripts/ingest_data.py](/scripts/ingest_data.py) to download yearly CSV files
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `WANDB_GROUP` (optional, defaults to `model-comparison`)

Example:

```bash
export DATA_URL="https://example.com/path/to/wta_matches_"
export WANDB_PROJECT="tennis-score-prediction"
export WANDB_ENTITY="your-team"
```

## How To Run

### Download and combine raw data

```bash
python scripts/ingest_data.py
```

### Run the full Kedro workflow

```bash
kedro run
```

### Run only one pipeline

```bash
kedro run --pipeline data_processing
kedro run --pipeline model_training
kedro run --pipeline automl_training
```

Because the default Kedro pipeline is the sum of all registered pipelines, `kedro run` executes the complete workflow end to end.

## Notebooks

The `notebooks/` directory contains exploratory and supporting work for:

- dataset inspection and description
- exploratory data analysis
- feature engineering experiments
- Random Forest hyperparameter tuning

These notebooks complement the production Kedro pipelines, but the reproducible project workflow lives in `src/` and `conf/`.

## Current Outputs

After a successful run, the project can produce:

- cleaned and feature-engineered tabular datasets
- train/test splits
- a trained Random Forest model
- an AutoGluon predictor directory
- JSON metric reports for both training approaches
- optional W&B experiment logs and model artifacts

## Notes

- raw datasets are intentionally not committed to the repository
- the project currently focuses on binary prediction of whether `player_A` wins a match
- the modeling workflow uses only pre-match information selected by the data-processing pipeline
