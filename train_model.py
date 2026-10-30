#!/usr/bin/env python3
"""Train a categorical-port LightGBM model and generate a Kaggle submission.

Expected data directory:
  data/train.csv
  data/test.csv
  data/sample_submission.csv

This script uses only the competition-provided data. It does not access hidden
labels or external labeled datasets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

LABELS = {"Benign", "Availability", "Reconnaissance", "Spoofing"}
CATEGORICAL_COLUMNS = ["proto", "service", "id.orig_p", "id.resp_p"]
SEED = 42


def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    required_train = {"id", "Target"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"train.csv must contain {sorted(required_train)}")
    if "id" not in test.columns:
        raise ValueError("test.csv must contain id")
    if train.drop(columns=["Target"]).columns.tolist() != test.columns.tolist():
        raise ValueError("Train/test feature columns or their order do not match")
    if sample.columns.tolist() != ["id", "Target"]:
        raise ValueError("Unexpected sample_submission.csv schema")
    if len(sample) != len(test):
        raise ValueError("Sample submission and test row counts differ")
    if train["id"].isna().any() or test["id"].isna().any():
        raise ValueError("Missing IDs found")
    if not train["id"].is_unique or not test["id"].is_unique:
        raise ValueError("IDs must be unique")
    if train["Target"].isna().any() or set(train["Target"].unique()) != LABELS:
        raise ValueError("Unexpected or missing target classes")
    return train, test, sample


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    feature_columns = [c for c in test.columns if c != "id"]
    X = train[feature_columns].copy()
    X_test = test[feature_columns].copy()
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]

    # Shared category vocabulary ensures unseen test categories are handled
    # deterministically and nominal identifiers are not treated as magnitudes.
    for column in categorical:
        train_values = X[column].fillna("__MISSING__").astype(str)
        test_values = X_test[column].fillna("__MISSING__").astype(str)
        levels = sorted(set(train_values.unique()) | set(test_values.unique()))
        X[column] = pd.Categorical(train_values, categories=levels)
        X_test[column] = pd.Categorical(test_values, categories=levels)

    return X, X_test, categorical


def build_model(n_estimators: int = 500):
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        colsample_bytree=0.85,
        subsample=0.85,
        subsample_freq=1,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame):
    if submission.columns.tolist() != ["id", "Target"]:
        raise ValueError("Submission columns must be exactly id,Target")
    if len(submission) != len(test) or len(submission) != len(sample):
        raise ValueError("Submission row count mismatch")
    if not submission["id"].equals(test["id"]):
        raise ValueError("Submission IDs/order do not match test.csv")
    if not submission["id"].equals(sample["id"]):
        raise ValueError("Submission IDs/order do not match sample_submission.csv")
    if submission["id"].isna().any() or not submission["id"].is_unique:
        raise ValueError("Submission IDs are missing or duplicated")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    invalid = set(submission["Target"].unique()) - LABELS
    if invalid:
        raise ValueError(f"Invalid target labels: {sorted(invalid)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--validate_only", action="store_true",
                        help="Run a stratified diagnostic split only; do not create a test submission.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test, sample = load_data(args.data_dir)
    X, X_test, categorical = prepare_features(train, test)
    y = train["Target"]

    if args.validate_only:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.20, random_state=SEED, stratify=y
        )
        model = build_model(args.n_estimators)
        model.fit(X_train, y_train, categorical_feature=categorical)
        pred = model.predict(X_valid)
        print(f"Random stratified holdout Macro F1: {f1_score(y_valid, pred, average='macro'):.6f}")
        print(classification_report(y_valid, pred, labels=sorted(LABELS), zero_division=0))
        print("Caution: random holdout may be optimistic under train/test distribution shift.")
        return

    model = build_model(args.n_estimators)
    model.fit(X, y, categorical_feature=categorical)
    predictions = model.predict(X_test)
    submission = pd.DataFrame({"id": test["id"], "Target": predictions})
    validate_submission(submission, test, sample)

    csv_path = args.output_dir / "submission_categorical_ports_retrained.csv"
    model_path = args.output_dir / "lightgbm_categorical_ports.txt"
    submission.to_csv(csv_path, index=False)
    model.booster_.save_model(str(model_path))
    print(f"Saved {csv_path}")
    print(f"Saved {model_path}")
    print("Predicted class counts:")
    print(submission["Target"].value_counts().to_string())
    print("Note: a fresh retraining run is a new candidate and may not exactly match the included recorded CSV.")


if __name__ == "__main__":
    main()
