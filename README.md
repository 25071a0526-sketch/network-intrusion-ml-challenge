# Network Intrusion Detection ML Challenge

A reproducible LightGBM baseline for four-class network-traffic classification.

## Objective

Predict one of these labels for each test record:

- `Benign`
- `Availability`
- `Reconnaissance`
- `Spoofing`

The competition metric is Macro F1. The required prediction schema is exactly `id,Target`.

## Repository contents

- `train_model.py` — training, optional diagnostic validation, and submission generation.
- `requirements.txt` — Python dependencies.
- `submission_categorical_ports.csv` — the saved categorical-port prediction candidate retained from the project experiments. This is a prediction artifact, not a guarantee that a fresh retraining run will reproduce it byte-for-byte.
- `.gitignore` — excludes local datasets, virtual environments, and temporary files.
- `data/` — empty placeholder. Put the competition-provided `train.csv`, `test.csv`, and `sample_submission.csv` here locally; these data files are intentionally not committed.

## Recorded competition result

The project log records a public Kaggle Macro F1 of **0.84847** for the categorical-port LightGBM candidate represented by the saved prediction CSV. Treat this as a recorded public result, not a guarantee of private leaderboard performance. The public leaderboard is not a substitute for a representative validation set.

## Method

1. Exclude `id` from the model features.
2. Use only the competition-provided training data.
3. Treat `proto`, `service`, `id.orig_p`, and `id.resp_p` as categorical identifiers rather than continuous magnitudes.
4. Create a shared train/test category vocabulary so unseen test categories are handled consistently.
5. Train a multiclass LightGBM classifier.
6. Validate the submission schema, row count, ID order, uniqueness, missing values, and permitted target labels before writing predictions.

The training/test distributions are substantially different, and ordinary random holdout validation can be optimistic. Do not interpret random-split Macro F1 as an estimate of the hidden test score.

## Setup

Use Python 3.10 or 3.11 in a virtual environment. Install the pinned dependency versions:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place these files in `data/`:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Run a diagnostic validation split

```bash
python train_model.py --data_dir data --output_dir artifacts --validate_only
```

This produces a stratified random-holdout diagnostic. Because the train/test distributions differ materially, this score should not be treated as a reliable hidden-test estimate.

## Train and generate a fresh candidate

```bash
python train_model.py --data_dir data --output_dir artifacts
```

Outputs:

- `artifacts/submission_categorical_ports_retrained.csv`
- `artifacts/lightgbm_categorical_ports.txt`

A fresh run is a new candidate; it is not asserted to reproduce the saved prediction CSV exactly. Use the saved CSV when you need the already-evaluated candidate.

## Integrity checks

The script checks that:
- The train/test feature columns and order match.
- IDs are present and unique.
- The sample submission has the expected `id,Target` columns and row count.
- All four target labels are present in training.
- The output has exactly `id,Target`.
- Submission IDs and order match both test data and sample submission.
- Predictions contain no missing values or invalid labels.

## Responsible use and competition compliance

This repository is intended for the stated competition. It uses the supplied training labels and does not require external labeled data, manual test labeling, or hidden test labels. Review the current official competition rules before submission and ensure any collaboration or code-sharing follows those rules.

## Limitations

- Severe train/test distribution shift was observed in the project analysis.
- The available unlabeled test data cannot establish that the relationship between features and labels is stable.
- The saved public score is not a guarantee of private leaderboard performance.
- This is a competition model, not a validated production intrusion-detection system.
