"""
Train the KNN heart disease classifier and save the artifacts the app loads.

This script exists because the original model shipped in this repo cannot be
reproduced: the training code was never saved, and the artifacts it produced
contain a data-leakage bug (the scaler was fitted on the test split).

What this does differently
--------------------------
1. The scaler is fitted on the TRAIN split only, never the test split.
2. Only the five genuinely numeric columns are standardised. Binary columns
   stay as 0/1, which is correct for KNN: scaling them distorts the Euclidean
   distances that decide which patients are neighbours.
3. Missing Cholesterol / RestingBP readings (0 means "not recorded" in this
   dataset) are replaced with the TRAIN median instead of the column mean, so
   "not measured" is not conflated with a plausible-looking real value.

Because only the numeric columns are scaled, inference must scale those same
five columns. The app detects this from the scaler's ``n_features_in_``, so it
works with both the legacy artifacts and these.

Usage
-----
    python train.py                # write *_fixed.pkl artifacts
    python train.py --compare      # also explain the legacy artifacts
    python train.py --promote      # overwrite the app's artifacts
"""

import argparse
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "heart.csv"
RANDOM_STATE = 42

# True measurements that may meaningfully vary in scale.
NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# In this dataset 0 means "not recorded", not a real measurement.
ZERO_MEANS_MISSING = ["Cholesterol", "RestingBP"]

ARTIFACTS = {
    "model": "knn_heart_model.pkl",
    "scaler": "scaler.pkl",
    "columns": "heart_columns.pkl",
}
FIXED = {
    "model": "knn_heart_model_fixed.pkl",
    "scaler": "scaler_fixed.pkl",
    "columns": "heart_columns_fixed.pkl",
}


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load the raw dataset and impute missing readings with the median."""
    df = pd.read_csv(path)

    for column in ZERO_MEANS_MISSING:
        missing = df[column] == 0
        if missing.any():
            df.loc[missing, column] = df.loc[~missing, column].median()

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals with drop_first, preserving a stable order.

    drop_first means one category per encoded column is absent and must be
    represented by an all-zero row.
    """
    features = df.drop(columns=["HeartDisease"])
    categorical = [c for c in features.columns if features[c].dtype == object]

    encoded = pd.get_dummies(features, columns=categorical, drop_first=True, dtype=int)

    # Keep the original column order, with each dummy sitting where its source
    # column was, so the layout matches the documented feature list.
    order: list[str] = []
    for column in features.columns:
        if column in categorical:
            order.extend(c for c in encoded.columns if c.startswith(f"{column}_"))
        else:
            order.append(column)

    return encoded[order]


def scale_frame(frame: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Standardise only the numeric columns; leave binary columns as 0/1."""
    out = frame.copy()
    out[NUMERIC_COLS] = scaler.transform(out[NUMERIC_COLS])
    return out


def select_k(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[int, dict[int, float]]:
    """Pick k by cross-validation instead of assuming 5 is correct."""
    scores = {
        k: cross_val_score(
            KNeighborsClassifier(n_neighbors=k), x_train, y_train, cv=5, scoring="accuracy"
        ).mean()
        for k in range(1, 26, 2)
    }
    return max(scores, key=scores.get), scores


def explain_legacy() -> None:
    """Report what is wrong with the artifacts this repo originally shipped."""
    print("\n" + "=" * 62)
    print("LEGACY ARTIFACTS (contain the leakage bug)")
    print("=" * 62)

    model = joblib.load(BASE_DIR / ARTIFACTS["model"])
    scaler = joblib.load(BASE_DIR / ARTIFACTS["scaler"])
    df = pd.read_csv(DATA_FILE)

    seen, fit = scaler.n_samples_seen_, model.n_samples_fit_
    print(f"  scaler fitted on {seen} rows")
    print(f"  model  fitted on {fit} rows")
    print(f"  {seen} + {fit} = {seen + fit} = total dataset rows ({len(df)})")
    print("  -> the scaler saw the TEST split and the model saw the TRAIN split.")
    print(f"  -> scaler.scale_[Age] = {scaler.scale_[0]:.3f} but raw Age sd = {df['Age'].std():.2f}")
    print("     so the scaler was never fitted on raw values.")
    print("  -> no honest held-out score exists for that pipeline.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="explain the legacy artifacts")
    parser.add_argument("--promote", action="store_true", help="overwrite the app's artifacts")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit(f"Missing dataset: {DATA_FILE}")

    df = load_and_clean(DATA_FILE)
    print(f"Loaded {len(df)} rows")
    for column in ZERO_MEANS_MISSING:
        print(f"  imputed {(pd.read_csv(DATA_FILE)[column] == 0).sum()} missing {column} with the median")

    encoded = encode(df)
    target = df["HeartDisease"]
    print(f"Encoded to {encoded.shape[1]} features")

    x_train, x_test, y_train, y_test = train_test_split(
        encoded, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )

    # The step the original pipeline got wrong: fit on TRAIN only.
    scaler = StandardScaler().fit(x_train[NUMERIC_COLS])
    x_train_s = scale_frame(x_train, scaler)
    x_test_s = scale_frame(x_test, scaler)

    best_k, scores = select_k(x_train_s, y_train)
    print("\nCross-validated accuracy by k (train split only):")
    for k, score in scores.items():
        print(f"  k={k:2d}  {score:.4f}{'  <- best' if k == best_k else ''}")

    model = KNeighborsClassifier(n_neighbors=best_k).fit(x_train_s, y_train)
    predictions = model.predict(x_test_s)

    print(f"\nHeld-out test accuracy (k={best_k}): {accuracy_score(y_test, predictions):.4f}")
    print(f"\n{classification_report(y_test, predictions)}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    for key, filename in FIXED.items():
        joblib.dump({"model": model, "scaler": scaler, "columns": encoded.columns.tolist()}[key],
                    BASE_DIR / filename)
        print(f"Wrote {filename}")

    # Record the exact recipe so the run is reproducible.
    (BASE_DIR / "training_report.json").write_text(
        json.dumps(
            {
                "random_state": RANDOM_STATE,
                "best_k": best_k,
                "test_accuracy": accuracy_score(y_test, predictions),
                "cv_scores": {str(k): v for k, v in scores.items()},
                "scaled_columns": NUMERIC_COLS,
                "median_imputed": ZERO_MEANS_MISSING,
                "rows": len(df),
                "features": encoded.shape[1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote training_report.json")

    if args.compare:
        explain_legacy()

    if args.promote:
        for key in ARTIFACTS:
            shutil.copyfile(BASE_DIR / FIXED[key], BASE_DIR / ARTIFACTS[key])
            print(f"Promoted {FIXED[key]} -> {ARTIFACTS[key]}")


if __name__ == "__main__":
    main()