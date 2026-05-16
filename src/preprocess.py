import json
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_DATA_PATH = Path("data/raw/apartments_rent_pl_2024_04-2024_06.csv")
PROCESSED_DIR = Path("data/processed")

TARGET_COLUMN = "price"
ID_COLUMN = "id"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def create_one_hot_encoder():
    """
    Create OneHotEncoder compatible with different scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_DATA_PATH}. "
            "Please download the raw CSV into data/raw first."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Raw data shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column `{TARGET_COLUMN}` not found in raw data.")

    # Drop ID before feature selection.
    # ID is unique for each listing and should not be one-hot encoded.
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Ensure sparse format before saving.
    if not sparse.issparse(X_train_processed):
        X_train_processed = sparse.csr_matrix(X_train_processed)

    if not sparse.issparse(X_test_processed):
        X_test_processed = sparse.csr_matrix(X_test_processed)

    save_npz(PROCESSED_DIR / "X_train.npz", X_train_processed)
    save_npz(PROCESSED_DIR / "X_test.npz", X_test_processed)

    y_train.to_frame("target").to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_frame("target").to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    joblib.dump(preprocessor, PROCESSED_DIR / "preprocessor.joblib")

    metadata = {
        "raw_file": str(RAW_DATA_PATH),
        "target_column": TARGET_COLUMN,
        "dropped_columns": [ID_COLUMN],
        "raw_shape_after_drop": list(df.shape),
        "train_shape": list(X_train_processed.shape),
        "test_shape": list(X_test_processed.shape),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }

    with open(PROCESSED_DIR / "preprocessing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed.")
    print(f"X_train shape: {X_train_processed.shape}")
    print(f"X_test shape: {X_test_processed.shape}")
    print(f"Saved files to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
