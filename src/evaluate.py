import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "best_model.joblib"
EVALUATION_PATH = MODEL_DIR / "evaluation_metrics.json"


def load_test_data():
    """Load processed test features and target values."""
    X_test = load_npz(PROCESSED_DIR / "X_test.npz")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["target"].astype(float)

    # The trained tree-based models expect dense input.
    X_test_dense = X_test.toarray()

    return X_test_dense, y_test


def calculate_metrics(y_true, y_pred):
    """Calculate final regression evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model file not found: {MODEL_PATH}. "
            "Please run src/train.py first."
        )

    print("Loading best model...")
    model = joblib.load(MODEL_PATH)

    print("Loading test data...")
    X_test, y_test = load_test_data()

    print("Evaluating best model...")
    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)

    output = {
        "model_file": str(MODEL_PATH),
        "evaluation_data": "held-out test set",
        "metrics": metrics,
    }

    with open(EVALUATION_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print("Evaluation completed.")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"R2: {metrics['R2']:.4f}")
    print(f"Saved evaluation metrics to: {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
