import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

RANDOM_STATE = 42
CV_FOLDS = 3


def make_json_safe(value):
    """Convert numpy/pandas objects into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value) if not isinstance(value, (dict, list, str)) else False:
        return None
    return value


def load_processed_data():
    """Load processed features and target values created by src/preprocess.py."""
    X_train = load_npz(PROCESSED_DIR / "X_train.npz")
    X_test = load_npz(PROCESSED_DIR / "X_test.npz")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["target"].astype(float)
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["target"].astype(float)

    return X_train, X_test, y_train, y_test


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred, prefix):
    """Calculate regression metrics with a prefix such as Train or Test."""
    return {
        f"{prefix}_RMSE": rmse(y_true, y_pred),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_fitted_model(model, model_name, X_train, X_test, y_train, y_test, tuning_method, best_params, best_cv_rmse=None):
    """Evaluate a fitted model on both train and held-out test sets."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    result = {
        "Model": model_name,
        **regression_metrics(y_train, train_pred, "Train"),
        **regression_metrics(y_test, test_pred, "Test"),
        "Best_CV_RMSE": best_cv_rmse,
        "Tuning_Method": tuning_method,
        "Best_Params": json.dumps(make_json_safe(best_params)),
    }

    return result


def train_regression_tree_baseline(X_train, X_test, y_train, y_test):
    """
    Train an unpruned regression tree.

    Purpose:
    - baseline model
    - shows the overfitting risk of a single decision tree
    """
    print("\nTraining baseline Regression Tree...")

    model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    result = evaluate_fitted_model(
        model=model,
        model_name="Regression Tree",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tuning_method="None - baseline unpruned tree",
        best_params={"random_state": RANDOM_STATE},
        best_cv_rmse=None,
    )

    return model, result, pd.DataFrame()


def train_pruned_tree(X_train, X_test, y_train, y_test):
    """
    Train a cost-complexity pruned regression tree.

    ccp_alpha is selected using 3-fold CV on the training set.
    """
    print("\nSelecting ccp_alpha for Pruned Tree...")

    base_tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
    base_tree.fit(X_train, y_train)

    path = base_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = np.unique(path.ccp_alphas)

    # Remove extremely large alpha values for stability.
    alpha_max = np.quantile(ccp_alphas, 0.99)
    ccp_alphas = ccp_alphas[ccp_alphas <= alpha_max]

    # Use a manageable and interpretable number of alpha candidates.
    max_candidates = 30
    if len(ccp_alphas) > max_candidates:
        idx = np.linspace(0, len(ccp_alphas) - 1, max_candidates).astype(int)
        alpha_grid = ccp_alphas[idx]
    else:
        alpha_grid = ccp_alphas

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
        param_grid={"ccp_alpha": alpha_grid},
        scoring=rmse_scorer,
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_cv_rmse = -float(search.best_score_)
    best_params = search.best_params_

    result = evaluate_fitted_model(
        model=best_model,
        model_name="Pruned Tree",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tuning_method="Cost-complexity pruning with 3-fold GridSearchCV",
        best_params=best_params,
        best_cv_rmse=best_cv_rmse,
    )

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["Model"] = "Pruned Tree"

    print(f"Best Pruned Tree CV RMSE: {best_cv_rmse:.2f}")
    print(f"Best Pruned Tree params: {best_params}")

    return best_model, result, cv_results


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train Random Forest.

    We keep the original project logic:
    - n_estimators = 200
    - max_features = sqrt
    - tune max_depth using 3-fold CV
    """
    print("\nSelecting max_depth for Random Forest...")

    depth_candidates = [None, 10, 20, 30, 40, 50]
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_rows = []

    for depth in depth_candidates:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=depth,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        cv_rmse_mean = -float(scores.mean())
        cv_rmse_std = float(scores.std())

        cv_rows.append(
            {
                "Model": "Random Forest",
                "param_max_depth": depth,
                "mean_test_RMSE": cv_rmse_mean,
                "std_test_RMSE": cv_rmse_std,
            }
        )

    cv_results = pd.DataFrame(cv_rows).sort_values("mean_test_RMSE").reset_index(drop=True)
    best_depth = cv_results.loc[0, "param_max_depth"]

    if pd.isna(best_depth):
        best_depth = None
    elif isinstance(best_depth, float) and best_depth.is_integer():
        best_depth = int(best_depth)

    print(f"Best Random Forest max_depth: {best_depth}")
    print(f"Best Random Forest CV RMSE: {cv_results.loc[0, 'mean_test_RMSE']:.2f}")

    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=best_depth,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    final_model.fit(X_train, y_train)

    best_params = {
        "n_estimators": 200,
        "max_depth": best_depth,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
    }

    result = evaluate_fitted_model(
        model=final_model,
        model_name="Random Forest",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tuning_method="3-fold CV over max_depth",
        best_params=best_params,
        best_cv_rmse=float(cv_results.loc[0, "mean_test_RMSE"]),
    )

    return final_model, result, cv_results


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    Train Gradient Boosting.

    This grid follows a small, interpretable search over the three most important
    Gradient Boosting parameters:
    - learning_rate
    - max_depth
    - n_estimators
    """
    print("\nSelecting hyperparameters for Gradient Boosting...")

    param_grid = {
        "learning_rate": [0.2, 0.3],
        "max_depth": [4, 5],
        "n_estimators": [1000, 1200],
    }

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    search = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring=rmse_scorer,
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_cv_rmse = -float(search.best_score_)
    best_params = search.best_params_

    result = evaluate_fitted_model(
        model=best_model,
        model_name="Gradient Boosting",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tuning_method="3-fold GridSearchCV over learning_rate, max_depth, n_estimators",
        best_params=best_params,
        best_cv_rmse=best_cv_rmse,
    )

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["Model"] = "Gradient Boosting"

    print(f"Best Gradient Boosting CV RMSE: {best_cv_rmse:.2f}")
    print(f"Best Gradient Boosting params: {best_params}")

    return best_model, result, cv_results


def save_outputs(best_model_name, best_model, comparison_df, cv_results_df):
    """Save best model, comparison table, CV results, and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_model_path = MODEL_DIR / "best_model.joblib"
    comparison_path = MODEL_DIR / "model_comparison.csv"
    cv_results_path = MODEL_DIR / "cv_results.csv"
    metadata_path = MODEL_DIR / "model_metadata.json"

    joblib.dump(best_model, best_model_path)
    comparison_df.to_csv(comparison_path, index=False)
    cv_results_df.to_csv(cv_results_path, index=False)

    metadata = {
        "best_model": best_model_name,
        "selection_metric": "Test_RMSE",
        "selection_rule": "Lowest RMSE on the held-out test set",
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "models_compared": comparison_df["Model"].tolist(),
        "results": make_json_safe(comparison_df.to_dict(orient="records")),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nBest model selected:", best_model_name)
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved model comparison to: {comparison_path}")
    print(f"Saved CV results to: {cv_results_path}")
    print(f"Saved model metadata to: {metadata_path}")


def main():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Dense input is safe here because preprocessing produced only 52 features.
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    models = {}
    results = []
    cv_results_list = []

    tree_model, tree_result, tree_cv = train_regression_tree_baseline(
        X_train_dense,
        X_test_dense,
        y_train,
        y_test,
    )
    models["Regression Tree"] = tree_model
    results.append(tree_result)

    pruned_model, pruned_result, pruned_cv = train_pruned_tree(
        X_train_dense,
        X_test_dense,
        y_train,
        y_test,
    )
    models["Pruned Tree"] = pruned_model
    results.append(pruned_result)
    cv_results_list.append(pruned_cv)

    rf_model, rf_result, rf_cv = train_random_forest(
        X_train_dense,
        X_test_dense,
        y_train,
        y_test,
    )
    models["Random Forest"] = rf_model
    results.append(rf_result)
    cv_results_list.append(rf_cv)

    gb_model, gb_result, gb_cv = train_gradient_boosting(
        X_train_dense,
        X_test_dense,
        y_train,
        y_test,
    )
    models["Gradient Boosting"] = gb_model
    results.append(gb_result)
    cv_results_list.append(gb_cv)

    comparison_df = pd.DataFrame(results).sort_values(
        "Test_RMSE",
        ascending=True,
    ).reset_index(drop=True)

    best_model_name = comparison_df.loc[0, "Model"]
    best_model = models[best_model_name]

    cv_results_df = pd.concat(cv_results_list, ignore_index=True, sort=False)

    print("\nFinal model comparison:")
    print(comparison_df)

    save_outputs(best_model_name, best_model, comparison_df, cv_results_df)


if __name__ == "__main__":
    main()
