"""Train and save the heating-load prediction pipeline.

Dataset: UCI Energy Efficiency (ENB2012) - 768 simulated buildings, 8 features.
Target : Y1 - heating load (kWh/m2).

Usage:
    python train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "ENB2012_data.xlsx"
MODEL_PATH = ROOT / "models" / "heating_model.joblib"
RESULTS_PATH = ROOT / "models" / "results.json"

FEATURES = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
]
TARGET = "heating_load"
RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    df = df.rename(columns={f"X{i}": name for i, name in enumerate(
        [
            "relative_compactness",
            "surface_area",
            "wall_area",
            "roof_area",
            "overall_height",
            "orientation",
            "glazing_area",
            "glazing_area_distribution",
        ],
        start=1,
    )})
    df = df.rename(columns={"Y1": "heating_load", "Y2": "cooling_load"})
    return df


def candidates() -> dict[str, tuple[Pipeline, dict]]:
    """At least three suitable regression algorithms, each paired with a search grid."""
    return {
        "LinearRegression": (
            Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            {},
        ),
        "KNeighborsRegressor": (
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
            {"model__n_neighbors": [3, 5, 7, 9, 11, 15]},
        ),
        "SVR": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR()),
                ]
            ),
            {
                "model__C": [1, 10, 100, 1000],
                "model__gamma": ["scale", 0.01, 0.1],
                "model__kernel": ["rbf"],
            },
        ),
        "RandomForestRegressor": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
                ]
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    }


def main() -> None:
    df = load_data()
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"train={len(X_train)} test={len(X_test)} features={len(FEATURES)}")

    results: dict[str, dict] = {}
    fitted: dict[str, GridSearchCV] = {}

    for name, (pipeline, grid) in candidates().items():
        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            cv=CV,
            scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
            refit="r2",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        fitted[name] = search

        # Honest generalisation estimate from cross-validation (never the test set).
        cv_scores = cross_validate(search.best_estimator_, X_train, y_train, cv=CV,
                                   scoring=["r2", "neg_root_mean_squared_error"])
        results[name] = {
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
            "cv_r2": float(cv_scores["test_r2"].mean()),
            "cv_rmse": float(-cv_scores["test_neg_root_mean_squared_error"].mean()),
        }
        print(f"{name:24s} cv R2={results[name]['cv_r2']:.4f} "
              f"cv RMSE={results[name]['cv_rmse']:.3f} params={search.best_params_}")

    best_name = max(results, key=lambda n: results[n]["cv_r2"])
    print(f"\nSelected on CV R2: {best_name}")

    # Final test-set evaluation, once, after model selection.
    final_pipeline = fitted[best_name].best_estimator_
    # Wrap so the deployed object accepts a DataFrame and returns a plain float array.
    deployable = TransformedTargetRegressor(regressor=final_pipeline, transformer=None)
    deployable.fit(X_train, y_train)

    y_pred = deployable.predict(X_test)
    test_metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    print(f"Test: R2={test_metrics['r2']:.4f} RMSE={test_metrics['rmse']:.3f} "
          f"MAE={test_metrics['mae']:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(deployable, MODEL_PATH)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "features": FEATURES,
                "target": TARGET,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "models": results,
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )
    print(f"\nSaved -> {MODEL_PATH} and {RESULTS_PATH}")


if __name__ == "__main__":
    main()
