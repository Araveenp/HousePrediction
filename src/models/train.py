from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.data_preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    load_data,
    make_pipeline,
    split_features_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "house_prices.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_JSON = MODELS_DIR / "metrics.json"
COMPARISON_PNG = MODELS_DIR / "model_comparison.png"

FAST_TEST = os.environ.get("FAST_TEST", "0") == "1"
CV_FOLDS = 3 if FAST_TEST else 5


def generate_synthetic_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate INR-priced data for 5 area types: City, Suburban, Rural Area, Village, Hitech City.

    - Price per sqft (₹/sqft) by area type
      Hitech City: 8k–14k, City: 9k–15k, Suburban: 5k–9k, Rural Area: 2k–4k, Village: 1.5k–3k
    - Size drives price; rooms/bathrooms small additive adjustments
    - Age depreciation ~0.5%/yr up to 30%
    - Multiplicative noise ~5% for stability
    """
    rng = np.random.default_rng(seed)

    area_ppsf = {
        "City": (9000, 15000),
        "Suburban": (5000, 9000),
        "Rural Area": (2000, 4000),
        "Village": (1500, 3000),
        "Hitech City": (8000, 14000),
    }

    areas = np.array(list(area_ppsf.keys()))
    weights = np.array([0.25, 0.35, 0.15, 0.10, 0.15])  # more Suburban/City occurrences
    weights = weights / weights.sum()

    location = rng.choice(areas, size=n, p=weights)
    size_sqft = np.clip(rng.normal(1200, 400, size=n), 500, 3200).astype(int)
    rooms = np.clip(np.round(size_sqft / 320 + rng.normal(0, 0.4, size=n)), 1, 5).astype(int)
    bathrooms = np.clip(rooms - 1 + rng.integers(0, 2, size=n), 1, 4).astype(int)
    age = np.clip(np.round(rng.normal(8, 7, size=n)), 0, 50).astype(int)

    ppsf_low = np.vectorize(lambda a: area_ppsf[a][0])(location)
    ppsf_high = np.vectorize(lambda a: area_ppsf[a][1])(location)
    price_per_sqft = rng.uniform(ppsf_low, ppsf_high)

    base_price = size_sqft * price_per_sqft
    dep_factor = 1.0 - np.minimum(age * 0.005, 0.30)
    layout_adj = (np.maximum(rooms - 2, 0) * 75_000) + (np.maximum(bathrooms - 2, 0) * 100_000)
    noise = rng.normal(1.0, 0.05, size=n)

    price = (base_price * dep_factor + layout_adj) * noise
    price = np.clip(price, 8_00_000, None)  # minimum ₹8 lakh

    df = pd.DataFrame(
        {
            "location": location,
            "size_sqft": size_sqft,
            "rooms": rooms,
            "bathrooms": bathrooms,
            "age": age,
            "price": price.astype(float),
        }
    )
    return df


def ensure_dataset_exists():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_dataset(n=300 if FAST_TEST else 800, seed=42)
        df.to_csv(DATA_PATH, index=False)
        print(f"Generated synthetic dataset at {DATA_PATH} ({len(df)} rows)")


def evaluate_models(X_train, y_train, models: Dict[str, object]) -> Dict[str, float]:
    scores = {}
    for name, model in models.items():
        pipe = make_pipeline(model)
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=CV_FOLDS, scoring="neg_root_mean_squared_error"
        )
        rmse_scores = -cv_scores
        scores[name] = float(np.mean(rmse_scores))
        print(f"{name}: CV RMSE = {scores[name]:.2f} ± {np.std(rmse_scores):.2f}")
    return scores


def plot_comparison(scores: Dict[str, float], out_path: Path) -> None:
    names = list(scores.keys())
    values = [scores[n] for n in names]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, values, color="steelblue")
    plt.ylabel("CV RMSE (lower is better)")
    plt.title("Model Comparison")
    plt.xticks(rotation=25, ha="right")
    # annotate
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> Tuple[str, float]:
    ensure_dataset_exists()

    df = load_data(str(DATA_PATH))
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if FAST_TEST:
        candidates = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
        }
    else:
        candidates = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=42),
            "Lasso": Lasso(alpha=0.0001, random_state=42, max_iter=10000),
            "ElasticNet": ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=42, max_iter=10000),
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(C=10.0, epsilon=0.2, kernel="rbf"),
            "KNN": KNeighborsRegressor(n_neighbors=7),
        }

    scores = evaluate_models(X_train, y_train, candidates)
    best_name = min(scores.keys(), key=lambda k: scores[k])
    best_model = candidates[best_name]

    # Fit best on full training set
    best_pipe = make_pipeline(best_model)
    best_pipe.fit(X_train, y_train)

    # Evaluate on holdout test set
    y_pred = best_pipe.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Save artifacts
    joblib.dump(best_pipe, BEST_MODEL_PATH)
    print(f"Saved best model: {BEST_MODEL_PATH}")

    plot_comparison(scores, COMPARISON_PNG)

    metrics = {
        "cv_rmse": scores,
        "best_model": best_name,
        "test_rmse": test_rmse,
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_JSON}")
    print(f"Saved comparison plot to {COMPARISON_PNG}")

    return best_name, test_rmse


if __name__ == "__main__":
    main()
