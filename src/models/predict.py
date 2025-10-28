from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first: python -m src.models.train"
        )
    return joblib.load(model_path)


def predict_single(location: str, size_sqft: float, rooms: int, bathrooms: int, age: float,
                   model_path: Path = DEFAULT_MODEL_PATH) -> float:
    model = load_model(model_path)
    data = {"location": [location], "size_sqft": [size_sqft], "rooms": [rooms],
            "bathrooms": [bathrooms], "age": [age]}
    X = pd.DataFrame(data)[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    pred = float(model.predict(X)[0])
    return pred


def format_inr(value: float) -> str:
    n = int(round(value))
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    if len(s) <= 3:
        return f"{sign}{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return f"{sign}{','.join(parts)},{last3}"


def main():
    parser = argparse.ArgumentParser(description="Predict house price from features.")
    parser.add_argument("--location", required=True, type=str)
    parser.add_argument("--size_sqft", required=True, type=float)
    parser.add_argument("--rooms", required=True, type=int)
    parser.add_argument("--bathrooms", required=True, type=int)
    parser.add_argument("--age", required=True, type=float)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    pred = predict_single(
        location=args.location,
        size_sqft=args.size_sqft,
        rooms=args.rooms,
        bathrooms=args.bathrooms,
        age=args.age,
        model_path=Path(args.model),
    )

    # Model outputs INR directly
    print(f"Predicted price: ₹{format_inr(pred)}")


if __name__ == "__main__":
    main()
