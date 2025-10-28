import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

from src.data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

# --- Helpers for currency formatting ---

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

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run training first: python -m src.models.train")
        return None
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Predictor (₹)")

st.markdown("Enter house features below and click Predict.")

# If model is missing, offer to train it now (keeps project self-contained)
if not MODEL_PATH.exists():
    st.info("Model not found yet. You can train it now (takes ~10-30s).")
    if st.button("Train model now"):
        with st.spinner("Training model..."):
            from src.models import train as _train
            _train.main()
        st.success("Training complete. You can make predictions now.")

# Currency: fixed to INR (model is trained directly in ₹)
st.caption("All prices are estimated in Indian Rupees (₹).")
# Area types matching the new training generator
locations = [
    "City",
    "Suburban",
    "Rural Area",
    "Village",
    "Hitech City",
]
col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("Location", options=locations)
    rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3, step=1)
    age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=10.0, step=1.0)
with col2:
    size_sqft = st.number_input("Size (sqft)", min_value=200.0, max_value=10000.0, value=1200.0, step=50.0)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)

if st.button("Predict"):
    model = load_model()
    if model is not None:
        X = pd.DataFrame([
            {
                "location": location,
                "size_sqft": size_sqft,
                "rooms": rooms,
                "bathrooms": bathrooms,
                "age": age,
            }
        ])[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        pred_inr = float(model.predict(X)[0])
        st.success(f"Estimated price: ₹{format_inr(pred_inr)}")

st.caption("Model is trained with a preprocessing pipeline (imputation, one-hot encoding, scaling) and several regressors with cross-validation. Best model by RMSE is saved to models/best_model.joblib.")
