import streamlit as st
import joblib
import numpy as np
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "risk_model.pkl"


@st.cache_resource
def load_model():
    """Load the trained decision tree model."""
    if not MODEL_PATH.exists():
        st.error("Model artifact missing. Run notebooks/trail_risk_train.ipynb to export risk_model.pkl first.")
        st.stop()
    return joblib.load(MODEL_PATH)


def gear_advice(label: str) -> str:
    """Return gear guidance based on the predicted risk label."""
    if label == 'green':
        return (
            "Low risk: lightweight hiking shoes or trainers, light shell, 1-1.5 L of water, snacks."
        )
    if label == 'yellow':
        return (
            "Moderate risk: mid/high-cut hiking boots, waterproof shell, insulating mid-layer, 2 L of water, trekking poles."
        )
    return (
        "High risk: stiff hiking boots, full waterproof shell plus warm layer (down or heavy fleece), warm gloves, headlamp, 3 L of water; travel with partners, soloing not advised."
    )


model = load_model()

st.title("HikeSafe Advisor 🏔️")
st.write("Enter trail conditions to estimate risk and receive gear guidance.")

distance_km = st.number_input("Total distance (km)", min_value=0.0, step=0.5, value=10.0)
elevation_gain_m = st.number_input("Total elevation gain (m)", min_value=0, step=50, value=800)
max_altitude_m = st.number_input("Maximum altitude (m)", min_value=0, step=100, value=2500)
min_temperature_c = st.number_input("Expected minimum temperature (°C)", min_value=-30, max_value=40, step=1, value=4)
exposed_ridge = st.selectbox("Any exposed ridges or cliffs?", [0, 1])
slippery_surface = st.selectbox("Likely slippery/snowy/icy surface?", [0, 1])
estimated_duration_h = st.number_input("Estimated total duration (hours)", min_value=0.0, step=0.5, value=5.0)

if st.button("Assess risk"):
    features = np.array([[
        distance_km,
        elevation_gain_m,
        max_altitude_m,
        min_temperature_c,
        exposed_ridge,
        slippery_surface,
        estimated_duration_h
    ]])
    pred_label = model.predict(features)[0]

    st.subheader(f"Risk level: {pred_label.upper()}")
    st.write(gear_advice(pred_label))
