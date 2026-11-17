"""Streamlit app: predict a building's heating load.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
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

ORIENTATIONS = {
    "2 - North": 2,
    "3 - East": 3,
    "4 - South": 4,
    "5 - West": 5,
}
GLAZING_DISTRIBUTIONS = {
    "0 - Uniform (not glazed)": 0.0,
    "1 - Uniform": 1.0,
    "2 - N/S glazing": 2.0,
    "3 - E/W glazing": 3.0,
    "4 - N/S + 10%": 4.0,
    "5 - E/W + 10%": 5.0,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


st.set_page_config(page_title="Building Heating Load Predictor", page_icon="🏢")

st.title("🏢 Building Heating Load Predictor")
st.write(
    "Estimate the **heating load** (energy needed to keep a building warm, "
    "kWh/m²) from its geometry and glazing properties, using a model trained "
    "on the UCI Energy Efficiency dataset (768 simulated buildings)."
)

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
results = load_results()

with st.form("predict"):
    st.subheader("Building characteristics")
    col1, col2 = st.columns(2)

    with col1:
        relative_compactness = st.slider(
            "Relative compactness", 0.62, 0.98, 0.76, 0.02,
            help="Overall surface-to-volume ratio of the building shape.",
        )
        surface_area = st.number_input(
            "Surface area (m²)", 514.5, 808.5, 671.5, 10.0,
        )
        wall_area = st.number_input(
            "Wall area (m²)", 245.0, 416.5, 318.5, 10.0,
        )
        roof_area = st.number_input(
            "Roof area (m²)", 110.25, 220.5, 176.6, 10.0,
        )

    with col2:
        overall_height = st.selectbox(
            "Overall height (m)", options=[3.5, 7.0], index=1,
            help="One or two storeys: 3.5 m or 7.0 m.",
        )
        orientation = st.selectbox("Orientation", list(ORIENTATIONS), index=1)
        glazing_area = st.slider(
            "Glazing area (fraction of floor)", 0.0, 0.4, 0.25, 0.05,
        )
        glazing_distribution = st.selectbox(
            "Glazing area distribution", list(GLAZING_DISTRIBUTIONS), index=1,
        )

    submitted = st.form_submit_button("Predict heating load", type="primary")

if submitted:
    row = {
        "relative_compactness": relative_compactness,
        "surface_area": surface_area,
        "wall_area": wall_area,
        "roof_area": roof_area,
        "overall_height": overall_height,
        "orientation": ORIENTATIONS[orientation],
        "glazing_area": glazing_area,
        "glazing_area_distribution": GLAZING_DISTRIBUTIONS[glazing_distribution],
    }
    # Same feature order the model was trained with.
    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    prediction = float(model.predict(X)[0])
    band = 0.5 * (results.get("test_metrics", {}).get("rmse", 1.0))

    st.success(f"Estimated heating load: **{prediction:.2f} kWh/m²**")
    st.caption(f"Typical prediction error (test RMSE) is about ±{band:.2f} kWh/m².")

    if results:
        with st.expander("Model information"):
            st.write(f"Selected model: **{results['selected_model']}**")
            m = results["test_metrics"]
            st.write(
                f"Test set: R² = {m['r2']:.4f}, RMSE = {m['rmse']:.3f}, "
                f"MAE = {m['mae']:.3f}"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Model": name, "CV R²": v["cv_r2"], "CV RMSE": v["cv_rmse"],
                         "Best params": str(v["best_params"])}
                        for name, v in results["models"].items()
                    ]
                ),
                hide_index=True,
            )

st.divider()
st.caption(
    "**Limitations:** The model was trained on simulated building configurations "
    "rather than real measurements. Predictions are an educational estimate of "
    "heating load in relative terms and are not a substitute for an energy "
    "audit, engineering design, or regulatory compliance calculation."
)
