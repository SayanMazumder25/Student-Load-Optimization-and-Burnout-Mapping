from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Student Burnout & Stress Predictor")
st.markdown("Enter your daily routine and academic target below to estimate your stress level.")
st.markdown("---")

MODEL_PATH = Path(__file__).resolve().parent / "burnout_model.joblib"
DATA_PATH = Path(__file__).resolve().parent / "cleaned_data.csv"
FEATURE_COLUMNS = [
    "Daily_Study_Hours",
    "Debate_Prep_Hours",
    "Gym_Hours",
    "Average_Sleep_Hours",
    "Academic_Score",
]

# Scale parameters: Raw data is 0–7, UI display is 0–10
RAW_STRESS_MIN = 0.0
RAW_STRESS_MAX = 7.0
DISPLAY_STRESS_MAX = 10.0


def convert_stress_to_display(raw_stress):
    """Convert the native 0–7 stress scale to an equivalent 10-point scale."""
    return (raw_stress / RAW_STRESS_MAX) * DISPLAY_STRESS_MAX


# -----------------------------------------------------------------------------
# 2. LOAD / TRAIN MODEL
# -----------------------------------------------------------------------------
def train_stress_model():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLUMNS + ["Reported_Stress_Level"]).copy()
    X = df[FEATURE_COLUMNS]
    y = df["Reported_Stress_Level"]

    # Upgraded to RandomForestRegressor to prevent overfitting
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_resource
def load_saved_model():
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            sample = pd.DataFrame([
                {
                    "Daily_Study_Hours": 6.0,
                    "Debate_Prep_Hours": 1.5,
                    "Gym_Hours": 1.0,
                    "Average_Sleep_Hours": 7.0,
                    "Academic_Score": 85.0,
                }
            ])
            prediction = float(model.predict(sample)[0])
            # Fixed validation range to match raw model scale (0.0 to 7.0)
            if RAW_STRESS_MIN <= prediction <= RAW_STRESS_MAX:
                return model
        except Exception:
            pass

    try:
        return train_stress_model()
    except FileNotFoundError:
        st.error("Model training data 'cleaned_data.csv' was not found. Please ensure project files are present.")
        st.stop()


model = load_saved_model()

# -----------------------------------------------------------------------------
# 3. USER INPUT FORM
# -----------------------------------------------------------------------------
st.subheader("Your Daily Routine & Academic Target")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Daily Study Hours", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
    debate_hours = st.number_input("Extra-Curricular Hours", min_value=0.0, max_value=24.0, value=1.5, step=0.5)
    gym_hours = st.number_input("Physical Exercise Hours", min_value=0.0, max_value=24.0, value=1.0, step=0.5)

with col2:
    sleep_hours = st.number_input("Average Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    acad_score = st.number_input("Current / Target Academic Score", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

# -----------------------------------------------------------------------------
# 4. PREDICTION LOGIC
# -----------------------------------------------------------------------------
if st.button("Predict Stress Level", type="primary", use_container_width=True):
    input_df = pd.DataFrame([
        {
            "Daily_Study_Hours": study_hours,
            "Debate_Prep_Hours": debate_hours,
            "Gym_Hours": gym_hours,
            "Average_Sleep_Hours": sleep_hours,
            "Academic_Score": acad_score,
        }
    ])

    predicted_stress_raw = float(model.predict(input_df)[0])
    predicted_stress_display = convert_stress_to_display(predicted_stress_raw)
    burnout_chance = min(max((predicted_stress_display / DISPLAY_STRESS_MAX) * 100, 0), 100)

    st.markdown("---")
    st.subheader("Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(
            label="Predicted Stress Level",
            value=f"{predicted_stress_display:.1f} / 10",
            delta=f"Equivalent to {predicted_stress_raw:.1f} / 7 raw",
        )
        st.metric(
            label="Burnout Risk Probability",
            value=f"{burnout_chance:.0f}%",
            delta="Estimated chance based on current routine",
        )

    with res_col2:
        if predicted_stress_display > 7.0:
            st.error("⚠️ **High Risk of Burnout**: Significant stress levels detected. Consider reducing active hours or increasing rest.")
        elif predicted_stress_display > 4.5:
            st.warning("⚡ **Moderate Stress**: Sustainable, but keep an eye on your rest and recovery balance.")
        else:
            st.success("✅ **Low Burnout Risk**: Healthy balance across workload, exercise, and sleep.")