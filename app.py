import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"
METRICS_PATH = ROOT / "outputs" / "evaluation_metrics.json"

st.set_page_config(page_title="Structural Damage AI", page_icon="🏗️", layout="wide")
st.title("🏗️ Structural Damage Intelligence")
st.caption("DNN-based preliminary concrete crack screening")
st.warning(
    "Educational prototype only. Image predictions are not structural engineering "
    "assessments and cannot establish that a building is safe."
)

try:
    health = requests.get(f"{API_URL}/api/health", timeout=3).json()
    ready = health.get("model_loaded", False)
except requests.RequestException:
    ready = False

if ready and METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Loaded")
    c2.metric("Held-out test accuracy", f"{metrics['test_accuracy']:.1%}")
    c3.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
else:
    st.info("Model not ready. Prepare the dataset, train the model, then restart the API.")

st.divider()
st.subheader("Analyze an image")
uploaded = st.file_uploader("Upload a concrete surface image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with right:
        if st.button("Analyze image", type="primary", disabled=not ready):
            try:
                response = requests.post(
                    f"{API_URL}/api/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )
                if response.ok:
                    result = response.json()
                    st.metric("Prediction", result["predicted_class"].replace("_", " ").title())
                    st.metric("Model probability", f"{result['confidence']:.1%}")
                    st.write(result["recommendation"])
                    st.bar_chart(result["probabilities"])
                    st.caption(result["disclaimer"])
                else:
                    st.error(f"API error {response.status_code}: {response.text}")
            except requests.RequestException as exc:
                st.error(f"Could not connect to API: {exc}")

st.divider()
st.subheader("Recent inspection history")
try:
    history = requests.get(f"{API_URL}/api/inspections?limit=50", timeout=3)
    if history.ok and history.json():
        records = history.json()
        frame = pd.DataFrame([{
            "Timestamp (UTC)": r["created_at"],
            "Filename": r["filename"],
            "Prediction": r["predicted_class"],
            "Probability": f"{r['confidence']:.1%}",
        } for r in records])
        st.dataframe(frame, use_container_width=True, hide_index=True)
    elif history.ok:
        st.write("No inspections recorded yet.")
    else:
        st.info("Inspection history is unavailable.")
except requests.RequestException:
    st.info("Start the API to display inspection history.")

with st.expander("About the model and its limitations"):
    st.markdown("""
    - The starter model classifies **crack vs. no crack** only.
    - Model probability is not a calibrated measure of structural risk.
    - A single image may miss hidden defects, reinforcement corrosion, or damage outside the frame.
    - Results are affected by lighting, camera distance, surface texture, and dataset bias.
    - Any safety concern must be evaluated by a qualified structural engineer.
    """)
