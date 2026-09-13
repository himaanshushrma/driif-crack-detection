
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model = YOLO(ROOT / "runs" / "segment" / "train" / "weights" / "best.pt")

st.set_page_config(page_title="Driif AI Crack Inspector", layout="wide")

st.title("🏗️ Df AI Crack Inspector")
st.write("Upload an infrastructure image for automatic crack inspection.")

uploaded = st.file_uploader(
    "Upload Image", type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model.predict(tmp.name, conf=0.30)

    result = results[0]
    plotted = result.plot()

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original")

    with col2:
        st.image(plotted, caption="AI Detection")

    crack_count = len(result.boxes)

    confidence = 0
    if crack_count:
        confidence = float(result.boxes.conf.max())

    st.divider()

    m1, m2, m3 = st.columns(3)

    m1.metric("Cracks", crack_count)
    m2.metric("Max Confidence", f"{confidence:.2f}")

    if crack_count == 0:
        severity = "No Damage"
    elif crack_count <= 2:
        severity = "Low"
    elif crack_count <= 5:
        severity = "Medium"
    else:
        severity = "High"

    m3.metric("Severity", severity)

    st.success("Inspection Complete ✅")