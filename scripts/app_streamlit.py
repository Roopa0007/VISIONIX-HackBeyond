# app_streamlit.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
from pathlib import Path

# ---------------------------------------------
# PATHS AND CONSTANTS
# ---------------------------------------------
BASE = Path(__file__).parent
MODEL_PATH = BASE / "runs" / "detect" / "train" / "weights" / "best.pt"

CLASS_NAMES = [
    'OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm',
    'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher'
]

# ---------------------------------------------
# LOAD MODEL
# ---------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

# ---------------------------------------------
# FILTER DETECTIONS
# ---------------------------------------------
def filter_detections(result, allowed_classes):
    if not allowed_classes:
        return result

    new_boxes = []
    for box in result.boxes:
        cls_name = CLASS_NAMES[int(box.cls)]
        if cls_name in allowed_classes:
            new_boxes.append(box)

    result.boxes = new_boxes
    return result

# ---------------------------------------------
# RUN INFERENCE
# ---------------------------------------------
def run_inference(model, img_pil, conf=0.5, allowed_classes=None):
    img = np.array(img_pil.convert("RGB"))

    t0 = time.time()
    results = model.predict(source=img, conf=conf, device="cpu")
    result = results[0]

    # filter selected classes
    result = filter_detections(result, allowed_classes)
    rendered = result.plot()

    detections = []
    for box in result.boxes:
        cls = int(box.cls)
        confv = float(box.conf)
        xyxy = box.xyxy[0].tolist()

        detections.append({
            "class_name": CLASS_NAMES[cls],
            "confidence": round(confv, 3),
            "bbox": [round(v, 2) for v in xyxy]
        })

    return rendered[:, :, ::-1], detections, round(time.time() - t0, 3)

# ---------------------------------------------
# MAIN APP
# ---------------------------------------------
def main():

    st.set_page_config(layout="wide", page_title="HackBeyond Detection App")

    st.title("🚀 HackBeyond Safety Equipment Detection App")

    model = load_model()

    # ---------------- SIDEBAR ----------------
    st.sidebar.header("⚙ Settings")

    conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.95, 0.5)

    selected_classes = st.sidebar.multiselect(
        "Show Only These Classes (optional)",
        CLASS_NAMES
    )
    # -----------------------------------------

    uploaded = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

    # Run sample image
    if st.button("Run Sample Image"):
        sample_folder = BASE.parent / "test3" / "images"
        samples = list(sample_folder.glob("*.png")) + list(sample_folder.glob("*.jpg"))
        if samples:
            uploaded = samples[0].open("rb")

    # ---------------- RUNNING DETECTION ----------------
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Input Image", use_container_width=True)

        if st.button("Detect"):
            st.subheader("Running YOLO inference...")
            out_img, detections, t = run_inference(model, img, conf, selected_classes)

            st.image(out_img, caption=f"Detections (Time: {t}s)", use_container_width=True)

            # ---------------- Detection Summary ----------------
            detected = [d["class_name"] for d in detections]

            st.subheader("📌 Detection Summary")
            st.write("Detected:", detected)

            if selected_classes:
                not_detected = [cls for cls in selected_classes if cls not in detected]
                st.write("Not Detected:", not_detected)

            st.write("Raw Detections:", detections)

    else:
        st.info("Upload an image to begin or click 'Run Sample Image'.")

# ---------------------------------------------
if __name__ == "__main__":
    main()
