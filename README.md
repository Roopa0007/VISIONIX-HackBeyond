 📘 README.md — VISIONIX: Safety Equipment Detection for Space Stations

 🚀 VISIONIX — Space Station Safety Equipment Detection

HackBeyond Hackathon 2025 — Final Submission

VISIONIX is an advanced object detection system designed to identify critical safety equipment inside space station environments.
Using YOLOv8, custom-trained on synthetic datasets generated via **Duality AI’s Falcon simulator**, the model accurately detects:

* 🧯 Fire Extinguishers
* 🔥 Fire Alarms
* 🩹 First Aid Boxes
* ☎ Emergency Phones
* 🧪 Oxygen Tanks
* 🧪 Nitrogen Tanks
* ⚡ Safety Switch Panels

The system includes a minimal, clean Streamlit interface for real-time testing on any image.

---

## 🧑‍🚀 Team VISIONIX

* email: roopasreeroyal007@gmail.com

---

## 🛠️ Installation

### 1️⃣ Create Environment

```bash
conda create -n visionix python=3.9 -y
conda activate visionix
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App

From inside the repo:

```bash
streamlit run scripts/app_streamlit.py
```

This launches the UI where you can upload an image and get detection results instantly.

---

## 📊 Model Performance

* Highest mAP50 achieved on test images
0.589 (58.9% mAP @ 0.5 IoU)

* Verified using official test dataset provided by HackBeyond.

---

## 🎯 Features

### ✔ Single-Image Detection

Upload or use sample images to perform YOLOv8 detection.

### ✔ Class Filtering

Select specific classes like “Fire Extinguisher” or “Emergency Phone”.

### ✔ Detection Summary

* Detected classes
* Not detected (for selected classes)
* Bounding box + confidence scores

### ✔ Ultra-light Repo

Only essential files included:

* Best model
* Minimal scripts
* Few demo images

---

## 📦 Bonus Challenge

YES — Completed

We implemented:

* Class-based filtering
* Not-detected tracking
* Light-weight, deployment-ready inference app

---

## 🌐 Dataset Source

The training data was synthetically generated from **Duality AI – Falcon Simulator**, containing multiple lighting and clutter conditions such as:

* light / dark / very dark
* clutter / unclutter
* mirrored perspectives

---

## 🧪 Run Inference from Command Line

```bash
python scripts/predict.py --img_path demo_images/000000003_light_unclutter.png
```

---

## 🏁 Final Submission Items Required by Judges

| Item                    | Status                    |
| ----------------------- | ------------------------- |
| GitHub Repo             | ✔ Done                    |
| Streamlit App           | ✔ Working                 |
| mAP Score               | ✔ 58.9%mAP                |
| Bonus Challenge         | ✔ Completed               |
| All team files uploaded | ✔ Included                |

---

## ❤️ Acknowledgment

This project was built as part of HackwithBeyond — Space Station Safety Hackathon 2025, using synthetic data from Duality AI Falcon Simulator.

