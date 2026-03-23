# ♻️ Waste Classification using Transfer Learning (MobileNetV2)

A deep learning project that classifies waste images into 10 categories using
Transfer Learning with MobileNetV2 and TensorFlow. Built for CS124P.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Real-Time Webcam](#real-time-webcam)
- [Known Limitations](#known-limitations)

---

## 🧠 Project Overview

This project builds an image classification pipeline that identifies what type
of waste an image contains. It uses Transfer Learning — reusing a MobileNetV2
model pretrained on ImageNet — and fine-tunes it for 10 waste categories.

**Why Transfer Learning?**
Training a deep CNN from scratch requires millions of images and days of
compute. MobileNetV2 already knows how to detect edges, textures, and shapes
from its ImageNet training. We reuse those weights and only teach it our
10 specific waste categories.

**Pipeline stages:**
  1. Data Preprocessing  →  split, augment, build tf.data pipelines
  2. Model Training      →  frozen base + fine-tuning phases
  3. Model Evaluation    →  classification report + confusion matrix
  4. Inference           →  single image + batch + real-time webcam

---

## 📁 Dataset

- **Source:** Garbage Classification V2 (Kaggle)
- **Folder used:** `archive/standardized_384/`
- **Total images:** 12,259
- **Split:** 70% train / 15% val / 15% test

| Class      | Total | Train | Val  | Test |
|------------|-------|-------|------|------|
| battery    | 756   | 529   | 113  | 114  |
| biological | 699   | 489   | 104  | 106  |
| cardboard  | 1411  | 987   | 211  | 213  |
| clothes    | 1892  | 1324  | 283  | 285  |
| glass      | 1736  | 1215  | 260  | 261  |
| metal      | 930   | 651   | 139  | 140  |
| paper      | 1336  | 935   | 200  | 201  |
| plastic    | 1597  | 1117  | 239  | 241  |
| shoes      | 1449  | 1014  | 217  | 218  |
| trash      | 453   | 317   | 67   | 69   |

> ⚠️ Dataset is imbalanced — `trash` has 4x fewer images than `clothes`.
> Class weights were applied during training to compensate.

---

## 🏗️ Model Architecture

```
Input (224, 224, 3)
  → mobilenet_v2.preprocess_input   [0,255] → [-1,1]
  → MobileNetV2 base (frozen)        154 layers, ImageNet weights
  → GlobalAveragePooling2D           (7,7,1280) → (1280,)
  → Dense(128, relu)
  → Dropout(0.3)
  → Dense(10, softmax)               10 class probabilities

Total params:      2,423,242
Trainable (Phase1):  165,258   (head only)
Non-trainable:     2,257,984   (frozen MobileNetV2)
```

**Training strategy:**
- Phase 1 — Base frozen, train classification head only (lr=0.001, 10 epochs)
- Phase 2 — Unfreeze top 30 MobileNetV2 layers for fine-tuning (lr=1e-5, 10 epochs)

**Callbacks:**
- `EarlyStopping` (patience=4, restores best weights)
- `ReduceLROnPlateau` (factor=0.5, patience=2)
- `ModelCheckpoint` (saves best val_accuracy only)

---

## 📊 Results

### Official Test Set (1,848 images)

**Overall Accuracy: 90.80%**

| Class      | Precision | Recall | F1-Score | Accuracy |
|------------|-----------|--------|----------|----------|
| clothes    | 0.9658    | 0.9895 | 0.9775   | 98.9% ⭐ |
| shoes      | 0.9769    | 0.9679 | 0.9724   | 96.8%    |
| biological | 0.9900    | 0.9340 | 0.9612   | 93.4%    |
| battery    | 0.9550    | 0.9298 | 0.9422   | 93.0%    |
| paper      | 0.8479    | 0.9154 | 0.8804   | 91.5%    |
| metal      | 0.8050    | 0.9143 | 0.8562   | 91.4%    |
| cardboard  | 0.9000    | 0.8873 | 0.8936   | 88.7%    |
| glass      | 0.8846    | 0.8812 | 0.8829   | 88.1%    |
| plastic    | 0.8899    | 0.8050 | 0.8453   | 80.5% ⚠️ |
| trash      | 0.8462    | 0.7971 | 0.8209   | 79.7% ⚠️ |

**Top misclassification pairs:**
- plastic → glass (19 times) — shiny/transparent surfaces look similar
- glass → plastic (17 times) — symmetric confusion
- cardboard → paper (16 times) — similar flat materials
- plastic → metal (12 times) — reflective surfaces

### Real-World Manual Test (10 personal photos)

**Score: 7/10 (70%)**

The 20% gap between official (90.8%) and real-world (70%) is expected —
training images were standardized studio shots while test photos were
taken with a phone camera under natural lighting.

---

## 📂 Project Structure

```
Here is the updated Project Structure section for your README.md. I have integrated the Flask backend and the required templates folder while maintaining the original organization.

CS124P_IC/
│
├── app.py                      # Flask Main Entry (Web Server)
├── waste_classifier_mobilenetv2.keras  # Final Trained Model
├── requirements.txt            # Dependencies
├── .gitignore                  # Git exclusion rules
├── README.md                   # Documentation
├── prediction_result.png       # Last inference visual result
│
├── scripts/                    # Core Logic & Pipelines
│   ├── data_preprocessing.py   # tf.data pipeline, splits, & augmentation
│   ├── train_model.py          # MobileNetV2 Transfer Learning logic
│   ├── model_evaluation.py     # Metrics, Reports & Confusion Matrix
│   ├── manual_upload.py        # Logic for processing file uploads
│   └── livecam_classification.py # Backend logic for the real-time stream
│
├── static/                     # Web Assets (Images/CSS/JS)
│   ├── manual_dataset/         # Library of internal test images
│   └── uploads/                # Temporary storage for user-uploaded images
│
├── templates/                  # Flask HTML UI
│   ├── home.html               # Main Dashboard
│   ├── live.html               # Real-time Webcam UI
│   └── manual.html             # Manual Analysis/Batch UI
│
├── archive/                    # Raw Dataset (Excluded from Git)
│   └── standardized_384/       # Subfolders by waste class
│
├── dataset_split/              # Auto-generated Split Folders
    ├── train/                  # 70% of data
    ├── val/                    # 15% of data
    └── test/                   # 15% of data
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/CS124P_IC.git
cd CS124P_IC
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download **Garbage Classification V2** from Kaggle and place it in:
```
archive/standardized_384/
```

---

## 🚀 How to Use

### Step 1 — Preprocess the dataset
```bash
python data_preprocessing.py
```
Splits the dataset into train/val/test and builds tf.data pipelines.
Output: `dataset_split/` folder with 3 subfolders.

### Step 2 — Train the model
```bash
python train_model.py
```
Runs Phase 1 (frozen) + Phase 2 (fine-tuning).
Output: `waste_classifier_mobilenetv2.keras`

### Step 3 — Evaluate the model
```bash
python model_evaluation.py
```
Generates classification report and confusion matrix.
Output: `confusion_matrix.png`

### Step 4 — Run inference on images
```bash
python inference.py
```
Edit the `IMAGE_PATH` or `test_images` list inside `main()` to test your own images.

---

## 📷 Real-Time Webcam

```bash
python app.py
```

- Hold a waste item in front of your webcam
- Predicted class and confidence appear on screen
- Top-3 predictions shown with confidence bars
- **UNCERTAIN** banner appears if confidence < 60%
- Press **Q** to quit

**CPU optimization features:**
- Predicts every 3rd frame (not every frame) for smooth video
- 10-frame prediction smoother to eliminate flickering
- Direct `model()` call instead of `model.predict()` for lower latency

---

## ⚠️ Known Limitations

1. **Plastic and trash** are the weakest classes (80.5% and 79.7%) due to
   visual overlap with other materials and class imbalance.

2. **Real-world performance** drops ~20% compared to dataset accuracy due to
   differences in lighting, background, and camera quality.

3. **Webcam inference runs on CPU** — predictions may lag on older machines.
   Reduce `PREDICT_EVERY_N_FRAMES` if too slow, increase if too fast.

4. **Unknown classes** — if you show an object not in the 10 classes (e.g. food),
   the model will still predict the closest match. Watch the confidence score.

---

## 🛠️ Requirements

```
tensorflow>=2.9
numpy
matplotlib
Pillow
scikit-learn
seaborn
opencv-python
```

---

## 👥 Authors

- **Marcin** — CS124P | Computer Vision & Deep Learning Project
- **Jherwin** - CS124P | User Interface and Flask

---

## 📄 License

This project is for academic purposes (CS124P coursework).
Dataset credit: Garbage Classification V2 — Kaggle.
