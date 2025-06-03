# 🌿 Semi-Supervised Weed Detection using YOLOv8

> A KRITI’25 Challenge Submission | IIT Guwahati

## 📌 Overview

This project tackles the challenge of detecting **weeds in sesame crop images** using **semi-supervised learning**. Instead of relying solely on annotated data, we leverage a small labeled dataset and a large set of unlabeled images to build a high-accuracy weed detection model.

The project was developed as part of the **4i Labs Semi-Supervised Weed Detection Challenge** organized during **KRITI’25** at **IIT Guwahati**.

---

## 📁 Dataset

- **Labeled images**: 200 (with bounding boxes for weed/crop)
- **Unlabeled images**: 1000 (no annotations)
- **Test images**: 100 (used for evaluation)

All data was provided by the organizers; **no external datasets** were used.

---

## ⚙️ Model Architecture

- Model: **YOLOv8 (YOLO11s.pt)**
- Output: Bounding boxes with confidence scores for 2 classes – crop and weed
- Framework: PyTorch + Ultralytics YOLO

---

## 🧠 Semi-Supervised Learning Approach

1. **Baseline Training** on 200 labeled images
2. **Pseudo-labeling** using initial YOLO model to annotate 1000 unlabeled images
3. **Retraining** the YOLO model on the combined dataset (labeled + pseudo-labeled)

---

## 🔁 Data Augmentation

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Blurring (Gaussian, Median)
- Grayscale conversion
- Horizontal Flipping
- Scaling, Translation

---

## 🛠 Training Details

- **Epochs**: 100  
- **Batch Size**: 32  
- **Image Size**: 256x256  
- **Optimizer**: RAdam  
- **Learning Rate**: 0.0002  

---

## 📊 Results

| Metric           | Value     |
|------------------|-----------|
| **Accuracy**     | 93.88%    |
| **F1-Score**     | 0.94      |
| **mAP@0.5:0.95** | 0.174     |
| **Final Score**  | 0.557     |

### Confusion Matrix
