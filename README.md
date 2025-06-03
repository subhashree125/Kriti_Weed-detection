# 🌱 Semi-Supervised Weed Detection using YOLO 🚜

## ✨ Overview

This repository implements a semi-supervised weed and crop detection pipeline using the YOLO11s (Ultralytics) model. The workflow leverages both labeled and unlabeled data to boost detection accuracy for two classes: `weed` and `crop`.

---

## 🌟 Features

- ⚡ **YOLO11s-based detection** with transfer learning  
- 🧠 **Semi-supervised learning** via pseudo-labeling  
- 🗂️ **Custom dataset support** (YOLO format)  
- 🧰 **Advanced augmentation** and regularization  
- 📈 **Comprehensive evaluation** (Precision, Recall, mAP)  

---

## 🗃️ Dataset Structure

datasets/
labeled/
images/
labels/
unlabeled/
images/
test/
images/
labels/

- Classes: `weed` (0), `crop` (1)
- Annotation format: YOLO (`.txt` files)
- Example YAML:
path: .
train: labeled/images
val: test/images
nc: 2
names: ['weed', 'crop']

---

## ⚙️ Installation

pip install ultralytics opencv-python numpy tqdm

---

## 🏋️ Training

### 1. **Supervised Training (YOLO11s)**

from ultralytics import YOLO

model = YOLO('yolo11s.pt')
model.train(
data='weed_detection_dataset.yaml',
epochs=100,
imgsz=256,
batch=32,
lr0=0.0002,
lrf=0.002,
momentum=0.98,
weight_decay=0.0001,
optimizer='RAdam',
cos_lr=True,
mosaic=0,
dfl=2.0,
name='yolo11s_weed_detection',
project='./results/',
augment=True,
verbose=True,
device='cpu'
)

---

### 2. **Pseudo-Labeling Unlabeled Data** 🤖

from ultralytics import YOLO
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import cv2

class YOLOPseudoLabeler:
def init(self, model_path, confidence_threshold=0.5):
self.model = YOLO(model_path)
self.conf_threshold = confidence_threshold

---

### 3. **Semi-Supervised Training (Combine Labeled + Pseudo-Labeled)**
- Merge your labeled and pseudo-labeled data into a new dataset folder (e.g., `combined_dataset/`).
- Update your YAML accordingly.

model = YOLO('results/yolo11s_weed_detection/weights/best.pt')
model.train(
data='combined_dataset/combined_dataset.yaml',
epochs=200,
imgsz=256,
batch=32,
lr0=0.0002,
weight_decay=0.0001,
optimizer='RAdam',
cos_lr=True,
augment=True,
mosaic=0,
name='semi_supervised',
project='./results/',
device='cpu'
)

---

## 🔍 Inference

model = YOLO('results/yolo11s_weed_detection/weights/best.pt')
results = model.predict(source='datasets/test/images', save=True, save_txt=True, conf=0.5)

---

## 🏆 Results

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| weed  | 0.90      | 0.87   | 0.96   | 0.68      |
| crop  | 0.80      | 0.81   | 0.83   | 0.55      |
| **All** | **0.85** | **0.84** | **0.90** | **0.61** |

- **Mean F1-Score:** 0.87
- **Final Score:** 0.76

---

## 📄 Project Report

[📝 View the full project report (PDF)](./Project%20Report_%20Semi-Supervised%20Sesame%20Crop%20and%20Weed%20Detection%20using%20YOLOv8%20%281%29.pdf)


Or [view it in your browser](.[/Project-Report_-Semi-Supervised-Sesame-Crop-and-Weed-Detection-using-YOLOv8-1.pdf](https://github.com/subhashree125/Kriti_Weed-detection/blob/main/Project%20Report_%20Semi-Supervised%20Sesame%20Crop%20and%20Weed%20Detection%20using%20YOLOv8%20(1).pdf)).

---

## 🙏 Acknowledgements

- See `FINAL.ipynb` for the full pipeline and report

---

> ⭐️ If you like this project, please star the repo!  
> 🐛 For issues or suggestions, open an issue or pull request.

