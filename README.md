# Semi-Supervised Weed Detection using YOLO

## Overview

This repository implements a semi-supervised weed and crop detection pipeline using the YOLO11s (Ultralytics) model. The workflow leverages both labeled and unlabeled data to boost detection accuracy for two classes: `weed` and `crop`.

---

## Features

- **YOLO11s-based detection** with transfer learning
- **Semi-supervised learning** via pseudo-labeling
- **Custom dataset support** (YOLO format)
- **Advanced augmentation** and regularization
- **Comprehensive evaluation** (Precision, Recall, mAP)

---

## Dataset Structure

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

---

## Installation

pip install ultralytics opencv-python numpy tqdm

---

## Training

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

### 2. **Pseudo-Labeling Unlabeled Data**
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
def create_directory(self, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def generate_pseudo_labels(self, unlabeled_images_dir, output_dir):
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    self.create_directory(output_images_dir)
    self.create_directory(output_labels_dir)
    image_files = [f for f in os.listdir(unlabeled_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Processing {len(image_files)} images...")
    for img_file in tqdm(image_files):
        img_path = os.path.join(unlabeled_images_dir, img_file)
        results = self.model.predict(source=img_path, conf=self.conf_threshold, verbose=False)
        if len(results) > 0 and len(results.boxes) > 0:
            shutil.copy2(img_path, os.path.join(output_images_dir, img_file))
            label_file = os.path.splitext(img_file) + '.txt'
            label_path = os.path.join(output_labels_dir, label_file)
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]
            with open(label_path, 'w') as f:
                for box in results.boxes:
                    cls_id = int(box.cls.item())
                    x, y, w, h = box.xywhn.tolist()
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    print("\nPseudo labeling completed!")
    print(f"Labeled images saved to: {output_images_dir}")
    print(f"Labels saved to: {output_labels_dir}")

Usage:
pseudo_labeler = YOLOPseudoLabeler('results/yolo11s_weed_detection/weights/best.pt')
pseudo_labeler.generate_pseudo_labels('datasets/unlabeled/images', 'datasets/unlabeled/pseudo_labeled')

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

## Inference

model = YOLO('results/yolo11s_weed_detection/weights/best.pt')
results = model.predict(source='datasets/test/images', save=True, save_txt=True, conf=0.5)

---

## Results

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| weed  | 0.90      | 0.87   | 0.96   | 0.68      |
| crop  | 0.80      | 0.81   | 0.83   | 0.55      |
| **All** | **0.85** | **0.84** | **0.90** | **0.61** |

- **Mean F1-Score:** 0.87
- **Final Score:** 0.76

---

## Citation

If you use this code or dataset, please cite the project and refer to the final report for methodology details.

---

## Acknowledgements

- See `FINAL.ipynb` for the full pipeline and report

---

## 📄 Project Report

[📝 Download the full project report (PDF)](./Project-Report_-Semi-Supervised-Sesame-Crop-and-Weed-Detection-using-YOLOv8-1.pdf)
