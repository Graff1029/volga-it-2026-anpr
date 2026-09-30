---
language: en
license: agpl-3.0
tags:
  - computer-vision
  - object-detection
  - license-plate
  - yolov11
  - ultralytics
  - finetuned
datasets:
  - roboflow/license-plate-recognition-rxg4e
metrics:
  - precision
  - recall
  - mAP@50
  - mAP@50-95
---

# YOLOv11-License-Plate Detection

This is a fine-tuned version of YOLOv11 (n, s, m, l, x) specialized for **License Plate Detection**, using a public dataset from Roboflow Universe:
[License Plate Recognition Dataset (10,125 images)](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11)

## ⚠️ Important Notice: Dataset Contamination

The upstream Roboflow dataset (`license-plate-recognition-rxg4e`) contains **train/test contamination** — the same source images appear in both the training and test splits with only minor manual augmentation applied (see [Discussion #2](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/discussions/2) for concrete examples). As a result:

- **The reported metrics below are likely overestimated**, because the test set is not a true held-out evaluation.
- Real-world generalization performance is expected to be lower than the numbers in the table.
- Treat all evaluation figures with caution and validate the model on your own held-out data before production use.

A clean re-split with perceptual-hash deduplication, group-aware splitting, and a re-trained `v2` release with honest metrics is planned. See **Roadmap** below.

## 🚀 Use Cases

- Smart Parking Systems
- Tollgate / Access Control Automation
- Traffic Surveillance & Enforcement
- ALPR with OCR Integration

## 🏋️ Training Details

- Base Model: YOLOv11 (`n`, `s`, `m`, `l`, `x`)
- Training Epochs: 300
- Input Size: 640x640
- Optimizer: SGD (Ultralytics default)
- Device: NVIDIA A100
- Data Format: YOLOv5-compatible (images + labels in txt)

## 📊 Evaluation Metrics (YOLOv11x)

> ⚠️ **These metrics are computed on a contaminated test split (see notice above) and should not be interpreted as a reliable measure of generalization.**

| Metric        | Value   |
|---------------|---------|
| Precision     | 0.9893  |
| Recall        | 0.9508  |
| mAP@50        | 0.9813  |
| mAP@50-95     | 0.7260  |

> For full table across models (n to x), please see the [README](README.md)

## 🐛 Known Limitations

- **Train/test leakage in upstream dataset** — see notice above. Metrics are inflated.
- **Fixed 640×640 inference resizes large images** — small or distant plates in high-resolution inputs (e.g. 1200×2400) may be missed. Workarounds: use a larger `imgsz` (e.g. 1280 or 1600), rectangular inference, or tile-based inference with [SAHI](https://github.com/obss/sahi). See [Discussion #1](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/discussions/1).
- Trained primarily on automotive license plates; performance on motorcycles, non-Latin scripts, or unusual plate formats is not guaranteed.

## 🗺️ Roadmap (v2)

1. Deduplicate the source dataset with perceptual hashing (pHash / dHash) to identify near-duplicate and augmented-variant pairs.
2. Re-split with group-aware logic so augmented variants of the same source image stay in the same fold.
3. Retrain across all model sizes and publish honest evaluation metrics.
4. Add an independent external test set for a more realistic generalization signal.

Contributions, cleaner datasets, or external benchmark suggestions are welcome via [Discussions](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/discussions).

## 📦 Model Variants

- PyTorch (.pt) — for use with Ultralytics CLI and Python API
- ONNX (.onnx) — for cross-platform inference

## 🧠 How to Use

With Python (Ultralytics API):

```python