# Gauge Vision — Dataset Augmentation & Pose/Detection Pipeline

Professional toolkit for creating, augmenting, and running vision models
to detect gauge elements and estimate pose. This repository contains
augmentation scripts, dataset structure, and example models for
training and inference using YOLO-style workflows.

**Contents**
- **augment_script.py**: Image augmentation pipeline used to generate
  synthetic training samples placed under `augmented_data/`.
- **reading.py**: Utilities for reading model outputs and converting
  detections to human-readable gauge values.
- **gauge_pose.yaml**: Pose/dataset configuration used for training
  pose estimation models.
- **yolo26n.pt** and **yolov8n-pose.pt**: Example/baseline model
  weights (committed for convenience; replace with your own weights).
- **augmented_data/** and **raw_data/**: Organized dataset folders with
  `images/` and `labels/` following YOLO format.

Getting Started
---------------

Prerequisites
- Python 3.8+ (Linux recommended)
- Install core packages (example):

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, typical dependencies include:
- `numpy`, `opencv-python`, `pillow`, and the training framework you use
  (Ultralytics YOLOv8 or similar).

Dataset & Labels
----------------
- Datasets are stored under `raw_data/` (originals) and
  `augmented_data/` (outputs from `augment_script.py`).
- Labels use the standard YOLO text format: one line per object with
  `<class> <x_center> <y_center> <width> <height>` in normalized coords.
- Pose-specific annotations follow entries referenced in
  `gauge_pose.yaml` (used for configuring pose training runs).

Usage
-----

1. Augment images

```bash
python augment_script.py --input raw_data/train/images --output augmented_data/images
```

Adjust script arguments as needed to control augmentation count and transforms.

2. Prepare dataset for training
- Ensure images and labels are aligned under `images/` and `labels/`.
- Create or update a dataset YAML (example: `raw_data/data.yaml`) pointing
  to train/val/test splits and class names.

3. Train / Fine-tune

Use your preferred training command (example using Ultralytics YOLOv8):

```bash
yolo task=detect mode=train data=raw_data/data.yaml model=yolov8n.pt epochs=50
```

For pose estimation replace `task=detect` with `task=pose` and use
`yolov8n-pose.pt` or similar as the base model.

4. Inference / Prediction

```bash
yolo task=detect mode=predict model=runs/train/weights/best.pt source=images/target.jpg
python reading.py --pred runs/predict/labels/target.txt
```

`reading.py` contains helpers to parse predictions and convert detections
into gauge values or pose visualizations.

Examples & Experiments
----------------------
- Training runs and predictions are stored under `runs/`.
- Inspect `runs/pose/predict/` for pose model outputs and visualization.

Best Practices
--------------
- Keep augmentation parameters realistic to the deployment domain.
- Use a hold-out validation set to avoid overfitting to synthetic data.
- When committing model weights to the repo, ensure they are small or
  provide download instructions for large artifacts.

License & Contact
-----------------
This repository is provided for research and development. Review and
adapt licensing to match your organizational requirements.

For questions or contributions, open an issue or contact the maintainer.
