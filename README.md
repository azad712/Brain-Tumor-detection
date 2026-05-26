# Brain Tumor Detection with YOLOv8

A comprehensive deep learning solution for detecting and classifying brain tumors using YOLOv8 object detection. This project provides end-to-end tools for training, evaluation, inference, and visualization of brain tumor detection models.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a state-of-the-art brain tumor detection system using YOLOv8, capable of:
- Detecting brain tumors in medical images
- Classifying tumors into 4 categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary**
- Providing bounding box localization with confidence scores
- Generating comprehensive evaluation metrics and visualizations

## 📊 Dataset

The dataset is organized in YOLO format with the following structure:

```
brain tumor with BB/
├── Train/
│   ├── Glioma/
│   │   ├── images/
│   │   └── labels/
│   ├── Meningioma/
│   │   ├── images/
│   │   └── labels/
│   ├── No Tumor/
│   │   ├── images/
│   │   └── labels/
│   └── Pituitary/
│       ├── images/
│       └── labels/
└── Val/
    ├── Glioma/
    ├── Meningioma/
    ├── No Tumor/
    └── Pituitary/
```

### Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | Glioma | Tumor originating from glial cells |
| 1 | Meningioma | Tumor originating from meninges |
| 2 | No Tumor | No tumor detected in brain scan |
| 3 | Pituitary | Tumor originating from pituitary gland |

## ✨ Features

- **Dataset Validation**: Comprehensive validation of dataset structure and annotations
- **Flexible Training**: Customizable training with multiple YOLOv8 variants (nano to xlarge)
- **Real-time Inference**: Fast prediction on images, directories, or videos
- **Comprehensive Evaluation**: Detailed metrics including mAP, precision, recall, and F1-score
- **Rich Visualizations**: Bounding box visualizations with confidence scores
- **Easy Configuration**: YAML-based configuration for easy customization
- **GPU Acceleration**: Full CUDA support for faster training and inference

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "brain tumor with BB"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install PyTorch with CUDA (for GPU support)

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 🏃 Quick Start

### 1. Validate Dataset

Before training, validate your dataset structure and annotations:

```bash
python validate_dataset.py
```

### 2. Train Model

Train a YOLOv8 model on your dataset:

```bash
# Basic training with default parameters
python train.py --model yolov8n.pt --epochs 100 --batch 16

# Advanced training with custom parameters
python train.py --model yolov8m.pt --epochs 150 --batch 8 --imgsz 640 --lr0 0.01
```

### 3. Run Inference

Perform inference on new images:

```bash
# Single image
python predict.py --model runs/train/brain_tumor_detection/weights/best.pt --source path/to/image.jpg

# Directory of images
python predict.py --model runs/train/brain_tumor_detection/weights/best.pt --source path/to/images/

# With custom confidence threshold
python predict.py --model runs/train/brain_tumor_detection/weights/best.pt --source path/to/images/ --conf 0.5
```

### 4. Evaluate Model

Evaluate model performance on validation set:

```bash
python evaluate.py --model runs/train/brain_tumor_detection/weights/best.pt --data data.yaml
```

### 5. Visualize Results

Create visualizations of predictions:

```bash
# Visualize single image
python visualize.py --model runs/train/brain_tumor_detection/weights/best.pt --source path/to/image.jpg

# Create grid visualization
python visualize.py --model runs/train/brain_tumor_detection/weights/best.pt --source path/to/images/ --grid --grid_cols 4
```

## 📖 Usage

### Training Options

```bash
python train.py --help
```

Key parameters:
- `--model`: YOLOv8 variant (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Input image size (default: 640)
- `--lr0`: Initial learning rate (default: 0.01)
- `--patience`: Early stopping patience (default: 50)
- `--device`: GPU device (e.g., 0, 0,1,2,3, or cpu)

### Model Variants

| Model | Size | Speed | mAP | Parameters |
|-------|------|-------|-----|------------|
| YOLOv8n | Nano | Fastest | Good | 3.2M |
| YOLOv8s | Small | Fast | Better | 11.2M |
| YOLOv8m | Medium | Moderate | Great | 25.9M |
| YOLOv8l | Large | Slow | Excellent | 43.7M |
| YOLOv8x | XLarge | Slowest | Best | 68.2M |

### Inference Options

```bash
python predict.py --help
```

Key parameters:
- `--model`: Path to trained model weights
- `--source`: Image, directory, or video file
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--save`: Save inference results
- `--show`: Display results in real-time

### Evaluation Options

```bash
python evaluate.py --help
```

Key parameters:
- `--model`: Path to trained model weights
- `--data`: Path to data.yaml
- `--split`: Dataset split to evaluate (train/val/test)
- `--batch`: Batch size for evaluation
- `--save_json`: Save results to JSON file

### Visualization Options

```bash
python visualize.py --help
```

Key parameters:
- `--model`: Path to trained model weights
- `--source`: Image or directory
- `--output`: Output directory for visualizations
- `--grid`: Create grid visualization
- `--show_conf`: Show confidence scores
- `--show_labels`: Show class labels

## 📁 Project Structure

```
brain tumor with BB/
├── Train/                      # Training dataset
├── Val/                        # Validation dataset
├── data.yaml                   # Dataset configuration
├── train.py                    # Training script
├── predict.py                  # Inference script
├── evaluate.py                 # Evaluation script
├── visualize.py                # Visualization script
├── validate_dataset.py         # Dataset validation script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── runs/                       # Training and inference results
    ├── train/                  # Training outputs
    │   └── brain_tumor_detection/
    │       ├── weights/        # Model weights
    │       │   ├── best.pt     # Best model
    │       │   └── last.pt     # Last checkpoint
    │       ├── results.png     # Training curves
    │       └── confusion_matrix.png
    ├── predict/                # Prediction outputs
    └── evaluate/               # Evaluation outputs
```

## 📈 Results

After training, you'll find the following outputs:

### Training Results
- **Model weights**: `runs/train/brain_tumor_detection/weights/best.pt`
- **Training curves**: Loss, mAP, precision, recall plots
- **Confusion matrix**: Class-wise performance visualization
- **Validation predictions**: Sample predictions on validation set

### Evaluation Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Example Results

```
EVALUATION METRICS SUMMARY
======================================================================

OVERALL METRICS:
----------------------------------------------------------------------
Precision (Box)          : 0.9234
Recall (Box)             : 0.8876
mAP@0.5                  : 0.9156
mAP@0.5:0.95             : 0.7823

PER-CLASS METRICS:
----------------------------------------------------------------------
Class                Precision    Recall       mAP@0.5      mAP@0.5:0.95
----------------------------------------------------------------------
Glioma               0.9345       0.9012       0.9234       0.7956
Meningioma           0.9123       0.8745       0.9078       0.7690
No Tumor             0.9456       0.9123       0.9345       0.8012
Pituitary            0.9012       0.8623       0.8967       0.7634
======================================================================
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch 8` or `--batch 4`
   - Use smaller model: `--model yolov8n.pt`
   - Reduce image size: `--imgsz 416`

2. **Slow Training**
   - Enable GPU: Check CUDA installation
   - Use mixed precision training (enabled by default)
   - Increase batch size if GPU memory allows

3. **Poor Performance**
   - Train for more epochs: `--epochs 200`
   - Use larger model: `--model yolov8m.pt`
   - Adjust learning rate: `--lr0 0.001`
   - Enable data augmentation (enabled by default)

4. **Dataset Validation Errors**
   - Check image and label file names match
   - Verify label format: `class_id x_center y_center width height`
   - Ensure coordinates are normalized (0-1)

## 💡 Tips for Best Results

1. **Data Quality**: Ensure high-quality, diverse training images
2. **Data Augmentation**: Use default augmentation settings for better generalization
3. **Model Selection**: Start with YOLOv8n for quick experiments, use YOLOv8m/l for production
4. **Training Duration**: Train for at least 100 epochs with early stopping
5. **Hyperparameter Tuning**: Experiment with learning rate and batch size
6. **Validation**: Regularly check validation metrics during training
7. **Ensemble**: Combine predictions from multiple models for better accuracy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Brain tumor dataset contributors
- PyTorch and OpenCV communities

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is for research and educational purposes. Always consult with medical professionals for actual diagnosis and treatment.
```

## 🔗 Useful Links

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

Made with ❤️ for medical AI research