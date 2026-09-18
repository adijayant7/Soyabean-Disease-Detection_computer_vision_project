# Soyabean Disease Detection using Computer Vision

**Student:** ADITYA JAYANT  
**Registration Number:** 24BAI10579  
**Course:** Computer Vision (CSE3010)  
**University:** VIT Bhopal University  
**Faculty:** Dr. AMRITA PARASHAR  
**Slot:** A21+A22  
**Submission Date:** September 2026

---

A PyTorch-based computer vision project for classifying soyabean leaf diseases using transfer learning with EfficientNet-B0.

## Project Overview

This project develops an automated system to classify soyabean leaf images into five disease categories:
- **Bacterial Blight**
- **Cercospora Leaf Blight**
- **Healthy**
- **Rust**
- **Sudden Death Syndrome (SDS)**

The system uses a pretrained EfficientNet-B0 model fine-tuned on stratified train/validation/test splits with image augmentation and achieves **97.18% test accuracy**.

## Features

- ✅ Automatic duplicate detection and removal
- ✅ Stratified dataset splitting (70% train, 15% validation, 15% test)
- ✅ Image augmentation with rotation, flipping, color jittering, and resizing
- ✅ GPU-accelerated training with PyTorch
- ✅ Pretrained EfficientNet-B0 transfer learning
- ✅ Confusion matrix and misclassified sample visualization
- ✅ Reproducible results with fixed random seeds
- ✅ Comprehensive testing and evaluation scripts

## Project Structure

```
Soyabean-Disease-Detection/
├── data/
│   ├── processed/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── src/
│   ├── train.py                 # Main training script
│   ├── model.py                 # EfficientNet-B0 model definition
│   ├── dataset.py               # PyTorch Dataset class
│   ├── dataloader.py            # DataLoader creation
│   ├── preprocessing.py         # Image preprocessing
│   ├── evaluate_model.py        # Test evaluation
│   ├── visualize_results.py     # Confusion matrix and error visualization
│   ├── split_dataset.py         # Train/val/test splitting
│   ├── check_duplicates.py      # Duplicate detection
│   └── test_*.py                # Unit tests
├── models/
│   └── best_model.pth           # Saved best model checkpoint
├── results/
│   ├── confusion_matrix.png     # Test confusion matrix
│   └── misclassified_images.png # Misclassified sample visualization
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional but recommended) NVIDIA GPU with CUDA support for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/adijayant7/Soyabean-Disease-Detection_computer_vision.git
cd Soyabean-Disease-Detection_computer_vision
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- **torch** - Deep learning framework
- **torchvision** - Computer vision utilities
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning utilities
- **matplotlib** - Data visualization
- **pillow** - Image processing

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Dataset Preparation

Your dataset should be organized as follows:

```
data/
└── raw/
    ├── Bacterial_Blight/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Cercospora_Leaf_Blight/
    │   └── ...
    ├── Healthy/
    │   └── ...
    ├── Rust/
    │   └── ...
    └── Sudden_Death_Syndrome/
        └── ...
```

### 1. Detect and Remove Duplicates

```bash
python src/check_duplicates.py
```

This script identifies and removes duplicate images from the dataset.

### 2. Create Train/Validation/Test Splits

```bash
python src/split_dataset.py
```

Generates stratified CSV files:
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

### 3. Train the Model

```bash
python src/train.py
```

**Training Parameters:**
- Epochs: 20
- Batch Size: 16
- Learning Rate: 0.0001
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Input Size: 224×224
- Model: EfficientNet-B0 (pretrained)

**Output:**
- Saves best model checkpoint to `models/best_model.pth`
- Displays training and validation metrics
- Automatically uses GPU if CUDA is available

**Expected Training Time:**
- GPU (RTX 5060): ~3-5 minutes
- CPU: ~15-30 minutes

### 4. Evaluate on Test Set

```bash
python src/evaluate_model.py
```

**Output:**
- Prints test accuracy and total correct predictions
- Generates `results/confusion_matrix.png`
- Generates `results/misclassified_images.png`

### 5. Visualize Results

```bash
python src/visualize_results.py
```

Displays confusion matrix and misclassified samples with predicted classes and confidence scores.

## Running Tests

Verify the installation with the included test scripts:

```bash
# Test dataset loading
python src/test_dataset.py

# Test DataLoader
python src/test_dataloader.py

# Test transformations
python src/test_transforms.py

# Test model forward pass (GPU if available)
python src/test_gpu_model.py
```

## Model Architecture

**Base Model:** EfficientNet-B0 (pretrained on ImageNet)

**Architecture Details:**
- Input: 224×224 RGB images
- Pretrained backbone: EfficientNet_B0_Weights.DEFAULT
- Custom classifier: 5-class linear layer
- Output: Class probabilities (softmax)

**Why EfficientNet-B0?**
- Excellent accuracy-to-efficiency trade-off
- Lower memory requirements than larger models
- Suitable for resource-constrained environments
- Strong transfer learning performance

## Data Augmentation

**Training Augmentation:**
- RandomResizedCrop(224, scale=(0.7, 1.0))
- RandomHorizontalFlip(p=0.5)
- RandomRotation(degrees=15)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- Normalization (ImageNet mean/std)

**Validation/Test Preprocessing:**
- Resize(256)
- CenterCrop(224)
- Normalization (ImageNet mean/std)

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 97.18% |
| Correct Predictions | 69/71 |
| Model Checkpoint | models/best_model.pth |

### Per-Class Performance (from Confusion Matrix)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Bacterial Blight | 15/15 | 100% | 100% |
| Cercospora Leaf Blight | 12/12 | 100% | 100% |
| Healthy | 12/14 | 86% | 89% |
| Rust | 14/14 | 100% | 100% |
| SDS | 16/16 | 100% | 100% |

### Visualizations

- **Confusion Matrix**: `results/confusion_matrix.png`
- **Misclassified Samples**: `results/misclassified_images.png`

## System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB
- Python 3.8+

### Recommended (for GPU Training)
- GPU: NVIDIA GPU with 4+ GB VRAM
- CUDA 11.8+ (for PyTorch)
- cuDNN 8.0+

### Hardware Used in Development
- GPU: NVIDIA GeForce RTX 5060 Laptop (8 GB VRAM)
- CPU: Intel Core i7
- RAM: 16 GB

## Troubleshooting

### "CUDA out of memory" error
**Solution:** Reduce batch size in `src/train.py`:
```python
batch_size = 8  # or 4 for larger models
```

### "ModuleNotFoundError: No module named 'torch'"
**Solution:** Reinstall dependencies:
```bash
pip install --force-reinstall -r requirements.txt
```

### Dataset path errors
**Solution:** Ensure your raw images are in `data/raw/ClassName/` structure and run:
```bash
python src/split_dataset.py
```

### Model not loading
**Solution:** Delete `models/best_model.pth` and retrain:
```bash
python src/train.py
```

## Dependencies

All dependencies are listed in `requirements.txt`. Key packages:

- **torch** (≥2.0.0): Deep learning framework
- **torchvision** (≥0.15.0): Computer vision utilities
- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical computing
- **scikit-learn**: Stratified splitting and metrics
- **matplotlib**: Visualization
- **pillow**: Image loading and processing

See `requirements.txt` for exact versions.

## Project Workflow

```
Raw Images
    ↓
Duplicate Detection & Removal
    ↓
Stratified Train/Val/Test Split (70/15/15)
    ↓
Data Augmentation & Preprocessing
    ↓
PyTorch DataLoader (batch_size=16)
    ↓
Pretrained EfficientNet-B0
    ↓
5-Class Classifier
    ↓
Training (20 epochs, AdamW, CrossEntropyLoss)
    ↓
Validation (save best checkpoint)
    ↓
Test Evaluation
    ↓
Confusion Matrix & Error Analysis
```

## Key Features Explained

### Stratified Splitting
Preserves class distribution across train/validation/test sets using `sklearn.model_selection.train_test_split` with `stratify=True`.

### Transfer Learning
Uses weights pretrained on ImageNet to accelerate convergence and improve accuracy with limited data.

### Validation-Based Checkpointing
Saves the model with the best validation accuracy, preventing overfitting to the training set.

### Reproducibility
Fixed random seeds (`random_state=42`, `torch.manual_seed()`) ensure consistent results across runs.

## Citations & References

1. Tan, M. & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." https://arxiv.org/abs/1905.11946

2. PyTorch Documentation. https://pytorch.org/docs/

3. Torchvision Documentation. https://pytorch.org/vision/stable/

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." https://arxiv.org/abs/1512.03385

## Future Enhancements

- [ ] Collect more field-based images
- [ ] Test with stronger architectures (ResNet-50, Vision Transformer)
- [ ] Hyperparameter tuning (learning rate schedules, augmentation strength)
- [ ] Grad-CAM visualizations for model interpretability
- [ ] Streamlit web interface for inference
- [ ] Mobile application deployment
- [ ] Confidence thresholds for uncertain predictions
- [ ] External dataset validation

## License

This project is provided for educational purposes as part of the VIT Bhopal University Computer Vision course.

## Support & Contact

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Last Updated:** September 2026

**Repository:** https://github.com/adijayant7/Soyabean-Disease-Detection_computer_vision
