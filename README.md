# EMNIST Handwritten Letter Classification using PyTorch

## Overview
This project implements and compares two deep learning models— a **Multilayer Perceptron (MLP)** and an **Enhanced Convolutional Neural Network (CNN)**—for handwritten letter recognition using the **EMNIST Letters** dataset. The goal is to evaluate how a fully connected network performs against a CNN on image-based classification and to analyze training behavior, accuracy, and confusion matrices.

The dataset is preprocessed with data augmentation, normalized grayscale inputs, and a controlled subset strategy to reduce training size while preserving class diversity.

---

## Dataset Handling
- Dataset used: **EMNIST (Letters split)**
- Images are **28×28 grayscale**
- Labels are adjusted from **1–26 → 0–25**
- Both train and test splits are merged, and **30% of the total data** is sampled
- Sampled data is further split into:
  - 90% training
  - 10% testing

---

## Data Augmentation & Preprocessing
### Training Transformations
- Grayscale conversion
- Random rotation (±10°)
- Random affine translation (±10%)
- Tensor conversion
- Normalization with mean 0.5 and std 0.5

### Testing Transformations
- Grayscale conversion
- Tensor conversion
- Normalization only (no augmentation)

---

## Models Implemented

### 1. Multilayer Perceptron (MLP)
A baseline non-convolutional model that treats each image as a flat vector.

**Architecture**
- Input: 784 (28×28 flattened)
- Fully connected layers:
  - 784 → 512 → 128 → 27
- ReLU activations
- Dropout (0.3)
- Output layer uses raw logits for CrossEntropyLoss

**Purpose**
- Serves as a baseline to highlight the limitations of non-spatial models on image data

---

### 2. Enhanced Convolutional Neural Network (CNN)
A deeper model designed to capture spatial features in handwritten characters.

**Architecture**
- Convolution Block 1:
  - Conv2D (1 → 32)
  - Batch Normalization
  - ReLU
  - Max Pooling
- Convolution Block 2:
  - Conv2D (32 → 64)
  - Batch Normalization
  - ReLU
  - Max Pooling
- Fully Connected Layers:
  - 64×7×7 → 128 → 27
- Dropout (0.4)

**Why this works better**
- Learns local stroke patterns
- Translation-invariant feature extraction
- Reduced sensitivity to noise and rotation

---

## Training Strategy
- Loss function: **CrossEntropyLoss**
- Optimizer: **Adam**
- Learning rate: `1e-3`
- Learning rate scheduler:
  - StepLR (step size = 5, gamma = 0.5)
- Epochs: **10**
- Batch size: **64**
- GPU used automatically if available

---

## Evaluation Metrics
For both models, the following are computed:
- Classification accuracy
- Precision, recall, and F1-score per class
- Confusion matrix visualization
- Training loss per epoch

Results are compared visually using:
- Loss vs Epoch plots
- Accuracy bar chart

---

## Visualization Outputs
- Confusion matrix heatmaps for each model
- Training loss comparison between MLP and CNN
- Accuracy comparison bar chart

These plots help analyze:
- Convergence behavior
- Misclassification patterns
- Performance gap between architectures

---

## Model Persistence
The trained CNN model weights are saved to disk:

```
enhanced_emnist_cnn_best.pth
```

This file can be loaded later for inference or fine-tuning without retraining.

---

## Key Takeaways
- CNN significantly outperforms MLP on handwritten character recognition
- Spatial feature extraction is critical for image-based tasks
- Data augmentation improves generalization
- Learning rate scheduling stabilizes training

---

## Technologies Used
- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
