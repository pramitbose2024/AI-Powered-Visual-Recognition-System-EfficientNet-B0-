# 🧠 AI-Powered Visual Recognition System (EfficientNet-B0)

A deep learning–based visual recognition system designed to accurately classify images across **53 distinct categories** using transfer learning with EfficientNet-B0. This project demonstrates an end-to-end **computer vision pipeline**, from data preprocessing to model interpretability.

---

## 🚀 Overview

This project implements a **scalable image classification system** leveraging pretrained EfficientNet-B0 to achieve high accuracy and strong generalization on a multi-class dataset.

Beyond standard classification, the system focuses on:

* **robust feature extraction using transfer learning**
* **efficient data pipelines for large-scale training**
* **interpretability through probabilistic predictions**

---

## ✨ Key Highlights

* **High-Accuracy Multi-Class Classification**
  Achieved **94% validation accuracy** and **92% test accuracy** across 53 classes.

* **Transfer Learning with EfficientNet-B0**
  Leveraged pretrained weights to capture rich visual features and improve convergence.

* **Optimized Data Pipeline**
  Built custom PyTorch `Dataset` and `DataLoader` for efficient batching and preprocessing.

* **Improved Generalization**
  Applied image resizing and structured data handling to enhance model robustness.

* **Model Interpretability**
  Implemented probability-based predictions and visualizations to analyze misclassifications and model confidence.

---

## 🧠 Model Architecture

| Component                    | Description                                                          |
| ---------------------------- | -------------------------------------------------------------------- |
| **EfficientNet-B0 Backbone** | Pretrained feature extractor for high-quality visual representations |
| **Feature Extraction Layer** | Removes final classification layer for transfer learning             |
| **Fully Connected Layer**    | Maps extracted features to 53 output classes                         |

---

## ⚙️ Training Setup

* **Framework:** PyTorch + `timm`
* **Optimizer:** Adam
* **Loss Function:** CrossEntropyLoss
* **Input Size:** 128×128
* **Batch Size:** 32

---

## 📊 Dataset

This project uses the **Cards Image Dataset for Classification**, a high-quality dataset designed for fine-grained visual recognition tasks.

### 📌 About the Dataset:

* All images are **224 × 224 × 3** in **JPG format**
* Each image is **cropped to contain a single playing card**
* The card occupies **more than 50% of the image area**, ensuring clarity and focus
* The dataset contains:

  * **7,624 training images**
  * **265 validation images**
  * **265 test images**
* The dataset is organized into **53 subdirectories**, each representing a unique card class
* Includes a **CSV file** for alternative dataset loading and analysis

### 🧠 Structure:

* Train / Validation / Test splits
* Folder-based labeling (compatible with PyTorch `ImageFolder`)
* Designed for **multi-class image classification**

---

## 📈 Results

* **Validation Accuracy:** 94%
* **Test Accuracy:** 92%
* **Training Loss:** 0.21
* **Validation Loss:** 0.13

These results demonstrate strong **generalization and model stability**.

---

## 🔍 Model Evaluation & Interpretability

* Generated **class probability distributions** for each prediction
* Visualized outputs using bar charts for **confidence analysis**
* Identified misclassification patterns to guide further improvements

---

## 🧰 Tech Stack

* Python
* PyTorch
* torchvision
* timm (EfficientNet)
* NumPy
* Matplotlib

---

## 🌍 Real-World Applications

This system reflects core components used in:

* **Retail & Inventory Recognition Systems**
* **Automated Visual Inspection**
* **Smart Surveillance & Object Recognition**
* **Mobile Vision Applications**

---

## 🔬 Future Improvements

* Upgrade to **EfficientNet-B3 / Vision Transformers (ViT)**
* Implement **advanced augmentation (MixUp, CutMix)**
* Deploy as a **real-time inference API or web app**
* Integrate **Grad-CAM for deeper model explainability**

---

## 🧠 Why This Project Matters

This project demonstrates:

* Strong understanding of **computer vision and transfer learning**
* Ability to build **scalable deep learning pipelines**
* Practical skills in **model evaluation, optimization, and interpretability**

