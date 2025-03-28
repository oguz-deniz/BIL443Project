# Mammographic Image Classification with Dual-Classifier Architecture

This repository contains code and notebooks for a project on binary classification of mammographic images into benign and malignant cases. The project leverages a dual-classifier architecture with a Weighted Double Constraint (WDC) mechanism and employs extensive data preprocessing to optimize image quality for classification. 

## Project Overview

The project aims to improve the accuracy of mammographic image classification by:

- **Data Preprocessing:** Converting raw DICOM images to PNG, cropping the breast region, and enhancing image contrast using CLAHE.
- **Double Fusion Strategy:** Fusing features extracted from two mammographic views per patient to improve diagnostic performance.
- **Model Training:** Evaluating multiple backbone configurations (EfficientNet, ResNet, DenseNet) with two types of classifier heads:
  - **Normal Classifier:** Uses standard binary cross-entropy (BCE) loss.
  - **Weighted Double Constraint (WDC) Classifier:** Incorporates a dual-classifier mechanism with an adaptive loss function to enhance performance on challenging samples.

## Prerequisites

- **Python 3.7 or later**
- **Pip** for package management

### Required Python Packages

- `pydicom`
- `opencv-python`
- `numpy`
- `torch` (or your preferred deep learning framework; these notebooks assume PyTorch)
- `torchvision`
- `albumentations` (for image augmentations)
- `matplotlib` (for visualization)

Install dependencies with:

```bash
pip install -r requirements.txt
