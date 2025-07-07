# T-spline 3D CNN

**Bachelor Thesis Project**  
Exploring the Applicability of T-splines in 3D Convolutional Neural Networks  
_ITMO University, 2025_

---

## 🚀 Overview

This project investigates an alternative representation of 3D data for deep learning — using T-spline-like surfaces (via Rhino3D SubD) to generate compact, adaptive point clouds as input for neural networks (PointNet).

The goal:
- Reduce input size and redundancy compared to voxels/meshes
- Test the applicability of CAD-origin geometries in 3D classification tasks
- Explore the effect of geometric flexibility on network performance

---

## 🏗️ Pipeline

1. **3D Model Preprocessing**  
   - Convert mesh to SubD surface (Rhino3D, manual/automated)
   - Uniformly sample point cloud from SubD surface (`preprocessing/point_sampler.py`)

2. **Data Preparation**  
   - Aggregate sampled point clouds, assign class labels, split into train/test
   - Store datasets as `.npz` files (with `points` and `labels` arrays)
   - Caching enabled for fast loading

3. **Model Training**  
   - Load data with caching and optional augmentation (`data/loader.py`)
   - Use PointNet with input and feature transformation (`pointnet/model.py`)
   - Train, validate and save best model (`pointnet/train.py`)

4. **Evaluation & Visualization**  
   - Evaluate classification accuracy and loss
   - Visualize training curves and point clouds (`utils/visualizer.py`)

---

## 🗂️ Project Structure
```
T-spline-3D-CNN/
├── pointnet/
│   ├── model.py           # PointNet and TNet architectures
│   └── train.py           # Training loop and validation
│
├── preprocessing/
│   └── point_sampler.py   # Mesh-to-point-cloud sampling utilities
│
├── data/
│   └── loader.py          # Cached dataset loader and torch DataLoader
│
├── utils/
│   ├── metrics.py         # Accuracy, feature transform regularization
│   └── visualizer.py      # Plotting loss curves and 3D point clouds
│
├── main.py                # Entry point for training
└── README.md              # This file
```
---

## ⚙️ Getting Started

1. **Install dependencies**
   pip install -r requirements.txt

2. **Prepare 3D data**
   - Use Rhino3D to convert meshes to SubD and export as `.obj`
   - Sample point clouds:  
     from preprocessing.point_sampler import batch_sample_from_dir
     batch_sample_from_dir('input_meshes', 'sampled_points', num_points=1024, ext='.obj')
   - Create `.npz` datasets (`points`, `labels`) and put in `data/`

3. **Train the model**
   python main.py

4. **Visualize results**
   - Use functions from `utils/visualizer.py` for point clouds and loss curves

---

## 💡 Credits

- Model: [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593)
- Geometry: Rhino3D, Trimesh
- Author: Arkadiy Ri, ITMO University

---
