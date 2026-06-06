# Disaster-Damage-Assessment

An end-to-end deep learning system for **building-level disaster damage assessment** using satellite imagery.  
The project combines **building segmentation**, **damage classification**, and an **interactive Streamlit demo** to visualize damage severity at the building level.


## Problem Overview

Post-disaster damage assessment from satellite images is critical for rapid response and recovery planning.  
This project aims to:

- Detect **buildings** from satellite imagery
- Classify **damage severity** at the building level
- Visualize results using an **interpretable overlay**
- Provide a clean, modular, and reproducible ML pipeline

Damage classes:
- **No Damage**
- **Minor Damage**
- **Major Damage**
- **Destroyed**


## Dataset

This project uses the [**xBD dataset**](https://arxiv.org/pdf/1911.09296), a large-scale satellite imagery dataset
for building damage assessment.

## System Overview

This project implements an end-to-end disaster damage assessment system that operates at the building level using pre- and post-disaster satellite imagery.
The system is designed as a modular pipeline, where each stage has a clearly defined responsibility and can be trained, evaluated, or replaced independently.


## Models

### Building Segmentation
- **Architecture:** SegFormer
- **Task:** Binary building segmentation
- **Output:** Pixel-level building masks
- **Dataset:** xBD dataset
- **Checkpoint:** [`Building_SegFormer_14_epochs.pth`](models)

### Damage Classification
- **Architecture:** CNN-based classifier
- **Input:** Cropped building images
- **Output:** Damage severity class
- **Resolution:** 96×96
- **Checkpoint:** [`XBD_Building_20_epochs_83.30_96x.pth`](models)


## Pipeline
### Key responsibilities:

- Run building segmentation

- Extract individual buildings using connected components

- Classify damage per building

- Generate mask-based colored overlays

- Produce summary statistics

The pipeline is inference-only and completely independent of training and UI code.

## Streamlit Demo

An interactive Streamlit application is provided in app.py.

### Features:

- Upload pre-disaster and post-disaster images

- Run the full damage assessment pipeline

- Visualize:

  - Original post-disaster image

  - Damage overlay (color-coded by severity)

- View per-image damage summary

### Run the app:
```
streamlit run app.py
```
## Training Code

For both segmentation and classification, the repository provides:

- **Python training scripts** for reproducibility

- **Jupyter notebooks** used during the original experimentation and model development

## Evaluation Highlights

### Building Segmentation

Metric: Intersection over Union (IoU)

IoU: 0.6604

### Damage Classification

| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| destroyed       | 0.74      | 0.79   | 0.76     | 2,908   |
| major_damage    | 0.48      | 0.64   | 0.55     | 3,067   |
| minor_damage    | 0.71      | 0.48   | 0.57     | 3,847   |
| no_damage       | 0.91      | 0.91   | 0.91     | 25,114  |
| **Accuracy**    |           |        | **0.83** | 34,936  |
| **Macro Avg**   | 0.71      | 0.70   | 0.70     | 34,936  |
| **Weighted Avg**| 0.84      | 0.83   | 0.83     | 34,936  |

