# RPS CNN Battle

Rock-Paper-Scissors image classification using TensorFlow/Keras and the official TensorFlow Dataset [`rock_paper_scissors`](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors). Built as a collaborative deep learning university project by a team of 4.

## Project Overview

The project explores the full computer vision pipeline: dataset loading, preprocessing, CNN architecture experimentation, evaluation, and a live webcam demo app.

**Main notebook:** `notebooks/rps_cnn_battels.ipynb`  
**Live app:** `app/live_webcam_predictor.py`

---

## Class Mapping

The TFDS `rock_paper_scissors` dataset assigns labels in alphabetical order:

| Index | Class |
|-------|-------|
| 0 | rock |
| 1 | paper |
| 2 | scissors |

This mapping is used consistently in the notebook, CSVs, and the Streamlit application.

---

## Experiments

All five models were trained on 80% of the TFDS training split and evaluated on the official held-out test set (372 images). Input size: 128×128×3, optimizer: Adam.

| Experiment | Architecture | Test Acc | Test Loss | Params |
|---|---|---|---|---|
| Baseline CNN | 3 Conv blocks (32-64-128) + GAP | 84.95% | 0.6262 | 101,699 |
| Simpler CNN | 2 Conv blocks (32-64) + GAP | 69.09% | 1.2696 | 23,747 |
| Deeper CNN | 4 Conv blocks (32-64-128-256) + GAP | 97.31% | 0.2262 | 405,059 |
| **Regularized CNN** *(final)* | Deeper CNN + Dropout(0.5) | **96.24%** | 0.0734 | 405,059 |
| MobileNetV2 (frozen) | MobileNetV2 backbone + GAP + Dropout | 88.17% | 0.3279 | — |
| MobileNetV2 (fine-tuned) | Unfroze last 30 layers, lr=1e-5 | 90.32% | 0.2825 | — |

> **Note:** Metrics for the baseline–regularized CNN rows are from the original notebook training run. The saved `models/rps_cnn.keras` was produced by a fresh retrain using the same architecture and hyperparameters, achieving **98.66%** test accuracy.

---

## Final Model

**Architecture:** Regularized CNN (4 Conv blocks + Dropout(0.5))  
**Saved to:** `models/rps_cnn.keras`  
**Verified test accuracy (current saved model):** 98.66%

---

## Repository Structure

```text
RPS-Project/
├── app/
│   └── live_webcam_predictor.py  # Streamlit webcam app
├── models/
│   └── rps_cnn.keras             # Saved final model
├── notebooks/
│   ├── rps_cnn_battels.ipynb     # Main experiment notebook
│   ├── master_experiment_table.csv
│   ├── final_model_comparison.csv
│   ├── model_vs_test_accuracy.png
│   └── train_val_test_accuracy.png
├── requirements.txt
└── README.md
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib pandas scikit-learn \
            streamlit streamlit-webrtc av opencv-python Pillow
```

### Run the notebook

Open and run all cells top-to-bottom:

```bash
jupyter notebook notebooks/rps_cnn_battels.ipynb
```

> The notebook loads the dataset, trains all models, and saves `models/rps_cnn.keras`.

### Run the Streamlit app

```bash
streamlit run app/live_webcam_predictor.py
```

The app requires `models/rps_cnn.keras` to exist. Run the notebook first to produce this file, or it will show a warning with instructions.

---

## Key Findings

- The Deeper CNN (4 convolutional blocks) substantially improved test accuracy from 84.95% to 97.31% compared with the baseline.
- Adding Dropout(0.5) in the Regularized CNN reduced the generalization gap while maintaining high test accuracy.
- MobileNetV2 transfer learning achieved 90.32% test accuracy after fine-tuning — strong but below our custom CNN.
- The Regularized CNN was selected as the final model for deployment in the live Streamlit demo.
