This repository contains a complete pipeline for training, evaluating, and applying a deep learning U-Net model to classify mesoscale cellular convection (MCC) patterns in radar or satellite data.

Overview
The repository is organized into four key steps:

(1) Train the U-Net Model:
Script: How_to_train_Deep_Learning_Model.py
This script demonstrates the U-Net architecture.

(2) Evaluate Model Performance:
Notebook: Deep_Learning_Performance.ipynb
Visualizes and evaluates the model's performance (e.g.,confusion matrices).
Also demonstrates how to load the trained model.

(3)Use the Trained Model for Classification:
Script: How_to_use_Deep_Learning_Model.py
Demonstrates how to load the trained model and apply it to new ARM data for MCC classification.

(4) Post-Processing Predicted Results:
Script: Post_Processing.py
Includes necessary post-processing steps to refine the deep learning predictions, such as estimate the duration of MCC.
