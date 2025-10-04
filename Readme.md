# AI READI CGM Diabetes Progression Analysis

## Overview
This repository contains code for training models to predict diabetes progression using CGM (Continuous Glucose Monitoring) data and HbA1c features from the AI READI dataset.

---

## CGM Model Training

Run **`CGM_model_training.ipynb`** to train all models based on CGM statistics.  
The implemented models include:

- **Random Forest** — trained on six hand-crafted CGM features  
- **CNN (1-week model)** — trained on one week of CGM data  
- **CNN (1-day model)** — trained on one day of CGM data, tested on each day, and evaluated by maximum probability across days  
- **Transformer model** — uses CNN-generated daily embeddings combined via a Transformer encoder  
- **XGBoost** — applied to PCA-reduced time series data  
- **TabPFN** — trained on PCA-reduced time series data

---

## HbA1c Model Training

Run **`HbA1c_model_training.ipynb`** to train a **Random Forest** model using HbA1c-derived features.

---

## Data Extraction from AI READI

To extract and preprocess data from the AI READI dataset:

1. Run **`create_df_cgm_data.ipynb`** to generate  
   `dataframe_with_glucose_info.pkl` — contains relevant CGM data

2. Run **`create_HbA1c_data.ipynb`** to generate  
   `HbA1c_all_patients.csv` — contains HbA1c data for all patients

> **Note:** You must have the **AI READI dataset** downloaded locally to execute these scripts.
