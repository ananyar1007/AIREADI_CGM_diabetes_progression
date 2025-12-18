# AI-READI CGM Diabetes Progression

AI-READI CGM Diabetes Progression investigates whether **CGM-derived glycemic dynamics** differ between individuals with **prediabetes or lifestyle-controlled diabetes** and those with **oral-medication-managed type 2 diabetes**, and whether CGM data provides **discriminative information beyond standard clinical measures such as HbA1c**.  

This repository contains code for preprocessing CGM time-series data, extracting features, and training classical machine learning and deep learning models to distinguish between these study groups.

## Key Results

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/cc2982d9-d65b-41d3-89ce-69fa0ed2e3f9"
    width="700"
    alt="Balanced accuracy comparison across models"
  >
</p>

<p align="center">
  <em>
    Balanced accuracy (BA) comparison across different modeling approaches on the AI-READI test set.
  </em>
</p>



## Overview

The goal of this project is to study how patterns in **continuous glucose monitoring (CGM) time-series data** reflect different stages of diabetes progression. We focus on distinguishing between:

- **Study Group 1**: Prediabetes and lifestyle-controlled diabetes  
- **Study Group 2**: Oral-medication-managed type 2 diabetes  

using CGM data from the **AI-READI dataset**, collected with the Dexcom G6 sensor.

In addition to internal validation on AI-READI, we evaluate generalization using the **CGMacros dataset**, which contains CGM recordings from two different sensors worn simultaneously (Dexcom and FreeStyle Libre). This allows us to assess robustness across datasets and sensor types.

## Repository Structure

- `calculate_cgm_features.py`  
  Script for extracting handcrafted features from CGM time-series data

- `cgm_dataset_transform.py`  
  Utilities for transforming and organizing CGM data for model training

- `model.py`  
  Model definitions and helper functions

- `train_model.py`  
  Training scripts for classical ML and deep learning models

- `test_model.py`  
  Model evaluation and testing utilities

- `create_df_cgm_data.ipynb`  
  Notebook for preprocessing raw CGM data into structured datasets

- `create_HbA1c_data.ipynb`  
  Notebook for preprocessing HbA1c clinical data

- `CGM_model_training.ipynb`  
  Notebook for training and evaluating CGM-based models

- `HbA1c_model_training.ipynb`  
  Notebook for training HbA1c-only baseline models

- `Read_CGM_results.ipynb`  
  Notebook for analyzing and visualizing model outputs

## Getting Started

### Requirements
- Python 3.8 or higher  
- NumPy, pandas, scikit-l
