# Fetal Health Classification App

## NOTE
The comparison metrics shown in the report were computed on the full dataset during initial experimentation. The deployed Streamlit application evaluates models on test data, reflecting realistic performance

## Problem Statement

To classify fetal health condition (Normal, Suspect, Pathological) using CTG data and compare multiple machine learning models to identify the best performing model.

## Overview
This project predicts fetal health condition using Machine Learning models based on CTG (Cardiotocography) data.

## Dataset
- 2126 records
- 21 features
- Target: fetal_health (Normal, Suspect, Pathological)

## Models Used
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest

## Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|--------|-----|----------|--------|-----|-----|
| Logistic Regression | 0.88 | 0.96 | 0.88 | 0.88 | 0.88 | 0.68 |
| Decision Tree | 0.91 | 0.86 | 0.90 | 0.91 | 0.90 | 0.74 |
| KNN | 0.87 | 0.93 | 0.86 | 0.87 | 0.86 | 0.61 |
| Naive Bayes | 0.80 | 0.87 | 0.86 | 0.80 | 0.82 | 0.57 |
| Random Forest | 0.93 | 0.97 | 0.93 | 0.93 | 0.93 | 0.80 |

## Observations

- Logistic Regression: Good baseline, high AUC but slightly lower MCC
- Decision Tree: Good accuracy but prone to overfitting
- KNN: Moderate performance, sensitive to scaling
- Naive Bayes: Fast but lowest accuracy
- Random Forest: Best performance across all metrics

## Best Model
Random Forest is the overall winner due to highest accuracy and MCC.

## Features
- Upload CSV dataset
- Select model
- View predictions
- Classification report
- Confusion matrix

## Tech Stack
- Python
- Scikit-learn
- Streamlit

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
