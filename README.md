# Fetal Health Classification App

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

| Model               | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
| ------------------- | -------- | ---- | --------- | ------ | ---- | ---- |
| Logistic Regression | 0.88     | 0.96 | 0.89      | 0.88   | 0.89 | 0.68 |
| Decision Tree       | 0.90     | 0.85 | 0.89      | 0.90   | 0.89 | 0.71 |
| KNN                 | 0.87     | 0.94 | 0.86      | 0.87   | 0.86 | 0.62 |
| Naive Bayes         | 0.81     | 0.88 | 0.86      | 0.81   | 0.83 | 0.57 |
| Random Forest       | 0.93     | 0.98 | 0.93      | 0.93   | 0.93 | 0.81 |

## Observations

- Logistic Regression: Strong baseline with high AUC, showing good class separation capability. Performance is stable but slightly lower MCC indicates moderate class-wise balance.
- Decision Tree: Achieves good accuracy but lower AUC suggests weaker generalization. Slight risk of overfitting due to its nature.
- KNN: Delivers consistent mid-range performance across all metrics. Highly dependent on feature scaling and data distribution.
- Naive Bayes: Fast and simple model with decent AUC, but lower accuracy and MCC show limited predictive strength. Assumption of feature independence impacts performance.
- Random Forest: Best overall model with highest accuracy, AUC, and MCC. Provides strong generalization and handles feature interactions effectively.

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