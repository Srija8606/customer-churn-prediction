# customer-churn-prediction-system
## Overview

This project builds a classification model to predict customer churn in a telecom dataset. The objective is to help businesses identify customers likely to leave and enable proactive retention strategies.

## Methodology

 - Data preprocessing using ColumnTransformer
 - OneHotEncoding for categorical features
 - Handling class imbalance using class weights
 - Logistic Regression baseline model
 - Threshold tuning to optimize recall for churn class

## Results

 - ROC–AUC: ~0.77 (example)
 - Improved recall for churn class through threshold adjustment
 - Demonstrated precision–recall tradeoff analysis
 - 
## Key Concepts Demonstrated

 - Class imbalance handling
 - Decision threshold tuning
 - Business-driven model evaluation
 - ML pipelines
