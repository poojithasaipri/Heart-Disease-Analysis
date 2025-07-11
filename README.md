# Heart Disease Prediction using Machine Learning

This project aims to predict the likelihood of heart disease using multiple machine learning models trained on the StatLog (Heart) dataset. It is designed to support early diagnosis and reduce clinical risk using data-driven insights.

## 🔍 Overview

- Built and evaluated 4 ML models:
  - Logistic Regression
  - Decision Tree
  - Naïve Bayes
  - K-Nearest Neighbors (KNN)
- Achieved up to **86.9% accuracy** using the Naive Bayes classifier.
- Focused on reducing **False Negative Rate (FNR)** for early detection.

## 📊 Dataset

- **Name:** StatLog (Heart) Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(heart))
- **Features:** 13 attributes related to age, sex, blood pressure, cholesterol, chest pain, etc.
- **Target:** Presence (1) or absence (0) of heart disease.

## Models Used

| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | 83.6%    |
| Decision Tree       | 81.9%    |
| KNN (k=7)        | 82.5%    |
| Naïve Bayes          | **86.9%** |

## 🧪 Tech Stack

- **Python 3.9**
- **Libraries:**
  - pandas, numpy, seaborn, matplotlib
  - scikit-learn

## 📈 Visualizations

- Correlation heatmaps
- Feature distributions
- Confusion matrices
- ROC curves

##  Features

- Preprocessing using:
  - KNN Imputation for missing values
  - Label encoding
- Model evaluation via:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
- Visualization for interpretability

