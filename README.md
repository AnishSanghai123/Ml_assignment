#  Machine Learning Classification Project

## 1️ Problem Statement

The objective of this project is to implement and compare multiple classical machine learning classification algorithms on a real-world dataset containing **500 instances and 12 features**.

Six machine learning models were trained and evaluated using multiple performance metrics to provide a comprehensive comparison. The evaluation metrics used are:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

The goal is to analyze the predictive performance of each model and determine which algorithm performs best on the selected dataset.

---

## 2️ Dataset Description

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, originally obtained from the UCI Machine Learning Repository and accessed via `sklearn.datasets`.

### Dataset Characteristics:

- Total samples used: **500**
- Number of features: **12**
- Target type: **Binary classification**
  - 0 → Malignant
  - 1 → Benign
- Feature type: Continuous numerical features

For this assignment:

- The dataset was stratified and reduced to **500 instances**
- Exactly **12 numerical features** were selected for training
- The data was split into **80% training** and **20% testing** using stratified sampling

---

## 3️ Model Performance Comparison

### Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|------|------------|---------|------|------|
| Logistic Regression | 0.90 | 0.9725 | 0.9077 | 0.9365 | 0.9219 | 0.7838 |
| Decision Tree | 0.86 | 0.8554 | 0.9016 | 0.8730 | 0.8871 | 0.7036 |
| kNN (k=5) | 0.92 | 0.9562 | 0.9231 | 0.9524 | 0.9375 | 0.8272 |
| Gaussian Naive Bayes | 0.88 | 0.9567 | 0.8806 | 0.9365 | 0.9077 | 0.7396 |
| Random Forest (Ensemble) | 0.90 | 0.9708 | 0.9206 | 0.9206 | 0.9206 | 0.7855 |
| XGBoost (Ensemble) | 0.90 | 0.9717 | 0.9344 | 0.9048 | 0.9194 | 0.7886 |

---

## 4️ Model Observations and Analysis

### Logistic Regression  
Logistic Regression achieved strong performance with an AUC of 0.9725, indicating excellent class separability. It demonstrated high recall (0.9365), making it effective at correctly identifying positive cases. The model provides stable and well-calibrated probability outputs.

---

### Decision Tree  
The Decision Tree model showed comparatively lower performance with an accuracy of 0.86 and AUC of 0.8554. While it captures nonlinear relationships, it may have overfitted the training data, resulting in reduced generalization performance compared to ensemble methods.

---

### kNN (k=5)  
k-Nearest Neighbors achieved the highest accuracy (0.92) and F1-score (0.9375) among all models. With strong recall (0.9524) and the highest MCC (0.8272), kNN demonstrated excellent overall classification balance after proper feature scaling.

---

### Gaussian Naive Bayes  
Gaussian Naive Bayes performed reasonably well with strong recall (0.9365), but slightly lower precision compared to other models. The independence assumption may limit performance when features are correlated, resulting in slightly reduced overall accuracy (0.88).

---

### Random Forest (Ensemble)  
Random Forest delivered stable and robust performance across all metrics, with an AUC of 0.9708. As an ensemble method, it reduces variance and improves generalization compared to a single Decision Tree.

---

### XGBoost (Ensemble)  
XGBoost achieved competitive performance with high precision (0.9344) and strong AUC (0.9717). Its gradient boosting framework allows it to model complex feature interactions effectively, providing strong predictive performance comparable to Random Forest and Logistic Regression.

---

## 5️ Conclusion

Among all implemented models, **kNN (k=5)** achieved the highest overall performance in terms of Accuracy, F1-score, and MCC, indicating strong predictive balance.

Ensemble methods such as **Random Forest and XGBoost** demonstrated robust and stable performance with high AUC scores, highlighting their ability to generalize well.

Logistic Regression also performed competitively, showing that the dataset exhibits a significant degree of linear separability.

This comparison demonstrates the importance of evaluating multiple performance metrics when selecting a model, rather than relying solely on accuracy.
