# Machine Learning Assignment-2
## Heart Disease Prediction using Machine Learning

**BITS Pilani, WILP**

---

# Student Details

- **Student Name:** Srinath
- **Program:** M.Tech (WILP)
- **Course:** Machine Learning
- **Assignment:** Assignment-2
- **Dataset:** Heart Disease Dataset (Kaggle)

---

# a. Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early detection of heart disease enables timely medical intervention and improves patient outcomes.

The objective of this project is to build a Machine Learning-based Heart Disease Prediction System that classifies whether a patient is likely to have heart disease based on various clinical parameters. Multiple supervised machine learning algorithms are implemented, evaluated, and compared. Finally, the trained models are deployed through a Streamlit web application where users can upload test data and obtain predictions along with evaluation metrics.

---

# b. Dataset Description

### Dataset Name

Heart Disease Dataset

### Source

Kaggle

### Dataset Description

The dataset contains patient medical records collected from healthcare institutions. Each record contains demographic information, clinical measurements, and diagnostic test results that are used to predict the presence or absence of heart disease.

### Features

| Feature | Description |
|----------|-------------|
| age | Age of the patient |
| sex | Gender (1 = Male, 0 = Female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia |
| target | Heart disease prediction (0 = No Disease, 1 = Disease) |

### Dataset Summary

- Number of Records : **303**
- Number of Features : **13**
- Target Variable : **target**
- Classification Type : **Binary Classification**

---

# c. GitHub Repository Link

GitHub Repository

**https://github.com/yourusername/Heart-Disease-Prediction**

*(Replace with your GitHub repository URL.)*

---

## Streamlit Deployment

**https://your-app-name.streamlit.app**

*(Replace with your deployed Streamlit application URL.)*

---

# d. Models Used

The following supervised machine learning algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

---

# Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------------------------|:-------:|:------:|:---------:|:------:|:--------:|:------:|
| Logistic Regression | **0.7869** | **0.8647** | **0.7632** | **0.8788** | **0.8169** | **0.5731** |
| Decision Tree | **0.8197** | **0.8236** | **0.7500** | **1.0000** | **0.8571** | **0.6748** |
| K-Nearest Neighbors (KNN) | **0.8033** | **0.8674** | **0.7692** | **0.9091** | **0.8333** | **0.6098** |
| Gaussian Naive Bayes | **0.8197** | **0.8853** | **0.7895** | **0.9091** | **0.8451** | **0.6410** |
| Random Forest (Ensemble) | **0.8033** | **0.8864** | **0.7561** | **0.9394** | **0.8378** | **0.6181** |

---

# Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **Logistic Regression** | Logistic Regression achieved an accuracy of **78.69%** with an AUC of **0.8647**. It provides a reliable baseline classifier and performs well in identifying positive heart disease cases due to its high recall. |
| **Decision Tree** | Decision Tree achieved the highest classification accuracy (**81.97%**) and a perfect recall (**100%**). It also produced the highest F1-score and MCC, making it the best-performing model on this dataset. |
| **K-Nearest Neighbors (KNN)** | KNN achieved balanced performance after feature scaling with an accuracy of **80.33%** and an F1-score of **0.8333**. It effectively classifies patients using distance-based learning. |
| **Gaussian Naive Bayes** | Gaussian Naive Bayes achieved **81.97%** accuracy with a high AUC (**0.8853**). Despite assuming feature independence, it delivered competitive classification performance while being computationally efficient. |
| **Random Forest (Ensemble)** | Random Forest achieved the highest ROC-AUC (**0.8864**) and very high recall (**93.94%**). Ensemble learning reduced model variance and produced stable prediction performance across the dataset. |

---

# Overall Winner for the Dataset

## Decision Tree

### Reason

Based on the experimental results, the **Decision Tree classifier** achieved the best overall performance on the Heart Disease dataset.

It achieved:

- Highest Accuracy (**81.97%**)
- Highest Recall (**100%**)
- Highest F1 Score (**0.8571**)
- Highest Matthews Correlation Coefficient (MCC) (**0.6748**)

Although the Random Forest model achieved the highest ROC-AUC score (**0.8864**), the Decision Tree demonstrated superior overall classification performance based on the majority of evaluation metrics. Therefore, it is selected as the best-performing model for this dataset.

---

# Project Structure

```
Heart_Disease_Prediction/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── comparison_results.csv
├── model_ranking.csv
├── model_observations.csv
│
├── dataset/
│   └── heart.csv
│
└── models/
    ├── scaler.pkl
    ├── logistic.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

---

# How to Run the Project

## Step 1: Install Required Packages

```bash
pip install -r requirements.txt
```

## Step 2: Train the Models

```bash
python train_models.py
```

## Step 3: Launch the Streamlit Application

```bash
streamlit run app.py
```

## Step 4: Use the Application

1. Open the Streamlit application in your browser.
2. Select the desired machine learning model.
3. Upload the generated `test_data.csv`.
4. View the evaluation metrics, confusion matrix, classification report, and prediction results.
5. Download the prediction results as a CSV file.

---

# Conclusion

This project successfully developed and compared five supervised machine learning algorithms for predicting heart disease using the Heart Disease dataset. The experimental evaluation demonstrated that all models achieved satisfactory performance, with the Decision Tree classifier producing the best overall results in terms of Accuracy, Recall, F1-score, and MCC. The project was further deployed using Streamlit, enabling users to upload test datasets and evaluate different machine learning models through an interactive web application.