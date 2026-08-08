# ==========================================================
# BITS Pilani WILP
# Machine Learning Assignment - 2
# Heart Disease Prediction
# train_models.py
# ==========================================================

# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ==========================================================
# Create Required Folders
# ==========================================================

if not os.path.exists("models"):
    os.makedirs("models")

print("=" * 70)
print("BITS PILANI WILP")
print("Machine Learning Assignment - 2")
print("Heart Disease Prediction")
print("=" * 70)

# ==========================================================
# Load Dataset
# ==========================================================

DATASET_PATH = "dataset/heart.csv"

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"\nERROR: Dataset not found at {DATASET_PATH}")
    print("Please place heart.csv inside the dataset folder.")
    raise

print("\nDataset Loaded Successfully")

# ==========================================================
# Basic Dataset Information
# ==========================================================

print("\nFirst Five Records")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ==========================================================
# Check Missing Values
# ==========================================================

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================================
# Remove Missing Values (if any)
# ==========================================================

df.dropna(inplace=True)

print("\nDataset Shape After Cleaning")
print(df.shape)

# ==========================================================
# Remove Duplicate Records
# ==========================================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate Records :", duplicate_count)

if duplicate_count > 0:
    df.drop_duplicates(inplace=True)

print("Dataset Shape After Removing Duplicates")
print(df.shape)

# ==========================================================
# Verify Target Column
# ==========================================================

if "target" not in df.columns:
    raise Exception("Target column 'target' not found in dataset.")

print("\nTarget Column Verified")

print("\nTarget Distribution")
print(df["target"].value_counts())

print("\nPart 1A Completed Successfully")

# ==========================================================
# Separate Features and Target
# ==========================================================

X = df.drop("target", axis=1)

y = df["target"]

print("\n")
print("=" * 70)
print("FEATURES AND TARGET")
print("=" * 70)

print("Feature Shape :", X.shape)

print("Target Shape  :", y.shape)

print("\nFeature Columns")

print(X.columns.tolist())

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\n")
print("=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print("Training Samples :", X_train.shape[0])

print("Testing Samples  :", X_test.shape[0])

print("Training Features :", X_train.shape)

print("Testing Features  :", X_test.shape)

# ==========================================================
# Save Original Test Dataset
# (Required for Streamlit)
# ==========================================================

test_data = X_test.copy()

test_data["target"] = y_test.values

test_data.to_csv(

    "test_data.csv",

    index=False

)

print("\nOriginal test_data.csv Saved Successfully")

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(

    scaler.fit_transform(X_train),

    columns=X_train.columns,

    index=X_train.index

)

X_test_scaled = pd.DataFrame(

    scaler.transform(X_test),

    columns=X_test.columns,

    index=X_test.index

)

print("\nFeature Scaling Completed Successfully")

# ==========================================================
# Save Scaler
# ==========================================================

joblib.dump(

    scaler,

    "models/scaler.pkl"

)

print("Scaler Saved Successfully")

# ==========================================================
# Verify Feature Names
# ==========================================================

print("\n")
print("=" * 70)
print("FEATURE VERIFICATION")
print("=" * 70)

print("Training Features")

print(X_train.columns.tolist())

print("\nScaled Features")

print(X_train_scaled.columns.tolist())

# ==========================================================
# Result Storage
# ==========================================================

results = []

print("\nResults List Initialized")

print("\nPart 1B Completed Successfully")

# ==========================================================
# PART 2A : Logistic Regression
# ==========================================================

print("\n")
print("=" * 70)
print("PART 2A : LOGISTIC REGRESSION")
print("=" * 70)

# ----------------------------------------------------------
# Create Model
# ----------------------------------------------------------

logistic_model = LogisticRegression(

    max_iter=1000,

    random_state=42

)

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

logistic_model.fit(

    X_train_scaled,

    y_train

)

print("\nLogistic Regression Model Trained Successfully")

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

joblib.dump(

    logistic_model,

    "models/logistic.pkl"

)

print("Model Saved Successfully")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred = logistic_model.predict(

    X_test_scaled

)

y_prob = logistic_model.predict_proba(

    X_test_scaled

)[:, 1]

# ----------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred

)

auc = roc_auc_score(

    y_test,

    y_prob

)

mcc = matthews_corrcoef(

    y_test,

    y_pred

)

# ----------------------------------------------------------
# Print Results
# ----------------------------------------------------------

print("\n")
print("=" * 70)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print(f"AUC Score : {auc:.4f}")

print(f"MCC Score : {mcc:.4f}")

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(

    y_test,

    y_pred

)

print("\nConfusion Matrix")

print(cm)

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report")

print(

    classification_report(

        y_test,

        y_pred

    )

)

# ----------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Coefficient": logistic_model.coef_[0]

})

feature_importance = feature_importance.sort_values(

    by="Coefficient",

    ascending=False

)

feature_importance.to_csv(

    "logistic_feature_importance.csv",

    index=False

)

print("\nFeature Importance Saved")

# ----------------------------------------------------------
# Save Results
# ----------------------------------------------------------

results.append({

    "Model": "Logistic Regression",

    "Accuracy": accuracy,

    "AUC": auc,

    "Precision": precision,

    "Recall": recall,

    "F1": f1,

    "MCC": mcc

})

print("\nLogistic Regression Completed Successfully")

# ==========================================================
# PART 2B : Decision Tree Classifier
# ==========================================================

print("\n")
print("=" * 70)
print("PART 2B : DECISION TREE CLASSIFIER")
print("=" * 70)

# ----------------------------------------------------------
# Create Model
# ----------------------------------------------------------

decision_tree_model = DecisionTreeClassifier(

    criterion="gini",

    max_depth=5,

    random_state=42

)

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

decision_tree_model.fit(

    X_train,

    y_train

)

print("\nDecision Tree Model Trained Successfully")

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

joblib.dump(

    decision_tree_model,

    "models/decision_tree.pkl"

)

print("Decision Tree Model Saved Successfully")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred = decision_tree_model.predict(

    X_test

)

y_prob = decision_tree_model.predict_proba(

    X_test

)[:, 1]

# ----------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred

)

auc = roc_auc_score(

    y_test,

    y_prob

)

mcc = matthews_corrcoef(

    y_test,

    y_pred

)

# ----------------------------------------------------------
# Print Results
# ----------------------------------------------------------

print("\n")
print("=" * 70)
print("DECISION TREE RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC Score : {auc:.4f}")
print(f"MCC Score : {mcc:.4f}")

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(

    y_test,

    y_pred

)

print("\nConfusion Matrix")

print(cm)

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report")

print(

    classification_report(

        y_test,

        y_pred

    )

)

# ----------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": decision_tree_model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

feature_importance.to_csv(

    "decision_tree_feature_importance.csv",

    index=False

)

print("\nDecision Tree Feature Importance Saved")

# ----------------------------------------------------------
# Save Results
# ----------------------------------------------------------

results.append({

    "Model": "Decision Tree",

    "Accuracy": accuracy,

    "AUC": auc,

    "Precision": precision,

    "Recall": recall,

    "F1": f1,

    "MCC": mcc

})

print("\nDecision Tree Completed Successfully")

# ==========================================================
# PART 3A : K-Nearest Neighbors (KNN)
# ==========================================================

print("\n")
print("=" * 70)
print("PART 3A : K-NEAREST NEIGHBORS")
print("=" * 70)

# ----------------------------------------------------------
# Create Model
# ----------------------------------------------------------

knn_model = KNeighborsClassifier(

    n_neighbors=5,

    metric="minkowski",

    p=2

)

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

knn_model.fit(

    X_train_scaled,

    y_train

)

print("\nKNN Model Trained Successfully")

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

joblib.dump(

    knn_model,

    "models/knn.pkl"

)

print("KNN Model Saved Successfully")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred = knn_model.predict(

    X_test_scaled

)

y_prob = knn_model.predict_proba(

    X_test_scaled

)[:,1]

# ----------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred

)

auc = roc_auc_score(

    y_test,

    y_prob

)

mcc = matthews_corrcoef(

    y_test,

    y_pred

)

# ----------------------------------------------------------
# Print Results
# ----------------------------------------------------------

print("\n")
print("=" * 70)
print("KNN RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC Score : {auc:.4f}")
print(f"MCC Score : {mcc:.4f}")

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(

    y_test,

    y_pred

)

print("\nConfusion Matrix")

print(cm)

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report")

print(

    classification_report(

        y_test,

        y_pred

    )

)

# ----------------------------------------------------------
# Training and Testing Accuracy
# ----------------------------------------------------------

train_accuracy = knn_model.score(

    X_train_scaled,

    y_train

)

test_accuracy = knn_model.score(

    X_test_scaled,

    y_test

)

print("\nTraining Accuracy :", round(train_accuracy,4))

print("Testing Accuracy  :", round(test_accuracy,4))

# ----------------------------------------------------------
# Find Best K
# ----------------------------------------------------------

k_values = []

accuracy_values = []

for k in range(1,21):

    model = KNeighborsClassifier(

        n_neighbors=k

    )

    model.fit(

        X_train_scaled,

        y_train

    )

    score = model.score(

        X_test_scaled,

        y_test

    )

    k_values.append(k)

    accuracy_values.append(score)

k_analysis = pd.DataFrame({

    "K": k_values,

    "Accuracy": accuracy_values

})

k_analysis.to_csv(

    "knn_k_analysis.csv",

    index=False

)

best_k = k_analysis.loc[
    k_analysis["Accuracy"].idxmax(),
    "K"
]

best_accuracy = k_analysis["Accuracy"].max()

print("\nBest K :", best_k)

print("Best Accuracy :", round(best_accuracy,4))

# ----------------------------------------------------------
# Save Results
# ----------------------------------------------------------

results.append({

    "Model":"K-Nearest Neighbors",

    "Accuracy":accuracy,

    "AUC":auc,

    "Precision":precision,

    "Recall":recall,

    "F1":f1,

    "MCC":mcc

})

print("\nKNN Completed Successfully")

# ==========================================================
# PART 3B : Gaussian Naive Bayes
# ==========================================================

print("\n")
print("=" * 70)
print("PART 3B : GAUSSIAN NAIVE BAYES")
print("=" * 70)

# ----------------------------------------------------------
# Create Model
# ----------------------------------------------------------

naive_bayes_model = GaussianNB()

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

naive_bayes_model.fit(

    X_train_scaled,

    y_train

)

print("\nGaussian Naive Bayes Model Trained Successfully")

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

joblib.dump(

    naive_bayes_model,

    "models/naive_bayes.pkl"

)

print("Gaussian Naive Bayes Model Saved Successfully")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred = naive_bayes_model.predict(

    X_test_scaled

)

y_prob = naive_bayes_model.predict_proba(

    X_test_scaled

)[:, 1]

# ----------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred

)

auc = roc_auc_score(

    y_test,

    y_prob

)

mcc = matthews_corrcoef(

    y_test,

    y_pred

)

# ----------------------------------------------------------
# Print Results
# ----------------------------------------------------------

print("\n")
print("=" * 70)
print("GAUSSIAN NAIVE BAYES RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC Score : {auc:.4f}")
print(f"MCC Score : {mcc:.4f}")

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(

    y_test,

    y_pred

)

print("\nConfusion Matrix")

print(cm)

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report")

print(

    classification_report(

        y_test,

        y_pred

    )

)

# ----------------------------------------------------------
# Save Feature Statistics
# ----------------------------------------------------------

feature_statistics = pd.DataFrame({

    "Feature": X.columns,

    "Mean": naive_bayes_model.theta_[0],

    "Variance": naive_bayes_model.var_[0]

})

feature_statistics.to_csv(

    "naive_bayes_feature_statistics.csv",

    index=False

)

print("\nNaive Bayes Feature Statistics Saved")

# ----------------------------------------------------------
# Save Results
# ----------------------------------------------------------

results.append({

    "Model": "Gaussian Naive Bayes",

    "Accuracy": accuracy,

    "AUC": auc,

    "Precision": precision,

    "Recall": recall,

    "F1": f1,

    "MCC": mcc

})

print("\nGaussian Naive Bayes Completed Successfully")

# ==========================================================
# PART 4A : Random Forest Classifier
# ==========================================================

print("\n")
print("=" * 70)
print("PART 4A : RANDOM FOREST CLASSIFIER")
print("=" * 70)

# ----------------------------------------------------------
# Create Model
# ----------------------------------------------------------

random_forest_model = RandomForestClassifier(

    n_estimators=200,

    criterion="gini",

    max_depth=10,

    random_state=42,

    n_jobs=-1

)

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

random_forest_model.fit(

    X_train,

    y_train

)

print("\nRandom Forest Model Trained Successfully")

# ----------------------------------------------------------
# Save Model
# ----------------------------------------------------------

joblib.dump(

    random_forest_model,

    "models/random_forest.pkl"

)

print("Random Forest Model Saved Successfully")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

y_pred = random_forest_model.predict(

    X_test

)

y_prob = random_forest_model.predict_proba(

    X_test

)[:, 1]

# ----------------------------------------------------------
# Evaluation Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred

)

auc = roc_auc_score(

    y_test,

    y_prob

)

mcc = matthews_corrcoef(

    y_test,

    y_pred

)

# ----------------------------------------------------------
# Print Results
# ----------------------------------------------------------

print("\n")
print("=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC Score : {auc:.4f}")
print(f"MCC Score : {mcc:.4f}")

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(

    y_test,

    y_pred

)

print("\nConfusion Matrix")

print(cm)

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

print("\nClassification Report")

print(

    classification_report(

        y_test,

        y_pred

    )

)

# ----------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": random_forest_model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

feature_importance.to_csv(

    "random_forest_feature_importance.csv",

    index=False

)

print("\nRandom Forest Feature Importance Saved")

# ----------------------------------------------------------
# Training / Testing Accuracy
# ----------------------------------------------------------

train_accuracy = random_forest_model.score(

    X_train,

    y_train

)

test_accuracy = random_forest_model.score(

    X_test,

    y_test

)

print("\nTraining Accuracy :", round(train_accuracy,4))

print("Testing Accuracy  :", round(test_accuracy,4))

# ----------------------------------------------------------
# Save Results
# ----------------------------------------------------------

results.append({

    "Model": "Random Forest",

    "Accuracy": accuracy,

    "AUC": auc,

    "Precision": precision,

    "Recall": recall,

    "F1": f1,

    "MCC": mcc

})

print("\nRandom Forest Completed Successfully")

# ==========================================================
# PART 4B : Model Comparison & Final Outputs
# ==========================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# ----------------------------------------------------------
# Convert Results to DataFrame
# ----------------------------------------------------------

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.round(4)

print("\nModel Comparison")

print(comparison_df)

# ----------------------------------------------------------
# Save Comparison Results
# ----------------------------------------------------------

comparison_df.to_csv(

    "comparison_results.csv",

    index=False

)

print("\ncomparison_results.csv Saved Successfully")

# ----------------------------------------------------------
# Rank Models
# ----------------------------------------------------------

ranking_df = comparison_df.sort_values(

    by=["Accuracy", "AUC", "MCC"],

    ascending=False

).reset_index(drop=True)

ranking_df.insert(

    0,

    "Rank",

    range(1, len(ranking_df) + 1)

)

ranking_df.to_csv(

    "model_ranking.csv",

    index=False

)

print("\nmodel_ranking.csv Saved Successfully")

# ----------------------------------------------------------
# Best Model
# ----------------------------------------------------------

best_model = ranking_df.iloc[0]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model :", best_model["Model"])

print("Accuracy :", best_model["Accuracy"])

print("AUC :", best_model["AUC"])

print("Precision :", best_model["Precision"])

print("Recall :", best_model["Recall"])

print("F1 :", best_model["F1"])

print("MCC :", best_model["MCC"])

# ----------------------------------------------------------
# Model Observations
# ----------------------------------------------------------

observations = [

    {
        "Model":"Logistic Regression",
        "Observation":"Good baseline model. Performs well on linearly separable data."
    },

    {
        "Model":"Decision Tree",
        "Observation":"Easy to interpret but can overfit without pruning."
    },

    {
        "Model":"K-Nearest Neighbors",
        "Observation":"Distance-based classifier. Requires scaled features."
    },

    {
        "Model":"Gaussian Naive Bayes",
        "Observation":"Fast probabilistic classifier assuming feature independence."
    },

    {
        "Model":"Random Forest",
        "Observation":"Ensemble model with high accuracy and low overfitting."
    }

]

observations_df = pd.DataFrame(observations)

observations_df.to_csv(

    "model_observations.csv",

    index=False

)

print("\nmodel_observations.csv Saved Successfully")

# ----------------------------------------------------------
# Final Summary
# ----------------------------------------------------------

print("\n")
print("=" * 70)
print("PROJECT FILES GENERATED")
print("=" * 70)

generated_files = [

    "models/scaler.pkl",
    "models/logistic.pkl",
    "models/decision_tree.pkl",
    "models/knn.pkl",
    "models/naive_bayes.pkl",
    "models/random_forest.pkl",
    "test_data.csv",
    "comparison_results.csv",
    "model_ranking.csv",
    "model_observations.csv",
    "logistic_feature_importance.csv",
    "decision_tree_feature_importance.csv",
    "random_forest_feature_importance.csv",
    "naive_bayes_feature_statistics.csv",
    "knn_k_analysis.csv"

]

for file in generated_files:
    print("✓", file)

print("\n")
print("=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)