
# ==================================================
# CREDIT CARD FRAUD DETECTION
# STEP 9: MACHINE LEARNING MODEL TRAINING
# ==================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 55)
print("CREDIT CARD FRAUD DETECTION SYSTEM")
print("STEP 9: MODEL TRAINING")
print("=" * 55)

# 1. Load Dataset
print("\n1. Loading Dataset...")

df = pd.read_csv("dataset/creditcard.csv")

# Remove duplicate records
df = df.drop_duplicates()

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# 2. Separate Features and Target
print("\n2. Separating Features and Target...")

X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Split Dataset
print("\n3. Splitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Feature Scaling
print("\n4. Applying Feature Scaling...")

scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

# 5. Create Models
print("\n5. Creating Machine Learning Models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

# 6. Train and Evaluate Models
results = []

print("\n6. Training Models...")

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save trained model
    os.makedirs("outputs", exist_ok=True)

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(model, "outputs/" + filename)

# 7. Display Model Comparison
print("\n7. MODEL COMPARISON")

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# 8. Save Scaler
joblib.dump(scaler, "outputs/scaler.pkl")

print("\nAll Models Trained Successfully!")
print("Models and results saved in outputs folder.")

print("=" * 55)
