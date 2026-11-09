# CREDIT CARD FRAUD DETECTION
# STEP 1: IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 2: LOAD DATASET

df = pd.read_csv("dataset/creditcard.csv")

# STEP 3: DISPLAY BASIC INFORMATION

print("===================================")
print(" CREDIT CARD FRAUD DETECTION SYSTEM")
print("===================================")

print("\n1. Dataset Loaded Successfully!")

print("\n2. Dataset Shape:")
print(df.shape)

print("\n3. First Five Records:")
print(df.head())

print("\n4. Dataset Information:")
print(df.info())

print("\n5. Statistical Summary:")
print(df.describe())

# STEP 4: DATA QUALITY ANALYSIS

print("\n========== DATA QUALITY ANALYSIS ==========")

# 1. Check Missing Values
print("\n1. Missing Values:")
print(df.isnull().sum().sum())

# 2. Check Duplicate Records
print("\n2. Duplicate Records:")
print(df.duplicated().sum())

# 3. Check Data Types
print("\n3. Data Types:")
print(df.dtypes)

# 4. Transaction Distribution
print("\n4. Transaction Distribution:")
print(df['Class'].value_counts())

# 5. Transaction Percentages
print("\n5. Transaction Percentages:")
print(df['Class'].value_counts(normalize=True) * 100)

# STEP 5: FRAUD DISTRIBUTION VISUALIZATION

plt.figure(figsize=(8, 5))

sns.countplot(
    x='Class',
    data=df
)

plt.title("Credit Card Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Type (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.xticks([0, 1], ['Legitimate', 'Fraud'])

plt.savefig("outputs/fraud_distribution.png")

plt.show()

# STEP 6: CORRELATION HEATMAP

print("\n6. Generating Correlation Heatmap...")

correlation = df.corr(numeric_only=True)

plt.figure(figsize=(12, 10))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap of Credit Card Transactions")

plt.tight_layout()

plt.savefig("outputs/correlation_heatmap.png")

plt.show()

# STEP 7: DATA PREPROCESSING

print("\n7. DATA PREPROCESSING")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())


# ==================================================
# STEP 8: FEATURE SCALING AND DATA SPLITTING
# ==================================================

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("\n" + "=" * 50)
print("8. FEATURE SCALING AND DATA SPLITTING")
print("=" * 50)

# Separate input features and target variable
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nOriginal Features Shape:")
print(X.shape)

print("\nTarget Variable Shape:")
print(y.shape)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# Apply feature scaling
scaler = StandardScaler()

# Fit scaler only on training data
X_train = X_train.copy()
X_test = X_test.copy()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

print("\nFeature Scaling Completed Successfully!")

# Display transaction distribution
print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())

print("\nSTEP 8 COMPLETED SUCCESSFULLY!")
print("=" * 50)
