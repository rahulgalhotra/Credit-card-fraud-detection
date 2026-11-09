# Credit Card Fraud Detection — Data Mining Lab Project

An advanced academic project that compares supervised classification and unsupervised anomaly detection on the European Credit Card Fraud dataset.

## Dataset

Download `creditcard.csv` from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it at `dataset/creditcard.csv`. The dataset is not included in this ZIP because it is hosted by Kaggle and requires a separate download.

## Requirements

- Python 3.10 or newer recommended
- Internet connection for initial package installation and dataset download

## Setup

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run training

From the project root:

```bash
python train.py
```

This script:
1. Loads and checks the dataset.
2. Removes exact duplicate rows.
3. Splits data using stratification.
4. Scales Time and Amount using training data only.
5. Generates exploratory data analysis plots.
6. Trains Logistic Regression, Random Forest and XGBoost.
7. Uses SMOTE only inside training pipelines for selected models.
8. Trains an Isolation Forest anomaly-detection baseline.
9. Compares models using precision, recall, F1, ROC-AUC and PR-AUC.
10. Saves the best supervised pipeline and output charts.

Generated files are saved in `outputs/` and `models/`.

## Run the dashboard

After training:

```bash
streamlit run app.py
```

Upload a CSV containing the same feature columns as the original dataset. The dashboard will show predictions and estimated fraud probabilities.

## Important academic notes

- Do not apply SMOTE before the train/test split.
- Do not use the test set for hyperparameter selection or threshold tuning.
- Accuracy alone is misleading for this highly imbalanced dataset.
- Compare recall, precision, F1, ROC-AUC and especially PR-AUC.
- Isolation Forest uses anomaly labels and is not directly comparable to supervised probability models.
- This is an educational demonstration, not a production banking system.

## Suggested report chapters

1. Introduction
2. Problem Statement and Objectives
3. Dataset Description
4. Data Preprocessing
5. Exploratory Data Analysis
6. Class Imbalance and SMOTE
7. Algorithms Used
8. Model Training and Evaluation
9. Results and Discussion
10. Dashboard
11. Limitations and Conclusion
12. References

## Project structure

```text
Credit_Card_Fraud_Detection/
├── dataset/
├── models/
├── outputs/
├── app.py
├── train.py
├── requirements.txt
└── README.md
```
