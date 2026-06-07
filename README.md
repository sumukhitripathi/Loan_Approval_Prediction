# Loan Approval Prediction

This repository contains a small Python machine learning project for predicting loan approval status from applicant and asset information. It uses a CSV dataset, basic preprocessing, and a Logistic Regression classifier from scikit-learn.

## Project Structure

```text
.
+-- loan_approval_dataset.csv  # Loan application dataset
+-- data_info.py                   # Dataset loading and exploratory summary helpers
+-- main.py                    # Model training, evaluation, and sample prediction
+-- README.md
```

## Dataset

The project expects `loan_approval_dataset.csv` in the repository root. The dataset includes the following columns:

- `loan_id`
- `no_of_dependents`
- `education`
- `self_employed`
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`
- `loan_status`

The target column is `loan_status`, which indicates whether a loan was approved.

## Requirements

Install the Python dependencies:

```bash
pip install pandas scikit-learn
```

Python 3.9 or later is recommended.

## Usage

To inspect the dataset shape, columns, data types, missing values, and summary statistics:

```bash
python data_info.py
```

To train and evaluate the Logistic Regression model:

```bash
python main.py
```

The training script:

1. Loads `loan_approval_dataset.csv`.
2. Fills missing numeric values with the mean.
3. Fills missing categorical values with the most frequent value.
4. Encodes categorical columns with `LabelEncoder`.
5. Scales numeric columns with `MinMaxScaler`.
6. Splits the data into training and testing sets.
7. Trains a Logistic Regression model.
8. Prints accuracy, confusion matrix, and classification report.
9. Runs a sample prediction for a new applicant.

## Model

The current model is a Logistic Regression classifier with `max_iter=1000`. The helper functions in `main.py` are:

- `train_logistic_regression(X_train, y_train)`
- `evaluate_model(model, X_test, y_test)`
- `predict_loan_approval(new_applicant, model)`

## Notes

- `loan_id` is dropped before training because it is an identifier rather than a predictive feature.
- The current script preprocesses the full dataset before splitting into train and test sets. For production-grade evaluation, fit imputers, encoders, and scalers only on the training data, then apply them to the test data.
- The sample applicant in `main.py` must use the same encoded/scaled feature format expected by the trained model.
