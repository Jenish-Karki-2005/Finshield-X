import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. DATA ANALYSIS
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget distribution:")
print(df["isFraud"].value_counts())

print("\nTarget percentage:")
print(df["isFraud"].value_counts(normalize=True) * 100)

print("\nData types:")
print(df.dtypes)

print("\nTransaction types:")
print(df["type"].value_counts())

print("\nFraud count by transaction type:")
print(pd.crosstab(df["type"], df["isFraud"]))

print("\nFraud percentage by transaction type:")
print(
    pd.crosstab(
        df["type"],
        df["isFraud"],
        normalize="index"
    ) * 100
)


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

df["orig_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

df["dest_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

df["amount_to_orig_balance"] = (
    df["amount"] / (df["oldbalanceOrg"] + 1)
)

print("\nNew engineered features:")
print(
    df[
        [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "orig_balance_change",
            "dest_balance_change",
            "amount_to_orig_balance"
        ]
    ].head()
)


df["is_large_transaction"] = (
    df["amount"] > df["amount"].quantile(0.95)
).astype(int)

df["is_balance_empty_after"] = (
    (df["newbalanceOrig"] == 0) |
    (df["newbalanceDest"] == 0)
).astype(int)

df["balance_error_orig"] = (
    df["oldbalanceOrg"]
    - df["amount"]
    - df["newbalanceOrig"]
)

df["balance_error_dest"] = (
    df["oldbalanceDest"]
    + df["amount"]
    - df["newbalanceDest"]
)

print("\nAdditional engineered features:")
print(
    df[
        [
            "amount",
            "is_large_transaction",
            "is_balance_empty_after",
            "balance_error_orig",
            "balance_error_dest"
        ]
    ].head()
)


# ==========================================
# 4. FEATURE SELECTION
# ==========================================

selected_features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_change",
    "dest_balance_change",
    "amount_to_orig_balance",
    "is_large_transaction",
    "is_balance_empty_after",
    "balance_error_orig",
    "balance_error_dest"
]

X = df[selected_features]

y = df["isFraud"]

print("\nSelected features:")
print(X.columns.tolist())

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True) * 100)


# ==========================================
# 6. FEATURE PREPROCESSING
# ==========================================

numerical_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_change",
    "dest_balance_change",
    "amount_to_orig_balance",
    "is_large_transaction",
    "is_balance_empty_after",
    "balance_error_orig",
    "balance_error_dest"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "type_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["type"]
        ),
        (
            "scaler",
            StandardScaler(),
            numerical_features
        )
    ],
    sparse_threshold=1.0
)

# Learn preprocessing from training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply learned preprocessing to testing data
X_test_processed = preprocessor.transform(X_test)


# ==========================================
# 7. FINAL PREPROCESSING CHECK
# ==========================================

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nData preprocessing completed successfully!")


# ==========================================
# 8. LOGISTIC REGRESSION MODEL
# ==========================================

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train_processed, y_train)

print("\nLogistic Regression model trained successfully!")


# ==========================================
# 9. LOGISTIC REGRESSION PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_processed)

print("\nFirst 20 predictions:")
print(y_pred[:20])


# ==========================================
# 10. LOGISTIC REGRESSION EVALUATION
# ==========================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 11. RANDOM FOREST MODEL
# ==========================================

sample_size = 500000

X_train_rf = X_train_processed[:sample_size]
y_train_rf = y_train.iloc[:sample_size]

print("\nRandom Forest training data shape:")
print(X_train_rf.shape)

rf_model = RandomForestClassifier(
    n_estimators=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nStarting Random Forest training...")

rf_model.fit(X_train_rf, y_train_rf)

print("\nRandom Forest model trained successfully!")

# ==========================================
# 12. RANDOM FOREST PREDICTIONS
# ==========================================

rf_pred = rf_model.predict(X_test_processed)

print("\nRandom Forest predictions:")
print(rf_pred[:20])


# ==========================================
# 13. RANDOM FOREST EVALUATION
# ==========================================

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))