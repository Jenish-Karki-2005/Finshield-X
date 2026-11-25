
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


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
# 7. FINAL SHAPE
# ==========================================

print("\nProcessed training shape:", X_train_processed.shape)

print("Processed testing shape:", X_test_processed.shape)

print("\nData preprocessing completed successfully!")
