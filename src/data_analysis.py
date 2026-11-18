import pandas as pd

# Load dataset
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget distribution:")
print(df["isFraud"].value_counts())

print("\nTarget percentage:")
print(df["isFraud"].value_counts(normalize=True) * 100)

print("\nData types:")
print(df.dtypes)

print("\nUnique values in transaction type:")
print(df["type"].value_counts())

print("\nData types:")
print(df.dtypes)

print("\nTransaction types:")
print(df["type"].value_counts())

print("\nFraud count by transaction type:")
print(pd.crosstab(df["type"], df["isFraud"]))

print("\nFraud percentage by transaction type:")
print(pd.crosstab(df["type"], df["isFraud"], normalize="index") * 100)

# -------------------------------
# Feature Engineering
# -------------------------------

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
    df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
)

df["balance_error_dest"] = (
    df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
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

# -------------------------------
# Separate Features and Target
# -------------------------------

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

print("\nFeatures (X):")
print(X.columns.tolist())

print("\nTarget (y):")
print(y.name)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

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