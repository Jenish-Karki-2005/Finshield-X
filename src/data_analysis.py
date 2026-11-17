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