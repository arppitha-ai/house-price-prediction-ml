import pandas as pd

df = pd.read_csv("data/houses.csv")

print("FIRST 5 ROWS:")
print(df.head())

print("\nSHAPE:")
print(df.shape)

print("\nMISSING VALUES:")
print(df.isnull().sum())