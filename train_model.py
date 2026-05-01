import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# load dataset
df = pd.read_csv("data/houses.csv")

# split features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# train model
model.fit(X_train, y_train)

# predictions
pred = model.predict(X_test)

# evaluation
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

print("\nModel training completed successfully!")