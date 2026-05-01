import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# load dataset
df = pd.read_csv("data/houses.csv")

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# prediction
pred = model.predict(X_test)

# evaluation
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# SAVE MODEL (IMPORTANT)
joblib.dump(model, "models/house_price_model.pkl")

print("\nModel saved successfully in models/ folder")