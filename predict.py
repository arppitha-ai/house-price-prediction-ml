import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# load dataset
df = pd.read_csv("data/houses.csv")

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# train model again (simple for testing)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# NEW HOUSE DATA (you are testing AI)
new_house = [[
    5000,   # LotArea
    7,      # OverallQual
    2005,   # YearBuilt
    2000,   # GrLivArea
    2,      # FullBath
    2       # GarageCars
]]

# prediction
price = model.predict(new_house)

print("🏠 Predicted House Price:", price[0])