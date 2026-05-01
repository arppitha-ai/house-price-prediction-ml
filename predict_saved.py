import joblib

# load saved model
model = joblib.load("models/house_price_model.pkl")

# new house
house = [[5000, 7, 2005, 2000, 2, 2]]

# prediction
price = model.predict(house)

print("🏠 Predicted Price:", price[0])