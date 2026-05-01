from fastapi import FastAPI
import joblib

app = FastAPI()

# load trained model
model = joblib.load("models/house_price_model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    # order must match training features
    features = [[
        data["LotArea"],
        data["OverallQual"],
        data["YearBuilt"],
        data["GrLivArea"],
        data["FullBath"],
        data["GarageCars"]
    ]]

    prediction = model.predict(features)[0]

    return {
        "predicted_price": prediction
    }