# house-price-prediction-ml
House Price Prediction using Machine Learning with FastAPI deployment
🏠 House Price Prediction using Machine Learning

📌 Overview

This project predicts house prices using Machine Learning models based on property features like area, quality, and year built.

It simulates a real-world real estate pricing system used by property platforms and banks.

---

🎯 Problem Statement

House pricing is complex and depends on multiple factors.
This project builds a regression model to estimate house prices automatically.

---

🚀 Features

- Data generation (synthetic dataset)
- Data preprocessing & analysis
- Machine Learning model training (Random Forest)
- Model evaluation (MAE, R² Score)
- Saved model using joblib
- FastAPI deployment
- Real-time prediction API

---

🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

---

📊 Model Used

- Random Forest Regressor

---

📈 Results

- Achieved good accuracy using regression model
- Model predicts house price based on key features

---

▶️ How to Run

1. Create virtual environment

python3 -m venv venv
source venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. Generate dataset

python src/generate_data.py

4. Train model

python src/train_save_model.py

5. Run API

uvicorn src.api:app --reload

---

🌐 API Usage

Endpoint:

"POST /predict"

Sample Input:

{
  "LotArea": 5000,
  "OverallQual": 7,
  "YearBuilt": 2005,
  "GrLivArea": 2000,
  "FullBath": 2,
  "GarageCars": 2
}

🎓 Learning Outcomes

- End-to-end ML pipeline
- Model training & evaluation
- API deployment using FastAPI
- Real-world project structuring

---

💡 Future Improvements

- Add XGBoost model
- Add frontend dashboard
- Use real dataset
- Add SHAP explainability

---

⭐ Author

Arppitha
