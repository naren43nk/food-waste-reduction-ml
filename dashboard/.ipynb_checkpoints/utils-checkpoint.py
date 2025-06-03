# dashboard/utils.py

import joblib
import pandas as pd

# Load the model
def load_model():
    model = joblib.load('../models/rf_demand_forecast_tuned.pkl')
    return model

# Predict from form input
def make_prediction(model, input_dict):
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]
    return round(prediction)
