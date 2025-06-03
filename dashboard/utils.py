import os
import joblib
import pandas as pd
import urllib.request

# Local path to save the model
MODEL_PATH = '../models/rf_demand_forecast_tuned.pkl'

# Direct download link from Google Drive
DRIVE_MODEL_URL = 'https://drive.google.com/uc?export=download&id=1DFIEchkQYQGkUdcJQLu1MEBCfMlwzYJv'

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("📥 Downloading model from Google Drive...")
        urllib.request.urlretrieve(DRIVE_MODEL_URL, MODEL_PATH)
        print("✅ Model downloaded successfully.")

def load_model():
    download_model()
    model = joblib.load(MODEL_PATH)
    return model

def make_prediction(model, input_dict):
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]
    return round(prediction)
