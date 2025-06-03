import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_model, make_prediction

# Load model
model = load_model()

st.set_page_config(page_title="Food Demand Predictor", layout="centered")
st.title("🥗 Food Demand Prediction")
st.write("Enter meal and center details to predict number of orders:")

# Prediction History Setup
if "history" not in st.session_state:
    st.session_state.history = []

# --- 📥 Input Form ---
with st.form("prediction_form"):
    center_id = st.number_input("Center ID", min_value=1, value=10)
    meal_id = st.number_input("Meal ID", min_value=1, value=1885)
    checkout_price = st.number_input("Checkout Price", value=250.0)
    base_price = st.number_input("Base Price", value=300.0)
    emailer_for_promotion = st.selectbox("Email Promotion", [0, 1])
    homepage_featured = st.selectbox("Homepage Featured", [0, 1])
    meal_popularity = st.slider("Meal Popularity Score", 0, 500, 100)
    center_popularity = st.slider("Center Popularity Score", 0, 500, 100)

    submitted = st.form_submit_button("Predict")

# --- 🧠 Prediction Logic ---
if submitted:
    discount = base_price - checkout_price
    discount_pct = round((discount / base_price), 2) if base_price > 0 else 0
    any_promo = 1 if emailer_for_promotion or homepage_featured else 0

    input_data = {
        'center_id': center_id,
        'meal_id': meal_id,
        'checkout_price': checkout_price,
        'base_price': base_price,
        'emailer_for_promotion': emailer_for_promotion,
        'homepage_featured': homepage_featured,
        'discount': discount,
        'discount_pct': discount_pct,
        'any_promo': any_promo,
        'meal_popularity': meal_popularity,
        'center_popularity': center_popularity
    }

    prediction = make_prediction(model, input_data)
    st.success(f"📦 Predicted Demand: {prediction} orders")

    # Save to session state history
    st.session_state.history.append((input_data, prediction))

# --- 📊 Feature Importance ---
st.subheader("🔍 Feature Importance (Model Insights)")
try:
    feature_names = model.feature_names_in_
except AttributeError:
    feature_names = list(input_data.keys())

importances = model.feature_importances_

fig, ax = plt.subplots(figsize=(8, 4))
pd.Series(importances, index=feature_names).sort_values().plot(kind='barh', ax=ax)
ax.set_title("Top Feature Importance")
st.pyplot(fig)

# --- 🧾 Prediction History ---
if st.session_state.history:
    st.subheader("📘 Prediction History")
    for i, (inp, pred) in enumerate(st.session_state.history[::-1]):
        st.markdown(f"**#{len(st.session_state.history)-i}:** `{pred} orders` for `{inp}`")
