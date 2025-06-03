# 🥗 Food Waste Reduction using AI

A machine learning project to predict food demand and reduce edible food waste in supply chains. Inspired by real-world use cases like Nestlé’s AI food waste reduction trial.

![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/your_username/food-waste-reduction-ml/dashboard/app.py)

---

## 📌 Project Highlights

- ✅ Predicts number of orders based on meal, center, price, and promotion
- ✅ Automatically downloads trained model from Google Drive
- ✅ Built with Random Forest (tuned via GridSearchCV)
- ✅ Streamlit dashboard for real-time prediction and visualizations

---

## 🧠 Features

- Feature importance visualization
- Discount, promo & price analysis
- Historical prediction log
- Model stored externally (not bloating GitHub)

---

## 🚀 Run Locally

```bash
git clone git@github.com:naren43nk/food-waste-reduction-ml.git
cd food-waste-reduction-ml
pip install -r requirements.txt
streamlit run dashboard/app.py
