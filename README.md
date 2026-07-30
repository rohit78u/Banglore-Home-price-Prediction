

---

# NEST — Bengaluru Property Intelligence

NEST is a Streamlit application that estimates Bengaluru home prices from **location**, **BHK**, **bathrooms**, and **square footage**. It pairs a trained machine-learning model with interactive market analytics.

🔗 **Live Demo:** [Bangalore Home Price Prediction App](https://bangalorehomepriceproject-hdgdndyjrfrxkd3gnnn5ac.streamlit.app/)

---

## 📋 Project Overview

The goal of this project is to build a predictive model that estimates the price of houses in Bengaluru based on real-world housing data.
It demonstrates **end-to-end Machine Learning workflow**, including:

* Data cleaning and feature engineering
* Model training and evaluation
* Web app deployment for user interaction

---

## ⚙️ Methodology

1. **Data Collection:** Dataset obtained from Kaggle with 13,000+ Bengaluru property listings.
2. **Data Cleaning:** Removed null values, standardized location and area formats.
3. **Feature Engineering:** Added `price_per_sqft` and simplified location categories.
4. **Outlier Removal:** Filtered unrealistic BHK/area ratios and price anomalies.
5. **Model Training:** Trained multiple models — Linear Regression, Ridge Regression, and Random Forest.
6. **Model Evaluation:**

   * Best model: **Linear Regression** (R² = 0.84, MAE = 12.5, RMSE = 18.9)
   * Validated using ShuffleSplit cross-validation (scores between 0.82 – 0.86).
7. **Deployment:** Integrated with Streamlit for an interactive prediction interface.

---

## 🧠 Models Used

* **Linear Regression:** Baseline model chosen for its simplicity and interpretability.
* **Ridge Regression:** L2 regularization to reduce overfitting and improve generalization.
* **Random Forest:** Ensemble model tested for non-linear pattern capturing.

---

## Production highlights

* Cached, validated loading of the model, feature schema, and market data.
* Defensive data cleaning for area ranges and incomplete source records.
* Clear error states when a required model artifact or data asset is unavailable.
* Responsive, high-contrast visual interface with accessible labels and interactive charts.
* Informational valuation disclaimer to avoid presenting estimates as formal appraisals.

## 💻 Technologies Used

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **IDE:** Jupyter Notebook

---

## 🚀 How to Run Locally

1. Clone this repository

   ```bash
   git clone https://github.com/your-username/bangalore-home-price.git
   cd bangalore-home-price
   ```
2. Create and activate a virtual environment (recommended), then install dependencies

   ```bash
   python -m pip install -r requirements.txt
   ```
3. Run Streamlit app

   ```bash
   streamlit run app.py
   ```

---

## 📊 Result

* **Final Model:** Linear Regression (best balance between accuracy and simplicity).
* **Example Prediction:**
  For a 2 BHK, 1000 sqft property in *Indira Nagar*, predicted price ≈ **₹193.31 Lakhs**.

---

## 🌐 Live Web App

👉 Try it here: [Bangalore Home Price Prediction App](https://bangalorehomepriceproject-hdgdndyjrfrxkd3gnnn5ac.streamlit.app/)

---



