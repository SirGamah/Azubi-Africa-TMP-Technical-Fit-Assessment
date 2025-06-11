# Azubi-Africa-TMP-Technical-Fit-Assessment
Delivarables for the Azubi Africa Talent Mobility Program (TMP) Technical Fit Assessment
# 💰 Term Deposit Subscription Prediction App

A Streamlit-based web application that analyzes, models, and predicts whether a bank client will subscribe to a term deposit offer based on their demographic, contact, campaign, and socioeconomic attributes.

## 📌 Project Objective

This app helps banks and marketers:
- Understand client patterns using interactive visual analytics
- Train and evaluate machine learning models for predicting term deposit subscriptions
- Make real-time predictions for new client profiles

---

## 📊 Dataset

`bank-additional-full.csv` with all examples (41188) and 20 inputs was used as it provides a wide range of inputs for analysis and model development. This includes features like:

- **Personal & Socioeconomic Attributes:** Age, job, marital status, education, etc.
- **Last Contact Info:** Communication type and timing
- **Campaign Details:** Number of contacts and past outcomes
- **Economic Context:** Employment and interest rates, inflation, etc.
- **Target Variable (`y`)**: Indicates whether the client subscribed to a term deposit (`yes` or `no`)
- etc

---

## 🚀 App Features

### 🧭 Overview
- Introduction to the dataset and project
- Business objective description
- Initial EDA

### 📈 Analysis
- Dropdown menu to select different feature groups for visualization
- Interactive charts built with **Plotly Express**
- Key insights with **business interpretations** and **recommendations**

### 🤖 Train Model
- Select features using checkboxes
- Choose from 5 classification models
- View:
  - Confusion matrix
  - Accuracy, precision, recall, F1 score
  - Feature importance plot
- Download trained model as `.pkl` file

### 🧠 Make Prediction
- Input values for a new client using dropdowns and sliders
- Predict whether they will subscribe
- Display model performance summary and predicted class

### ℹ️ About
- App summary
- Developer contact details

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Plotly Express** (visualizations)
- **scikit-learn** (modeling)
- **Pandas / NumPy** (data processing)
- **joblib / pickle** (model persistence)

---

## 📂 How to Run the App

1. Clone the repository:

```bash
git clone https://github.com/your-username/term-deposit-predictor.git
cd term-deposit-predictor
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Launch the Streamlit app:
```bash
streamlit run app.py
```
