import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb


from sklearn.model_selection import train_test_split, GridSearchCV


# -----------------------------
# Page Title
# -----------------------------
st.set_page_config(page_title="Predictive Maintenance for Gas Engine", layout="wide")
st.title("Predictive Maintenance for Gas Engine")
st.write(
    "This app uses engine data to predict whether a failure will occur."
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("engine_data.csv")
    return data

data = load_data()











# -----------------------------
# Drop unnecessary columns
# -----------------------------
# data = data.drop(columns=["UDI", "Product ID"], errors="ignore")

# -----------------------------
# Define target and features
# -----------------------------
y = data["Engine_Condition"]
X = data.drop(columns=["Engine_Condition"], errors="ignore")

# Convert categorical columns
# X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# Scale data
# -----------------------------
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss", 
    random_state=42
)
grid_search = GridsearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(x_train_scaled, y_train)

model = grid_search.best_estimator_
              

# -----------------------------
# Predict and Evaluate
# -----------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")














# -----------------------------
# User Input Section
# -----------------------------
st.subheader("Try a Prediction")

# type_option = st.selectbox("Type", ["L", "M", "H"])
Lub_oil_pressure = st.number_input("Oil Pressure[Bar] (Max 7.27)", value=3.5)
lub_oil_temp = st.number_input("Oil Temperature[Bar] (Max 89.59)", value=45)
Coolant_temp = st.number_input("Coolant Temperature[C] (Max 195.53)", value=100)
Fuel_pressure = st.number_input("Fuel Pressure[Bar] (Max 21.14)", value=11)
Coolant_pressure = st.number_input("Coolant Pressure[Bar] (Max 7.48)", value=3.5)
Engine_rpm = st.number_input("Rpm (Max 2239)", value=1000)


input_df = pd.DataFrame({
    
    "Oil Temperature[Bar] (Max 89.59)": [lub_oil_temp],
    "Coolant Temperature[C] (Max 195.53)": [Coolant_temp],
    "Oil Pressure[Bar] (Max 7.27)": [Lub_oil_pressure],
    "Fuel Pressure[Bar] (Max 21.14)": [Fuel_pressure],
    "Coolant Pressure[Bar] (Max 7.48)": [Coolant_pressure],
    "Rpm (Max 2239)": [Engine_rpm]
})

# Match training columns after get_dummies
input_df = pd.get_dummies(input_df, drop_first=True)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
prediction_prob = model.predict_proba(input_scaled)[0][1]

if st.button("Predict Failure"):
    st.write(f"Prediction: {'Failure' if prediction == 1 else 'No Failure'}")
    st.write(f"Probability of Failure: {prediction_prob * 100:.2f}%")


