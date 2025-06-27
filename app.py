import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# Simple session-based authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Login credentials (for demo purpose)
USER_CREDENTIALS = {"admin": "pass"}

# Login page
if not st.session_state.authenticated:
    st.set_page_config(page_title="Login", layout="centered")
    st.title("🔐 Login to Fraud Detection App")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if USER_CREDENTIALS.get(username) == password:
            st.session_state.authenticated = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid credentials.")
    st.stop()

# Logout button
if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.rerun()

# Load model and scaler
model = joblib.load('fraud_xgboost_model.pkl')
scaler = joblib.load('fraud_scaler.pkl')

# Set page config with custom icon
st.set_page_config(page_title="Fraud Detection App", layout="centered", page_icon="🔒")

# Add animated full background using CSS
st.markdown(
    """
    <style>
    @keyframes moveBackground {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #1a2a6c, #b21f1f, #fdbb2d, #0f2027);
        background-size: 400% 400%;
        animation: moveBackground 20s ease infinite;
        color: white;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Segoe UI', sans-serif;
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add logo or image
st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=80)
st.title("🔍 Real-Time Fraud Detection")

st.markdown("""
Use this tool to analyze transactions and detect **potential fraud** using a trained XGBoost model.
Simply input transaction details and hit **Detect Fraud**!
""")

# Input form
with st.form("fraud_form"):
    st.subheader("📥 Transaction Input")
    CUSTOMER_ID = st.number_input("Customer ID", min_value=0, value=1000)
    TERMINAL_ID = st.number_input("Terminal ID", min_value=0, value=2000)
    TX_AMOUNT = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    TX_TIME_SECONDS = st.number_input("Transaction Time (seconds)", min_value=0, max_value=86400, value=36000)
    TX_TIME_DAYS = st.number_input("Transaction Day", min_value=0, value=100)

    submitted = st.form_submit_button("🚨 Detect Fraud")

if submitted:
    input_data = pd.DataFrame({
        'CUSTOMER_ID': [CUSTOMER_ID],
        'TERMINAL_ID': [TERMINAL_ID],
        'TX_AMOUNT': [TX_AMOUNT],
        'TX_TIME_SECONDS': [TX_TIME_SECONDS],
        'TX_TIME_DAYS': [TX_TIME_DAYS],
        'TX_FRAUD_SCENARIO': [0]  # Dummy value to satisfy model input
    })

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected!\nProbability: {probability:.4f}")
        st.image("https://cdn-icons-png.flaticon.com/512/1828/1828843.png", width=80)
    else:
        st.success(f"✅ Legitimate Transaction.\nProbability of fraud: {probability:.4f}")
        st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=80)
