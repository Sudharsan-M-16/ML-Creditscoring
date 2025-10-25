import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
st.set_page_config(
    page_title="Credit Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'credit_default_model.pkl'
SCALER_PATH = BASE_DIR / 'scaler_model.pkl'
if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error("Model or Scaler not found. Please run 'pro.py' first to train and save them.")
    st.stop()
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
st.title("Credit Default Risk Prediction")
st.subheader("Predict whether a customer will default on their next credit card payment")
st.divider()
with st.sidebar:
    st.header("Customer Details")
    st.divider()
    st.subheader("Financial Information")
    LIMIT_BAL = st.slider('Credit Limit', 10000, 1000000, 50000)
    AGE = st.slider('Age', 18, 100,30)
    st.divider()
    st.subheader("Demographic Information")
    col1, col2 = st.columns(2)
    with col1:
        SEX = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        MARRIAGE = st.selectbox("Marital Status", ["Married", "Single", "Others"])
    
    EDUCATION = st.selectbox("Education Level", 
                            ["Graduate School", "University", "High School", "Others"],
                            help="Highest level of education completed")
    
    st.divider()
    st.subheader("Payment History")
    st.caption("Payment status for the past 6 months")
    st.caption("(-2=No consumption, -1=Paid in full, 0=Revolving credit, 1-8=Months of delay)")
    with st.expander("Recent Months(Sep-July)"):
        PAY_0=st.slider('Payment Status September',-2,8,0)
        PAY_2=st.slider('Payment Status August',-2,8,0)
        PAY_3=st.slider('Payment Status July',-2,8,0)
    with st.expander("Earlier Months(June-April)"):
        PAY_4=st.slider('Payment Status June',-2,8,0)
        PAY_5=st.slider('Payment Status May',-2,8,0)
        PAY_6=st.slider('Payment Status April',-2,8,0)
    st.divider()
    st.subheader("Bill Statements")
    with st.expander("Bill Amounts (Past 6 months)"):
        BILL_AMT1 = st.number_input('September Bill (USD)', -5000, 1000000, 50000, step=1000)
        BILL_AMT2 = st.number_input('August Bill (USD)', -5000, 1000000, 50000, step=1000)
        BILL_AMT3 = st.number_input('July Bill (USD)', -5000, 1000000, 50000, step=1000)
        BILL_AMT4 = st.number_input('June Bill (USD)', -5000, 1000000, 50000, step=1000)
        BILL_AMT5 = st.number_input('May Bill (USD)', -5000, 1000000, 50000, step=1000)
        BILL_AMT6 = st.number_input('April Bill (USD)', -5000, 1000000, 50000, step=1000)
    st.divider()
    st.subheader("Payment Amounts")
    with st.expander("Payment Amounts (Past 6 Months)"):
        PAY_AMT1 = st.number_input('September Payment (USD)', 0, 1000000, 10000, step=1000)
        PAY_AMT2 = st.number_input('August Payment (USD)', 0, 1000000, 10000, step=1000)
        PAY_AMT3 = st.number_input('July Payment (USD)', 0, 1000000, 10000, step=1000)
        PAY_AMT4 = st.number_input('June Payment (USD)', 0, 1000000, 10000, step=1000)
        PAY_AMT5 = st.number_input('May Payment (USD)', 0, 1000000, 10000, step=1000)
        PAY_AMT6 = st.number_input('April Payment (USD)', 0, 1000000, 10000, step=1000)
    st.divider()
    predict_button = st.button("Predict Default Risk", use_container_width=True, type="primary")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Summary")
    
    summary_data = {
        "Category": ["Financial", "Demographics", "Recent Payment Status"],
        "Details": [
            f"Credit Limit: ${LIMIT_BAL:,} | Age: {AGE}",
            f"{SEX} | {MARRIAGE} | {EDUCATION}",
            f"Sep: {PAY_0} | Aug: {PAY_2} | Jul: {PAY_3}"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

with col2:
    st.subheader("Model Information")
    model_info = pd.DataFrame({
        "Metric": ["Algorithm", "Accuracy", "ROC-AUC Score"],
        "Value": ["Random Forest", "~81%", "~0.77"]
    })
    st.dataframe(model_info, use_container_width=True, hide_index=True)


if predict_button:
    SEX_encoded = 1 if SEX == "Male" else 2
    
    if EDUCATION == "Graduate School":
        EDUCATION_encoded = 1
    elif EDUCATION == "University":
        EDUCATION_encoded = 2
    elif EDUCATION == "High School":
        EDUCATION_encoded = 3
    else:
        EDUCATION_encoded = 4
    
    if MARRIAGE == "Married":
        MARRIAGE_encoded = 1
    elif MARRIAGE == "Single":
        MARRIAGE_encoded = 2
    else:
        MARRIAGE_encoded = 3
    
    data = {
        'LIMIT_BAL': LIMIT_BAL,
        'SEX': SEX_encoded,
        'EDUCATION': EDUCATION_encoded,
        'MARRIAGE': MARRIAGE_encoded,
        'AGE': AGE,
        'PAY_0': PAY_0,
        'PAY_2': PAY_2,
        'PAY_3': PAY_3,
        'PAY_4': PAY_4,
        'PAY_5': PAY_5,
        'PAY_6': PAY_6,
        'BILL_AMT1': BILL_AMT1,
        'BILL_AMT2': BILL_AMT2,
        'BILL_AMT3': BILL_AMT3,
        'BILL_AMT4': BILL_AMT4,
        'BILL_AMT5': BILL_AMT5,
        'BILL_AMT6': BILL_AMT6,
        'PAY_AMT1': PAY_AMT1,
        'PAY_AMT2': PAY_AMT2,
        'PAY_AMT3': PAY_AMT3,
        'PAY_AMT4': PAY_AMT4,
        'PAY_AMT5': PAY_AMT5,
        'PAY_AMT6': PAY_AMT6
    }
    
    input_df = pd.DataFrame(data, index=[0])
    input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    with st.spinner('Analyzing credit risk...'):
        prediction = model.predict(input_df_scaled)
        probability = model.predict_proba(input_df_scaled)[0][1]
    st.divider()
    st.subheader("Prediction Result")
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        if prediction == 1:
            st.error("**HIGH RISK OF DEFAULT**")
            st.metric("Risk Probability", f"{probability*100:.2f}%",delta=f"{50-probability*100:.2f}% from baseline", delta_color="inverse")
        else:
            st.success("**LOW RISK OF DEFAULT**")
            st.metric("Risk Probability", f"{probability*100:.2f}%",delta=f"{50-probability*100:.2f}% below baseline", delta_color="normal")
    with col3:
        st.info("This prediction is based on the customer's financial history and behavior.")
    st.subheader("Risk Assessment")
    risk_pecentage = probability*100
    if risk_pecentage < 30:
        risk_level="Low Risk"
        risk_color="normal"
    elif risk_pecentage < 60:
        risk_level="Medium Risk"
        risk_color="normal"
    else:
        risk_level="High Risk"
        risk_color="inverse"
    st.progress(probability,f"{risk_level} - Default Probability:{risk_pecentage:.2f}%")
    
    st.divider()
    st.subheader("Financial Analysis")
    col1,col2,col3=st.columns(3)
    with col1:
        avg_bill = (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6) / 6
        st.metric("Average Monthly Bill", f"${avg_bill:.0f}")
    with col2:
        avg_payment = (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6) / 6
        st.metric("Average Monthly Payment", f"${avg_payment:.0f}")
    with col3:
        utilization=(avg_bill/LIMIT_BAL *100) if LIMIT_BAL>0 else 0
        st.metric("Credit Utilization", f"{utilization:.1f}%")
    st.divider()
    st.subheader("Payment Behaviour Summary")
    payment_history=[PAY_0,PAY_2,PAY_3,PAY_4,PAY_AMT5,PAY_6]
    on_time=sum(1 for p in payment_history if p<=0)
    delay=sum(1 for p in payment_history if p>0)
    maxdelay=max(payment_history)
    behavior_data = pd.DataFrame({
        "Metric": ["On-time/Early Payments", "Delayed Payments", "Maximum Delay (months)", "Payment Consistency"],
        "Value": [
            f"{on_time} out of 6 months",
            f"{delay} out of 6 months",
            f"{maxdelay} month(s)" if maxdelay > 0 else "None",
            "Good" if delay == 0 else "Fair" if delay <= 2 else "Poor"
        ]
    })
    st.dataframe(behavior_data, use_container_width=True, hide_index=True)
    st.divider()
    
    