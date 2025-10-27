import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Churn Prediction App", page_icon="💹", layout="centered")

# Custom CSS for beautification
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #FAFAFA;
        }
        .main {
            background-color: #161a23;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #4FC3F7;
        }
        .result {
            text-align: center;
            padding: 1.5rem;
            border-radius: 10px;
            font-size: 1.2rem;
            font-weight: bold;
        }
        .success {
            background-color: #004d40;
            color: #A5D6A7;
        }
        .danger {
            background-color: #3e2723;
            color: #EF9A9A;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("Customer Churn Prediction")

st.markdown("### Enter Customer Details:")

# -------------------- INPUT FIELDS --------------------
col1, col2 = st.columns(2)

with col1:
    Geography = st.selectbox('🌍 Geography', ['France', 'Germany', 'Spain'])
    Gender = st.selectbox('⚧ Gender', ['Male', 'Female'])
    CreditScore = st.number_input('💳 Credit Score', min_value=300, max_value=850, step=1)
    Age = st.slider('🎂 Age', 18, 70, 30)
    Tenure = st.slider('📅 Tenure (years)', 1, 10, 5)

with col2:
    Balance = st.number_input('💰 Balance', min_value=0.0)
    NumOfProducts = st.slider('🛍 Number of Products', 1, 5, 1)
    HasCrCard = st.selectbox('💳 Has Credit Card?', [1, 0])
    IsActiveMember = st.selectbox('✅ Is Active Member?', [1, 0])
    EstimatedSalary = st.number_input('💼 Estimated Salary', min_value=0.0)

# -------------------- LOAD MODEL & ENCODERS --------------------
model = load_model('ANNModel.h5')

with open('gender.pkl', 'rb') as file:
    gender = pickle.load(file)
with open('geography.pkl','rb') as file:
    geography = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# -------------------- PREDICTION --------------------
if st.button('🔍 Predict'):
    input_data = pd.DataFrame([{
        "CreditScore": CreditScore,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary
    }])

    # Encoding
    input_data['Gender'] = gender.transform(input_data['Gender'])
    location = geography.transform([[Geography]])
    location = pd.DataFrame(location, columns=geography.get_feature_names_out(['Geography']))
    location.rename(columns={
        'Geography_France':'France', 
        'Geography_Germany':'Germany', 
        'Geography_Spain':'Spain'
    }, inplace=True)

    input_data = pd.concat([input_data.reset_index(drop=True), location], axis=1)
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prob = prediction[0][0]

    # -------------------- RESULT DISPLAY --------------------
    if prob < 0.5:
        st.markdown(f"<div class='result danger'>🚨 Customer is **LIKELY TO LEAVE** the bank<br>Probability: **{prob:.2%}**</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result success'>✅ Customer is **NOT LIKELY TO LEAVE**<br>Probability: **{prob:.2%}**</div>", unsafe_allow_html=True)
