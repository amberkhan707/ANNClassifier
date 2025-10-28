<img width="913" height="706" alt="image" src="https://github.com/user-attachments/assets/388c09ed-cdfe-425a-9d5c-916da6f31987" />
# 💹 Customer Churn Prediction App

## 🚀 Overview

This Streamlit web app predicts whether a bank customer is likely to leave (churn) or stay based on key demographic and financial details. It uses a trained Artificial Neural Network (ANN) model built with TensorFlow/Keras, along with pre-trained encoders and a scaler for preprocessing user input before prediction.

## 🧠 Tech Stack

| Category | Technology |
|----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| Machine Learning | TensorFlow / Keras |
| Data Handling | Pandas |
| Model Persistence | Pickle |
| Styling | Custom HTML + CSS |

## 📦 Project Structure
Customer_Churn_Prediction/
│
├── ANNModel.h5 # Trained ANN model
├── gender.pkl # LabelEncoder for Gender
├── geography.pkl # OneHotEncoder for Geography
├── scaler.pkl # StandardScaler for numeric features
├── app.py # Streamlit application code
├── requirements.txt # List of dependencies
└── README.md # Project documentation


## ⚙️ Installation

### 1️⃣ Clone the repository
bash
git clone https://github.com/amberkhan707/ANNClassifier.git
cd ANNClassifier


### 2️⃣ Create a virtual environment
bash
python -m venv venv
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On macOS/Linux



### 3️⃣ Install dependencies
bash
pip install -r requirements.txt
### 📂 Requirements
Here's an example requirements.txt file for your app:


text
streamlit
tensorflow
pandas
numpy
scikit-learn
pickle-mixin
## ▶️ Run the App
Once everything is installed, run:


bash
streamlit run app.py
Then open the displayed local URL (usually http://localhost:8501/) in your browser.



## 🧩 Input Features
Feature	Description	Example
Geography	Customer's country	France / Germany / Spain
Gender	Male or Female	Male
Credit Score	Credit rating of customer	600
Age	Age of customer	35
Tenure	Years with the bank	5
Balance	Bank account balance	120000.00
NumOfProducts	Number of bank products	2
HasCrCard	Has a credit card (1=yes, 0=no)	1
IsActiveMember	Active customer (1=yes, 0=no)	1
EstimatedSalary	Annual salary	50000.00
### 📊 Output
After clicking "🔍 Predict", the app displays:


✅ Not Likely to Leave → Customer is predicted to stay


🚨 Likely to Leave → Customer is predicted to churn


Each result also includes the predicted probability.



## 🎨 UI Design
The interface is styled using custom CSS inside the Streamlit app:

Dark theme for better visual contrast

Color-coded results (green for safe, red for churn risk)

Emoji-enhanced labels for better user experience



## 🧮 How It Works
User inputs customer details in the Streamlit form

Model encoders transform categorical features

gender.pkl → LabelEncoder for Gender

geography.pkl → OneHotEncoder for Geography

Numerical features are standardized using scaler.pkl

Preprocessed input is passed to the trained ANN model (ANNModel.h5)

Model outputs churn probability → displayed as a result



## 🧪 Example Prediction
Input	Value
Geography	France
Gender	Female
CreditScore	600
Age	40
Tenure	3
Balance	60000
NumOfProducts	2
HasCrCard	1
IsActiveMember	1
EstimatedSalary	50000


## ➡️ Output:
✅ Customer is NOT LIKELY TO LEAVE (Probability: 72.45%)


## 🧰 Model Information
Type: Artificial Neural Network (Feedforward ANN)

Framework: TensorFlow / Keras

Training Dataset: Bank Customer Churn Dataset

Output Activation: Sigmoid (Binary Classification)

Loss Function: Binary Crossentropy

Optimizer: Adam


