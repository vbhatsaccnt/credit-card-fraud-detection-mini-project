import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("final_model.pkl", "rb"))

def predict_default(features):
    features = np.array(features).astype(np.float64).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return prediction, probability

def main():
    st.title("Credit Default Prediction")
    
    education_status = ["Graduate School", "University", "High School", "Others"]
    marital_status = ["Married", "Single", "Others"]
    
    payment_status = [
        "Account started that month with a zero balance, and never used any credit",
        "Account had a balance that was paid in full",
        "At least the minimum payment was made, but the entire balance wasn't paid",
        "Payment delay for 1 month",
        "Payment delay for 2 month",
        "Payment delay for 3 month",
        "Payment delay for 4 month",
        "Payment delay for 5 month",
        "Payment delay for 6 month",
        "Payment delay for 7 month",
        "Payment delay for 8 month",   
    ]
    
    st.subheader("Please enter the following information:")
    
    features = {}
    features['EDUCATION'] = st.selectbox("Education", education_status)
    features['MARRIAGE'] = st.selectbox("Marital Status", marital_status)
    features['PAY_1'] = st.selectbox("Payment Status", payment_status)
    
    features['LIMIT_BAL'] = st.number_input("Credit Limit", min_value=0)
    features['AGE'] = st.number_input("Age", min_value=0)
    
    features['BILL_AMT1'] = st.number_input("Bill Amount 1", min_value=0)
    features['BILL_AMT2'] = st.number_input("Bill Amount 2", min_value=0)
    features['BILL_AMT3'] = st.number_input("Bill Amount 3", min_value=0)
    features['BILL_AMT4'] = st.number_input("Bill Amount 4", min_value=0)
    features['BILL_AMT5'] = st.number_input("Bill Amount 5", min_value=0)
    features['BILL_AMT6'] = st.number_input("Bill Amount 6", min_value=0)
    
    features['PAY_AMT1'] = st.number_input("Payment Amount 1", min_value=0)
    features['PAY_AMT2'] = st.number_input("Payment Amount 2", min_value=0)
    features['PAY_AMT3'] = st.number_input("Payment Amount 3", min_value=0)
    features['PAY_AMT4'] = st.number_input("Payment Amount 4", min_value=0)
    features['PAY_AMT5'] = st.number_input("Payment Amount 5", min_value=0)
    features['PAY_AMT6'] = st.number_input("Payment Amount 6", min_value=0)
    
    if st.button("Predict"):
        try:
            features['EDUCATION'] = education_status.index(features['EDUCATION']) + 1
            features['MARRIAGE'] = marital_status.index(features['MARRIAGE']) + 1
            features['PAY_1'] = payment_status.index(features['PAY_1']) - 2
            
            actual_feature_names = ['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            feature_values = [features[i] for i in actual_feature_names]
            
            prediction, probability = predict_default(feature_values)
            if prediction[0] == 1:
                st.error("This account will be defaulted with a probability of {}%.".format(round(np.max(probability)*100, 2)))
            else:
                st.success("This account will not be defaulted with a probability of {}%.".format(round(np.max(probability)*100, 2)))
        except:
            st.error("Please enter relevant information.")

if __name__ == "__main__":
    main()
