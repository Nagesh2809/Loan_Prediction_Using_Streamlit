# pip install streamlit
# streamlit run app1.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

from EDA_TEST import EDA22

# Load the trained model
model = joblib.load('loan1.pkl')
lb = joblib.load('label_encoders.pkl')
ss = joblib.load('scaler.pkl')

# Configure the app
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "About"])

# Main Section
if page == "Single Prediction":
    # Title
    st.title("Loan Prediction - Single Prediction")
    
    # Input fields for loan application details
    st.subheader("Enter Applicant Details:")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])   #.str.extract('(\d+)' )
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self-Employed", ["Yes", "No"])
    applicant_income = int(st.number_input("Applicant Income", min_value=0))
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = int(st.number_input("Loan Amount", min_value=0))
    loan_amount_term = st.selectbox("Loan Amount Term (in days)", [360.0, 120.0, 240.0, 180.0, 60.0, 300.0, 480.0, 36.0, 84.0, 12.0])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    # Encode the categorical data
    # make a dataframe for storing values in dataframe 

    data=[gender,married,dependents,education,self_employed ,applicant_income, coapplicant_income, loan_amount,loan_amount_term,credit_history,property_area]
    col=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area']
    
    df=pd.DataFrame([data],columns=col)

    
    lb=joblib.load('label_encoders.pkl')
    cat_col=lb.keys()
    for i in cat_col:
        df[i]=lb[i].transform(df[i])
    

    # Scale the single prediction continuous data by using training  data set mean and standard deviation
    # # Scale the numerical data (reshaping to 2D array for scaler)
    num_col=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    ss=joblib.load('scaler.pkl')

    st.write(df) # this will show taken output
    df[num_col]=ss.transform(df[num_col])  # Do not use fit here bacause your model alredy fit values when you done scalling on training data


    print(df)
    # Prediction
    if st.button("Predict"):
        st.write(df)  # this will show you transformed data
        prediction = model.predict(df)[0]
        result = "Approved" if prediction == 1 else "Rejected"
        st.success(f"The loan application is **{result}**")

    st.write(df) # for checking purpose

elif page == "Batch Prediction":
    pass
    # Title
    st.title("Loan Prediction - Batch Prediction")
    
    # File Upload
    st.subheader("Upload a CSV File with Loan Applications:")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Display a message
            st.write("File uploaded successfully!")
            
            # Initialize the EDA22 class
            # eda_instance = EDA22()
            
            # Pass the uploaded file to the class method
            predictions,t_df = EDA22().test_eda(uploaded_file)  # Pass uploaded_file directly

            # Display the results
            st.write("Batch Prediction Results:")
            st.dataframe(t_df)

            # Display the results
            st.write("Batch Prediction Results:")
            st.dataframe(predictions)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a CSV file.")


    
elif page == "About":
    # About Page
    st.title("About the Loan Prediction App")
    st.write("""
    This application predicts loan approval status based on applicant details.
    - **Single Prediction**: Enter applicant details manually to get a prediction.
    - **Batch Prediction**: Upload a CSV file containing multiple applications for predictions.
    
    Created with ❤️ using Streamlit.
    """)

# Footer
st.markdown("---")
st.markdown("Developed by [Nagesh]")