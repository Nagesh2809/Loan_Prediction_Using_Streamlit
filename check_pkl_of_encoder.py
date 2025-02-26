
# python check_pkl_of_encoder.py


import joblib

# Load the trained model
model = joblib.load('loan1.pkl')
lb = joblib.load('label_encoders.pkl')
ss = joblib.load('scaler.pkl')

cat_col=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Loan_Amount_Term', 'Credit_History', 'Property_Area']
    
# print(lb.keys())
for i in cat_col:
    print(lb[i].classes_)