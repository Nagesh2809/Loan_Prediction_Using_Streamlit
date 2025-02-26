# python EDA_TEST.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib 

class EDA22:

    def test_eda(self,file):
        print('hii')
        t_dff=pd.read_csv(file)
        t_df=t_dff.drop('Loan_ID',axis=1)
        # t_df.head()

        # Fill Missing values
        # t_df.isnull().sum()
        t_df['Gender']=t_df['Gender'].fillna(t_df['Gender'].mode()[0])
        t_df['Dependents']=t_df['Dependents'].fillna(t_df['Dependents'].mode()[0])

        t_df['Self_Employed']=t_df['Self_Employed'].fillna(t_df['Self_Employed'].mode()[0])
        t_df['LoanAmount']=t_df['LoanAmount'].fillna(t_df['LoanAmount'].median())

        t_df['Loan_Amount_Term']=t_df['Loan_Amount_Term'].fillna(t_df['Loan_Amount_Term'].mode()[0])
        t_df['Credit_History']=t_df['Credit_History'].fillna(t_df['Credit_History'].mode()[0])


        # Change the data type of columns

        cat_col=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        for i in cat_col:
            t_df[i]=t_df[i].astype('object')

        t_df['ApplicantIncome']=t_df['ApplicantIncome'].astype('float64')
        t_df['CoapplicantIncome']=t_df['CoapplicantIncome'].astype('float64')

        
        # Do outlier analysis

        df11 = pd.DataFrame()
        df = pd.DataFrame()

        num_col=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        
        for i in num_col:
            q1 = np.percentile(t_df[i], 25)
            q3 = np.percentile(t_df[i], 75)
            IQR = q3 - q1
            lb = q1 - (1.5 * IQR)
            ub = q3 + (1.5 * IQR)

            cond1 = t_df[i] < lb
            cond2 = t_df[i] > ub

            df11[i] = np.where(cond1, lb, t_df[i])
            t_df[i] = np.where(cond2, ub, df11[i])

        # Now i will do encoding and Scalling
            # First i will import the encoding pickle file and standard scalar pickle file
            # which is used for encode scalling of the training data
        
        import joblib as jb
        lb = joblib.load('label_encoders.pkl')
        ss = joblib.load('scaler.pkl')

        # for i in cat_col:
        #     t_df[i]=lb[i].transform(t_df[i])   

        # This code will give error like
        # ValueError: y contains previously unseen labels: 350.0
        # so i will try to convert it into nearest of its label or related
        # it means according to project requirement

        # so i will replace it with previous label which is related (360 to 8)

        for i in cat_col:
            try:
                t_df[i] = lb[i].transform(t_df[i])
            except ValueError:
                # Handle unseen labels by mapping them to a default value
                t_df[i] = t_df[i].apply(lambda x: lb[i].transform([x])[0] if x in lb[i].classes_ else 8)

        # Now i will do Scalling
        t_df[num_col]=ss.transform(t_df[num_col])

        # Now Load the model and Predict
        lg=joblib.load('loan1.pkl')

        yy_pred=pd.DataFrame()

        yy_pred['Loan_ID']=t_dff['Loan_ID']

        yy_pred['Prediction']=lg.predict(t_df)  # Prediction you will give this line
        yy_pred['Prediction']=yy_pred['Prediction'].map({1:'yes',0:'no'})

        t_df['Loan_ID']=t_dff['Loan_ID']

        # print(yy_pred)
        return(yy_pred,t_df)

 


if __name__=='__main__':
    EDA22().test_eda('Loan_test.csv')