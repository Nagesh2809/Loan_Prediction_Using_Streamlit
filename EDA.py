
# python EDA.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pickle 
import joblib 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from Read_File import READ
# data=READ().File('loan_train.csv')   

class PROCESS:
    def Data_Analysis(self,data):

        print('Entered in EDA')
        
        # Find and Drop the duplicates
        # data.duplicated().sum()
        data.drop_duplicates(inplace=True)

        # find the missing values
        # df.isnull().sum()

        # Check data types and change according to data
        # Fill Missing values
        data['Gender']=data['Gender'].fillna(data['Gender'].mode()[0])
        data['Married']=data['Married'].fillna(data['Married'].mode()[0])
        # data['Dependents']=data['Dependents'].str.replace('3+','3')
        data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
        data['Self_Employed']=data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
        data['Self_Employed']=data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
        data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
        data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
        data['LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].median())
        
        data['Dependents']=data['Dependents'].astype('object')
        data['Loan_Amount_Term']=data['Loan_Amount_Term'].astype('object')
        data['Credit_History']=data['Credit_History'].astype('object')

        # 1.Dependents
        # 2.Loan_Amount_Term
        # 3.Credit_History
        # This columns Does not require outlier analysis because of their categorical or ordinal nature.

        num_col=data.select_dtypes(exclude='object').columns

        # Outlier Analysis
        # we fill outliers using Capping method (lower bound and upper bound)


        df11 = pd.DataFrame()
        df = pd.DataFrame()

        for i in num_col:
            q1 = np.percentile(data[i], 25)
            q3 = np.percentile(data[i], 75)
            IQR = q3 - q1
            lb = q1 - (1.5 * IQR)
            ub = q3 + (1.5 * IQR)

            cond1 = data[i] < lb
            cond2 = data[i] > ub
            # cond3 = cond1 | cond2

            df11[i] = np.where(cond1, lb, data[i])
            data[i] = np.where(cond2, ub, df11[i])

        df=data.drop(['Loan_ID','Loan_Status'],axis=1)
        cat_col=df.select_dtypes(include='object').columns
        num_col=df.select_dtypes(exclude='object').columns

        # Encoding of Categorical columns   
        # save encoder with column labels for  prediction file
        
        label_encoders={}
        
        # lb=LabelEncoder()   # Do not use it outside use it inside forloop  
        for i in cat_col:
            lb=LabelEncoder()   # use it outside use it inside for loop because encoder is associated with each column
            df[i]=lb.fit_transform(df[i])
            label_encoders[i]=lb    
        
        joblib.dump(label_encoders, 'label_encoders.pkl')    #Here your saved encoder with column labels


        # Scalling 
        # Save encoder for data which require Sacalling

        ss=StandardScaler()
        df[num_col]=ss.fit_transform(df[num_col])

        joblib.dump(ss,'scaler.pkl')   # Here you saved  StandardScaler for prediction file


        
        # Add output column to dataframe
        df['Loan_Status']=data['Loan_Status'].map({'Y':1,'N':0})


        # print(data.head())
        # pass
        print('EDA complted')
        print(cat_col,num_col)
        return df



if __name__=='__main__':
    data=READ().File('loan_train.csv') 
    PROCESS().Data_Analysis(data)
    print('EDA complted')