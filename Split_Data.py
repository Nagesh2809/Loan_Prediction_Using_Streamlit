
# python Split_Data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from EDA import PROCESS
from sklearn.model_selection import train_test_split


from Read_File import READ
from EDA import PROCESS

# data=READ().File('loan_train.csv')
# df=PROCESS().Data_Analysis(data)


class SPLIT:
    def sp(self,df):
        print('Entered in split_class')
        
        X=df.drop('Loan_Status',axis=1)
        y=df['Loan_Status']

        X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=8)

        print('Spliting is complted')
        return(X_train, X_test, y_train, y_test)


if __name__=='__main__':
    data=READ().File('loan_train.csv')
    df=PROCESS().Data_Analysis(data)
    SPLIT().sp(df)

