
# python Train_And_Save_Model.py

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import pickle as pk 
import joblib as jb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc,precision_score,recall_score,f1_score,accuracy_score,roc_curve

from Read_File import READ
from EDA import PROCESS
from Split_Data import SPLIT

# data=READ().File('loan_train.csv')
# df=PROCESS().Data_Analysis(data)
# X_train, X_test, y_train, y_test=SPLIT().sp(df)


class TRAIN:
    def train_model(self,X_train, X_test, y_train, y_test) :

        print('Model trainning is started')

        lg=LogisticRegression()
        lg.fit(X_train,y_train)

        y_pred=lg.predict(X_test)
        y_proba=lg.predict_proba(X_test)[:,1]

        # AUC-ROC Curve
        fpr,tpr,threshold=roc_curve(y_test,y_proba)
        plt.plot(fpr,tpr,'Blue',label='Logistic_Regresssion')
        plt.plot([0,1],[0,1],'Red',label='Random_Model')
        plt.title('AUC_ROC_Curve')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.legend()
        plt.show()

        # Evolution Metrics
        lg_acc=accuracy_score(y_test,y_pred)
        lg_f1=f1_score(y_test,y_pred)
        lg_pre=precision_score(y_test,y_pred)
        lg_reca=recall_score(y_test,y_pred)
        AUC=auc(fpr,tpr)

        li=[lg_acc,lg_f1,lg_pre,lg_reca,AUC]
        l2=['Accuracy','F1_score','Precision','Recall','AUC']
        df1=pd.DataFrame(li,index=l2,columns=['Logistic_Regresssion'])
        # print(df1)

        pk.dump(lg,open('loan1.pkl','wb'))
        # jb.dump(lg,'loan2.joblib')

        print('Model Saved in Directory')




if __name__=='__main__':
    data=READ().File('loan_train.csv')
    df=PROCESS().Data_Analysis(data)
    X_train, X_test, y_train, y_test=SPLIT().sp(df)
    TRAIN().train_model(X_train, X_test, y_train, y_test)