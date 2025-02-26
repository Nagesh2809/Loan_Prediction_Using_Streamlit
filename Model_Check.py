
# python Model_Check.py


import pickle as pk
import joblib as jb
import warnings
warnings.filterwarnings('ignore')

lg=pk.load(open('loan1.pkl','rb'))

# Single Prediction
print(lg.predict([[1,1,1,2,1,5849.0,1508.0,128.0,1,2,1]]))

# Batch Prediction
print(lg.predict([[1,1,1,2,1,5849.0,1508.0,128.0,1,2,1],[1,1,1,2,1,5849.0,1508.0,128.0,1,2,1]]))