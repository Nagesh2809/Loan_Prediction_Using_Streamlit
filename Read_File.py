
# python Read_File.py

import pandas as pd

class READ:
    def File(self,file):
       
       df= pd.read_csv(file)
       print('\nReading of File Successfully Done')

       return(df)
    
    
    
if __name__=='__main__':
    READ().File('loan_train.csv')
    