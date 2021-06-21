import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib

Adm_Data=pd.read_csv("C:\\Users\\satye\\Data_Analytics\\Admission_Predict.csv")
Adm_Data.set_index('Serial No.',inplace=True)
Adm_Data.drop(['GRE Score'],axis=1,inplace=True)

Features=Adm_Data.iloc[:,:-1]
Target=Adm_Data.iloc[:,-1:]

Feature_Train,Feature_Test,Target_Train, Target_Test=train_test_split(Features,Target,test_size=0.20,random_state=42)

lm=LinearRegression()
lm.fit(Feature_Train,Target_Train)

pred=lm.predict(Feature_Test)

joblib.dump(lm,"Admission_Pred.pkl")
