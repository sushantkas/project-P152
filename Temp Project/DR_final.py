#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment:Logistic Regression')
st.sidebar.header("User Input Parameter")

def user_ip():
    Age = st.sidebar.number_input("Insert age")
    Systolic_bp = st.sidebar.number_input("Insert Systolic BP")
    Diastolic_bp = st.sidebar.number_input("Insert Diastolic_bp")
    Cholesterol = st.sidebar.number_input("Insert Cholesterol")
    data={'Age':Age,'Systolic_bp':Systolic_bp,'Diastolic_bp':Diastolic_bp,'Cholesterol':Cholesterol}
    features = pd.DataFrame(data,index=[0])
    return features
df=user_ip()
st.subheader("User input parameters")
st.write(df)

DR = pd.read_csv("C:/Users/PANCHI/Downloads/pronostico_dataset.csv",sep=";")
DR.drop(["ID"],inplace=True,axis=1)
DR1 =pd.get_dummies(DR,columns=['prognosis'])
DR2=DR1.drop(['prognosis_retinopathy'],axis=1)

X = DR2.drop('prognosis_no_retinopathy',axis=1)
Y=DR2['prognosis_no_retinopathy']
logreg=LogisticRegression()
logreg.fit(X,Y)

pred = logreg.predict(df)
pred_prob=logreg.predict_proba(df)

st.subheader("prediction Result")
st.write('Yes' if pred_prob[0][1]>0.5 else 'No')

st.subheader('Prediction Probablity')
st.write(pred_prob)


# In[ ]:




