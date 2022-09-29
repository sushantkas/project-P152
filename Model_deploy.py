import streamlit as st
import pandas as pd
import numpy as np

st.title("Predicting whether patient has Retinpathy or not Based on the Provided Input")

st.sidebar.radio("Navigation",["Home","Predict","Contributors of Mdoel","Contact Us"])