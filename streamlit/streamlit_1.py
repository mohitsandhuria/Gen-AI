import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib


name=st.text_input("enter your name")
if name:
    st.title(f"Hello {name},")
else:
    st.title("Hello ,")
st.write("Enter Your Age:")
quantity=st.slider("Select Minimum Quantity:", 0,10,2)
if quantity:
    st.write(f"your Quantity is {quantity}")
else:
    st.write("select your Quantity.")

gender=st.selectbox("Select the Gender",["Male","Female"])
st.write(f"Your Gender is {gender}")

data=pd.read_csv("retail_sales_dataset.csv")
print(data)
data = data[(data["Quantity"] == quantity) & (data["Gender"] == gender)]
st.write(data)

data1=st.file_uploader("Upload a file","csv")
try:
    data1=pd.read_csv(data1)
    st.write(data1)
except Exception as e:
    st.write("Upload a file to get a dataframe")