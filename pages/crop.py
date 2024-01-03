# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from PIL import Image


st.set_page_config(page_title="Crop Recommendation System", page_icon=":ear_of_rice:", layout="wide", initial_sidebar_state="expanded")


# Reading Data
df = pd.read_csv('crop.csv')

# Preprocessing Data
df.dropna(inplace=True)

# Splitting Data into Train and Test Sets
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Function to Predict Crop
def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    inputs = [[n, p, k, temperature, humidity, ph, rainfall]]
    prediction = gnb.predict(inputs)[0]
    return prediction

# Streamlit App
# Display the filtered data
st.title(" 🍇🌽🍊🌾Crop Recommendation System🌽🍊🌾🍇")
st.write('<div style="text-align: justify;">The crop recommendation system predicts the best crop to cultivate based on user input values such as nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall. This system can be used by farmers or anyone interested in agriculture to make informed decisions about the best crop to plant based on their soil and weather conditions.</div>', unsafe_allow_html=True)

image="https://www.ceres.org/sites/default/files/2022-09/Untitled%20design%20%2810%29_1.png"
st.image(image, use_column_width=True)

# User Input
st.write('## Enter the following values:')
col1, col2 = st.columns(2)
with col1:
    n = st.number_input('Nitrogen (N) in ppm', min_value=0, max_value=200, step=1)
    p = st.number_input('Phosphorus (P) in ppm', min_value=0, max_value=200, step=1)
with col2:
    k = st.number_input('Potassium (K) in ppm', min_value=0, max_value=200, step=1)
    temperature = st.number_input('Temperature in Celsius', min_value=0, max_value=50, step=1)

col3, col4 = st.columns(2)
with col3:
    humidity = st.number_input('Humidity in %', min_value=0, max_value=100, step=1)
    ph = st.number_input('Soil pH', min_value=0.0, max_value=14.0, step=0.1)
with col4:
    rainfall = st.number_input('Rainfall in mm', min_value=0, max_value=600, step=1)


import base64

if st.button('Recommend'):
    recommended_crop = predict_crop(n, p, k, temperature, humidity, ph, rainfall)
    st.write(f"**<div style='text-align:center'>{recommended_crop}</div>**", unsafe_allow_html=True)
    image = Image.open(f'{recommended_crop}.jpg')
    with open(f'{recommended_crop}.jpg', 'rb') as f:
        data = f.read()
        data_url = base64.b64encode(data).decode('utf-8')
        st.write(f'<div style="display: flex; justify-content: center; align-items: center;"><img src="data:image/jpg;base64,{data_url}" width="500" /></div>', unsafe_allow_html=True)

# import base64

# if st.button('Recommend'):
#     recommended_crop = predict_crop(n, p, k, temperature, humidity, ph, rainfall)
#     st.write(f"<div style='text-align:center'>{recommended_crop}</div>", unsafe_allow_html=True)
#     image = Image.open(f'{recommended_crop}.jpg')
#     with open(f'{recommended_crop}.jpg', 'rb') as f:
#         data = f.read()
#         data_url = base64.b64encode(data).decode('utf-8')
#         st.write(f'<div style="display: flex; justify-content: center; align-items: center;"><img src="data:image/jpg;base64,{data_url}" width="500" /></div>', unsafe_allow_html=True)

# # Center-align the button
# st.markdown("<div style='text-align:center;'><br><br></div>", unsafe_allow_html=True)


