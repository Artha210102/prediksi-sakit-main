import pickle
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Title of the web app
st.title('Prediksi Diabetes')
st.subheader('Masukkan Data untuk Memprediksi Diabetes')

# Input layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    Pregnancies = st.number_input('Angka Kehamilan', min_value=0, step=1)
with col2:
    Glucose = st.number_input('Glukosa', min_value=0, step=1)
with col3:
    BloodPressure = st.number_input('Tekanan Darah (mmHg)', min_value=0, step=1)
with col4:
    Age = st.number_input('Usia (tahun)', min_value=0, step=1)

col5, col6, col7, col8 = st.columns(4)

with col5:
    Insulin = st.number_input('Insulin serum (mu U/ml)', min_value=0, step=1)
with col6:
    BMI = st.number_input('BMI', min_value=0.0, step=0.1)
with col7:
    DiabetesPedigreeFunction = st.number_input('Riwayat Diabetes Keluarga', min_value=0.0, step=0.01)
with col8:
    SkinThickness = st.number_input('Ketebalan Kulit (mm)', min_value=0, step=1)

# Prediksi
if st.button('Test Prediksi Diabetes'):
    prediction = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    if prediction[0] == 0:
        st.success('Pasien tidak terkena Diabetes')
    else:
        st.error('Pasien terkena Diabetes')

# Footer
st.write("---")
st.write("Aplikasi Prediksi Diabetes berbasis Machine Learning")
