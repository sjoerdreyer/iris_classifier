import streamlit as st
import requests
import json
import pickle
from package_folder.utils import from_number_to_flower

st.title("My iris classifier app")

st.write("Select your features")

# Creating four sliders
value1 = st.slider('Select a value for Sepal length', min_value=0, max_value=10, value=1, step=1)
value2 = st.slider('Select a value for Sepal width',  min_value=0, max_value=10, value=1, step=1)
value3 = st.slider('Select a value for Petal length',  min_value=0, max_value=10, value=1, step=1)
value4 = st.slider('Select a value for Petal width',  min_value=0, max_value=10, value=1, step=1)

# TEST LINE
# response = requests.get(f"https://mvp-cctigceneq-ew.a.run.app/predict?sepal_length={value1}&sepal_width={value2}&petal_length={value3}&petal_width={value4}").json()
# response = requests.get(f"https://lecture-api-2thwkxfi3a-ew.a.run.app/predict?sepal_length={value1}&sepal_width={value2}&petal_length={value3}&petal_width={value4}").json()

with open('models/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

prediction = from_number_to_flower(float(model.predict([[value1,value2,value3,value4]])[0]))


st.write("The flower belongs to class", str(prediction))
