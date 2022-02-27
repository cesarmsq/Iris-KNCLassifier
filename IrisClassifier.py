import streamlit as st
import joblib

model = joblib.load('data/IrisClassifierKN.joblib')

st.markdown("""
<h1 style='text-align: center; color: #642275;'>Iris Classifier</h1>

---
""", unsafe_allow_html = True)

col1, col2, col3 = st.columns(3)

length_sepal = col1.number_input('Sepal length')
width_sepal = col2.number_input('Sepal width')

length_petal = col1.number_input('Petal length')
width_petal = col2.number_input('Petal width')

predicted = model.predict([[
    length_sepal, 
    width_sepal, 
    length_petal, 
    width_petal]])[0]

col3.markdown(f"""<h5 style='text-align: center;'>{predicted}</h5>""", unsafe_allow_html=True)

if predicted == "Iris-setosa":
    col3.image('images/setosa.png')
elif predicted == "Iris-versicolor":
    col3.image('images/versicolor.png')
elif predicted == "Iris-virginica":
    col3.image('images/virginica.png')

"""---
Created by Cesar Supo
"""


