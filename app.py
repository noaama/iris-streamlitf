import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('Classification des fleurs d iris')
st.markdown('Modèle pour classer les fleurs d iris en(setosa, versicolor, virginica) en fonction de leur sépale/pétaleet de leur longueur/largeur.')

st.header("Caractéristiques de la plante")
col1, col2 = st.columns(2)

with col1:
    st.text("Caractéristiques du sépale")
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Caractéristiques du pétale")
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

st.text('')
if st.button("Prédire le type d'Iris"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])


st.text('')
st.text('')
st.markdown(
    '`Create by` [Nouaman AMARI](https://www.linkedin.com/in/nouamane-amari-566a07204) | \
         `Code:` [GitHub](https://github.com/noaama/iris-streamlitf/)')
