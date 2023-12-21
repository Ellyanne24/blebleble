import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

st.write("# Iris Species Prediction")
st.write("This app predicts the **Iris flower** species!")

st.sidebar.header('Score the following characteristics')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = sns.load_dataset('iris')
X = data.drop(['species'],axis=1)
Y = data.species.copy()

import pickle
pickle.dump(modeldt, open("IrisPrediction.h5","wb"))
print("model is saved")

prediction = modelTreeIris.predict(df)
prediction_proba = modelTreeIris.predict_proba(df)

st.subheader('Species categories and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
