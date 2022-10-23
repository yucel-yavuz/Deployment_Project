import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image

st.sidebar.title("Please select features of car you interested")

st.markdown("<h1 style='text-align:center; color:black;'>Car Price Prediction</h2>", unsafe_allow_html=True)

# Adding image
img = Image.open("cars.png")
new_img=img.resize((750, 250))
st.image(new_img)


age=st.sidebar.selectbox("Age of car",(0,1,2,3))
hp=st.sidebar.slider("Horse Power (kW)", 40, 300, step=5)
km=st.sidebar.slider("Kilometers (km)", 0,350000, step=1000)
gearing_type=st.sidebar.radio('Gear Type',('Automatic','Manual','Semi-automatic'))
car_model=st.sidebar.selectbox("Model Type", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))


model=pickle.load(open("rf_model_new","rb"))
transformer = pickle.load(open('transformer', 'rb'))


my_dict = {
    "age": age,
    "hp_kW": hp,
    "km": km,
    'Gearing_Type':gearing_type,
    "make_model": car_model
    
}

df = pd.DataFrame.from_dict([my_dict])



# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)


# st.subheader("The feature of car is below")


#st.header("The configuration of your car is below")
st.table(df)

df2 = transformer.transform(df)

#st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = model.predict(df2)
    st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))
    
