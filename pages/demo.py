import streamlit as st
import tensorflow as tf

@st.cache
def load_models():
    age_model = tf.keras.models.load_model('../models/age_model.h5')
    ethnicity_model = tf.keras.models.load_model('../models/ethnicity_model.h5')
    gender_model = tf.keras.models.load_model('../models/gender_model.h5')

def predict_single(model_type, img_data):
    map = ['age', 'ethnicity', 'gender']

st.title('App Demo')