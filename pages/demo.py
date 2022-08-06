from json import load
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


@st.cache
def load_models():
    age_model = tf.keras.models.load_model('../models/age_model.h5')
    ethnicity_model = tf.keras.models.load_model('../models/ethnicity_model.h5')
    gender_model = tf.keras.models.load_model('../models/gender_model.h5')
    return age_model, ethnicity_model, gender_model

load_models()

def predict_single(model_type, img_data):
    map = ['age', 'ethnicity', 'gender']

def predict_all(path):
    '''Predict gender, ethnicity, and age from any image'''
    
    img1 = Image.open(path).convert(mode="RGB")
    img1 = img1.resize((48,48))
    img1 = ImageOps.grayscale(img1)
    array1 = np.array(img1.getdata())
    input_img = np.reshape(array1, (48,48,1)) / 255.0

    plt.imshow(input_img)
    
    # Gender
    pred1 = age_model.predict(np.expand_dims(input_img, axis=0))[0]
    gender = 'Female'
    if pred1 < 0.5:
        gender = 'Male'

    # Ethnicity
    pred2 = ethnicity_model.predict(np.expand_dims(input_img, axis=0))
    label_map = ['Caucasian', 'African', 'East Asian', 'South Asian', 'Latino']
    ethnicity = label_map[np.argmax(pred2)]

    # Age
    age = gender_model.predict(np.expand_dims(input_img, axis=0))
        
    print('--Predictions--')
    print(f'Gender: {gender} ({pred1})')
    print(f'Ethnicity: {ethnicity} ({pred2[0][np.argmax(pred2)]})')
    print(f'Age: {age[0][0]}')

st.title('App Demo')