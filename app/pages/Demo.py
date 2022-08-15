from json import load
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


@st.cache(allow_output_mutation=True, show_spinner=True, hash_funcs={"MyUnhashableClass": lambda _: None})
def load_models():
    print('**Models Loaded**')
    return tf.keras.models.load_model('models/age_model.h5'), \
        tf.keras.models.load_model('models/ethnicity_model.h5'), \
        tf.keras.models.load_model('models/gender_model.h5')

age_model, ethnicity_model, gender_model = load_models()

def predict_uploaded_image(input_image):
    '''Process and upload an image into the 3 CNNs'''
    raw_image = Image.open(input_image).convert('RGB')
    image_pil = raw_image.resize((48,48))

    img1 = ImageOps.grayscale(image_pil)
    array1 = np.array(img1.getdata())
    processed_image = np.reshape(array1, (48,48,1)) / 255.0

    # Gender
    pred1 = gender_model.predict(np.expand_dims(processed_image, axis=0))[0]
    gender = 'Female'
    if pred1 < 0.5:
        gender = 'Male'

    # Ethnicity
    pred2 = ethnicity_model.predict(np.expand_dims(processed_image, axis=0))
    label_map = ['Caucasian', 'African', 'East Asian', 'South Asian', 'Latino']
    ethnicity = label_map[np.argmax(pred2)]

    # Age
    age = round(age_model.predict(np.expand_dims(processed_image, axis=0))[0][0])
        
    return gender, ethnicity, age
# def predict_all(path):
#     '''Predict gender, ethnicity, and age from any image'''
    
#     img1 = Image.open(path).convert(mode="RGB")
#     img1 = img1.resize((48,48))
#     img1 = ImageOps.grayscale(img1)
#     array1 = np.array(img1.getdata())
#     input_img = np.reshape(array1, (48,48,1)) / 255.0

#     plt.imshow(input_img)
    
#     # Gender
#     pred1 = age_model.predict(np.expand_dims(input_img, axis=0))[0]
#     gender = 'Female'
#     if pred1 < 0.5:
#         gender = 'Male'

#     # Ethnicity
#     pred2 = ethnicity_model.predict(np.expand_dims(input_img, axis=0))
#     label_map = ['Caucasian', 'African', 'East Asian', 'South Asian', 'Latino']
#     ethnicity = label_map[np.argmax(pred2)]

#     # Age
#     age = gender_model.predict(np.expand_dims(input_img, axis=0))
        
#     print('--Predictions--')
#     print(f'Gender: {gender} ({pred1})')
#     print(f'Ethnicity: {ethnicity} ({pred2[0][np.argmax(pred2)]})')
#     print(f'Age: {age[0][0]}')

st.title('App Demo')
st.write('Using the user-inputs below, upload (or select) an **image** and run the select `CNN(s)`')
st.write('----')
type = st.radio(
     'Select an image upload method',
     ('Upload from local disk', 'Select from testing data'))

if type == 'Upload from local disk':
    uploaded_image = st.file_uploader('Upload An Image', help='Select an image on local device for the CNNs to process and output', type=['jpg', 'png'])
    if uploaded_image:
        if st.button('Predict'):
            pred_gender, pred_ethnicity, pred_age = predict_uploaded_image(uploaded_image)

            col1, col2 = st.columns([1,1])
            with col1:
                st.image(uploaded_image)
            with col2:
                st.write('**Deteced...**')

                st.write(f'Gender: `{pred_gender}`')
                st.write(f'Ethnicity: `{pred_ethnicity}`')
                st.write(f'Age: `{pred_age}`')