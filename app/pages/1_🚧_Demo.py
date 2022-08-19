import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time

# Page set up
st.set_page_config(page_title="Demo", page_icon="👨", layout='centered', initial_sidebar_state="expanded")
st.sidebar.success('Select a page above')

# Buffer loading time
my_bar = st.sidebar.progress(0)
for percent_complete in range(100):
     time.sleep(0.001)
     my_bar.progress(percent_complete + 1)

# Save models into cache
@st.cache(allow_output_mutation=True, show_spinner=True, hash_funcs={"MyUnhashableClass": lambda _: None})
def load_models():
    print('**Models Loaded**')
    return tf.keras.models.load_model('models/age_model.h5'), \
        tf.keras.models.load_model('models/ethnicity_model.h5'), \
        tf.keras.models.load_model('models/gender_model.h5')

age_model, ethnicity_model, gender_model = load_models()

# Process an uploaded image
def predict_uploaded_image(input_image):
    '''Process and upload a local image into the 3 CNNs'''
    raw_image = Image.open(input_image).convert('RGB')
    image_pil = raw_image.resize((48,48))

    img1 = ImageOps.grayscale(image_pil)
    array1 = np.array(img1.getdata())
    processed_image = np.reshape(array1, (48,48,1)) / 255.0

    return cnn_predict(processed_image)

# Process a testing-dataset image
def predict_dataset_image(path):
    '''Select an image from the testing-dataset and upload it into the 3 CNNs'''
    img1 = Image.open(path).convert(mode="RGB")
    img1 = ImageOps.grayscale(img1)
    array1 = np.array(img1.getdata())
    processed_image = np.reshape(array1, (48,48,1)) / 255.0

    return cnn_predict(processed_image)

# Predict identity features given pixel data of a 48x48 image
def cnn_predict(x_data):
    '''Predict gender, ethnicity, and age from a 48x48 image'''
    # Gender
    pred1 = gender_model.predict(np.expand_dims(x_data, axis=0))[0]
    gender = 'Female'
    if pred1 < 0.5:
        gender = 'Male'
    raw_pred1 = str(abs((pred1-0.5)/0.5)[0]*100) + '% ' + gender

    # Ethnicity
    pred2 = ethnicity_model.predict(np.expand_dims(x_data, axis=0))
    label_map = ['Caucasian', 'African', 'East Asian', 'South Asian', 'Latino']
    ethnicity = label_map[np.argmax(pred2)]
    raw_pred2 = pred2[0]

    # Age
    raw_pred3 = str(age_model.predict(np.expand_dims(x_data, axis=0))[0][0]) + ' years old'
    age = round(age_model.predict(np.expand_dims(x_data, axis=0))[0][0])
        
    return gender, ethnicity, age, [raw_pred1, raw_pred2, raw_pred3]

# User interface
st.title('App Demo')
st.write('Use the data inputs below to upload (or select) a **portrait** and run the `CNNs`')
st.write('----')
type = st.radio(
     'Select an image upload method',
     ('Upload from local disk', 'Select from testing data'))

if type == 'Upload from local disk':
    uploaded_image = st.file_uploader('Upload An Image', help='Select an image on your local device for the CNNs to process and output', type=['jpg', 'png'])
    if uploaded_image:
        file_name = 'Uploaded Image: ' + str(uploaded_image.name) + ' (' + str(uploaded_image.size) +'bytes)'
        
        pred_button = st.button('Predict')
        
        col1, col2 = st.columns([1,1])
        with col1:
            st.image(uploaded_image, width=300, caption=file_name)
        with col2:
            if pred_button:
                pred_gender, pred_ethnicity, pred_age, pred_raw = predict_uploaded_image(uploaded_image)
                st.write('**Predictions**')

                st.write(f'Gender: `{pred_gender}`')
                st.write(f'Ethnicity: `{pred_ethnicity}`')
                st.write(f'Age: `{pred_age}`')
                st.write('----')
                st.write('*Raw Output:*', pred_raw)
else:
    image_index = st.select_slider(
     'Select a test image',
     options=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11' ,'12'])
    
    image_dir = 'app/demo_images/image' + image_index + '.png'
    file_name = 'Selected Image: image' + str(image_index) + '.png'

    st.write(image_dir[4:])

    pred_button = st.button('Predict')

    col1, col2 = st.columns([1,1])
    with col1:
        st.image(image_dir, width=300, caption=file_name)
    with col2:
        if pred_button:
            pred_gender, pred_ethnicity, pred_age, pred_raw = predict_dataset_image(image_dir)
            st.write('**Predictions**')

            st.write(f'Gender: `{pred_gender}`')
            st.write(f'Ethnicity: `{pred_ethnicity}`')
            st.write(f'Age: `{pred_age}`')
            st.write('*Raw Output:*', pred_raw)
