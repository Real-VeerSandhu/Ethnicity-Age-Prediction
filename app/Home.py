import streamlit as st

st.set_page_config(page_title="Identity Prediction", page_icon="👨", layout='centered', initial_sidebar_state="expanded") # Config

st.title('Identity Prediction')
st.write('Predicting ethnicity, age, and gender with CNNs')

st.sidebar.success('Select a page above')
st.caption('Home: information | Demo: app | About: resources')