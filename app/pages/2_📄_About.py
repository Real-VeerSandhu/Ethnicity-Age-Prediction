import streamlit as st

# Page set up
st.set_page_config(page_title="Demo", page_icon="👨", layout='centered', initial_sidebar_state="expanded")
st.sidebar.success('Select a page above')

# App
st.title('About')
st.write('''
- Github Repository: `https://github.com/Real-VeerSandhu/Ethnicity-Age-Prediction`
- Dataset: `https://www.kaggle.com/datasets/age-gender-ethnicity-face-data-csv`
- Research Source: `https://ieeexplore.ieee.org/abstract/document/6406799`''')