import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import sys
from keras.preprocessing import image  
import keras                


st.set_page_config(
    page_title="Dog Breed Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title('      Dog Breed Classifier')
st.sidebar.title('App Description')
st.sidebar.markdown('This is an App to classify Human Faces and Dog breeds using **Convolutional Neural Networks**')
st.sidebar.markdown('As part of the final project for [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd0259) from Udacity')
st.sidebar.markdown('The complete Github repo can be found in [Github](https://github.com/rafaelmata357/Data-Science-Udacity/tree/master/dog%20breed%20classifier)')
st.sidebar.markdown('[![An old rock in the desert](./linkedinlogo.png)](www.nacion.com)')

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

col0,col1, col2, col3 = st.beta_columns([1,1,6,1])
st.write(sys.version)
st.write(keras.__version__)

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    with col0:
        st.write('')
    with col1:
        st.write('')

    with col2:
        st.image(
    img, caption=f"Processed image", width= 400) #use_column_width=True)
    st.write('Procesing Image.... {}'.format(img_file_buffer.name))
    img1 = image.load_img('/Users/rafaelmata357/Downloads/Limon.JPG', target_size=(224, 224))
    

    with col3:
        st.write('')
    

st.subheader('Classifier Results:')