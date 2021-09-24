import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from PIL import Image
import sys
from keras.preprocessing import image  
import keras    
import cv2   


def file_selector(folder_path):
    try:
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
    except:
         st.info("Select one or more files.")
         st.write('Please provid a valid folder path')

    return os.path.join(folder_path, selected_filename)



def main():


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

    folderPath = st.text_input('Enter folder path:')

    if folderPath:    
        filename = file_selector(folderPath)
    else:
        #fileslist.clear()  # Hack to clear list if the user clears the cache and reloads the page
        st.info("Select one or more files.")


    try: 
        filenames = os.listdir(folderPath)
        selected_filename = st.selectbox('Select a file', filenames)
        st.write(os.path.join(folderPath, selected_filename))
    except:
        
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    col0,col1, col2, col3 = st.beta_columns([1,1,6,1])


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
    #img1 = image.load_img(img, target_size=(224, 224))
    

        with col3:
            st.write('')
    

    #img1 = Image.open(io.BytesIO(img_file_buffer))
    #img1 = cv2.imread(np.array(img))
#img = img.convert('RGB')
#img = img.resize(target_size, Image.NEAREST)
#img = image.img_to_array(img)
        st.subheader('Classifier Results:')

# Call the main process

main()