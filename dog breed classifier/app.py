#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  15 Set 2021                                 
# REVISED DATE :  25 Set 2021
# PURPOSE: Create a program to classify Dog breeds and human faces with a web interface where users can choose the image 
#          
# 
# Command Line:  streamlit app.py


# Imports python modules


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
from utils import *



def main():

    ''' Main program for the app '''


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
        filename, valid_file = file_selector(folderPath)
        if valid_file:
            #st.write(filename)
            display_image(filename)
            st.write('Procesing Image.... {}'.format(filename.split('/')[-1]))
            st.subheader('Classifier Results:')
            
    
   
        
   # img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    


    #if img_file_buffer is not None:

     #   img = Image.open(img_file_buffer)
      #  with col0:
       #     st.write('')
       # with col1:
        #    st.write('')

        #with col2:
         #   st.image(
        #img, caption=f"Processed image", width= 400) #use_column_width=True)
        #st.write('Procesing Image.... {}'.format(img_file_buffer.name))
        #data = img_file_buffer.read()
        #img1 = image.load_img(data, target_size=(224, 224))
    

        #with col3:
         #   st.write('')
    

    #img1 = Image.open(io.BytesIO(img_file_buffer))
    #img1 = cv2.imread(np.array(img))
#img = img.convert('RGB')
#img = img.resize(target_size, Image.NEAREST)
#img = image.img_to_array(img)
        

# Call the main process

main()