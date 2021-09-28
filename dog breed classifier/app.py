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
from classifier import *

import time

@st.cache()
def tiempo():
    return time.ctime()

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

    st.write(tiempo())
    
    ResNet50_model = import_resnet50_model() # import ResNet 50 pretrainned model with transfer learning
    path = 'saved_models/weights.best.Resnet50.hdf5' # Path to the best weights Resnet 50 model trainned
    new_Resnet50_model = load_new_Resnet50(path)

    folderPath = st.text_input('Enter folder path:')

    

    if folderPath:    
        filename, valid_file = file_selector(folderPath)
        if valid_file:
            #st.write(filename)
            display_image(filename)
            st.write('Procesing Image.... {}'.format(filename.split('/')[-1]))
            st.subheader('Classifier Results:')
            
            scale = 1.35
            minNeighbors = 4 
            face_detected, face_image = face_detector2(filename, scale, minNeighbors)
            if face_detected:
                st.image(face_image, caption=f"Processed image", width= 400)

            if dog_detector(filename,ResNet50_model):
                st.write('Guato detectado...')
         
    return None    

# Call the main program
main()