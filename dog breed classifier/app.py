#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  15 Set 2021                                 
# REVISED DATE :  29 Set 2021
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
    
    # Load the ResNet50 Model and Inceptionv3 models
    ResNet50_model = import_resnet50_model() 
    path_save_weights = 'saved_models/weights.best.Inception.hdf5' # Path to the best weights Inception model trainned
    Inception_model = load_Inception_model(path_save_weights)
    Inception_bottleneck = extract_Inception_bottleneck()

    # Load dog names list
    dog_names_path = 'dog_names.json'
    dog_names = load_dog_names(dog_names_path)

    folderPath = st.text_input('Enter images folder path:')
    
    if folderPath:    
        filename, valid_file = file_selector(folderPath)
        if valid_file:
            
            
            col2, col3, imageLocation = display_image(filename)
           
            image_detected  = classify_images(filename,  ResNet50_model)
            
            with col3:
                st.subheader('Classifier Results: {}'.format(image_detected))
         
            if image_detected == 'Human':
                scale = 1.35
                minNeighbors = 4
                human_face_img = face_detector2(filename, scale, minNeighbors, True)
                with col2:
                    imageLocation.image(human_face_img, caption=f"Processed image", width= 400)
            #------

            if image_detected == 'Human' or image_detected == 'Dog':   # Try to indetify the Dog breed
                breed_detected = Inception_predict_breed(filename, Inception_model,dog_names, Inception_bottleneck)
            else:
                breed_detected = 'None'
            with col3:
                st.subheader('Possible Dog breed: {}'.format(breed_detected))
         
    return None    

# Call the main program
if __name__ == '__main__':
    main()