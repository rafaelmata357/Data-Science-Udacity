#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  15 Set 2021                                 
# REVISED DATE :  25 Set 2021
# PURPOSE: Create a program with utils fuctions to be used in the main program app.py to classify Dogs breeds



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


def only_image_files(files):
    ''' Function to return the list of images files

        Params:
        -------
        files : list, list of files

        Returns:
        img_files: list of jpg, png or jpeg files

    '''


    pass






def file_selector(folder_path):

    ''' Function to return the path and file selected

        Params:
        -------
        folder_path: string path to the files

        Returns:
        path + file : string, path and file selected
        valid_file : boolean, True if a valid file is selected
    '''
  
    selected_filename = ''
    valid_file = False
    try:
        filenames = ['<select>'] + os.listdir(folder_path)
        selected_filename = st.selectbox('Select an image file', filenames, index=0)
        valid_file = True
        if selected_filename == '<select>':
            valid_file = False
        
    except:
        st.info('Please provid a valid folder path')

    return os.path.join(folder_path, selected_filename), valid_file

def display_image(filename):
    ''' Function to display an image using st

    Params:
    img : strig, path

    Returns:
    None
    '''

    col0,col1, col2, col3 = st.beta_columns([1,1,6,1])
    img = Image.open(filename)
    with col0:
            st.write('')
    with col1:
        st.write('')

    with col2:
        st.image(
        img, caption=f"Processed image", width= 400) #use_column_width=True)
        

    return None