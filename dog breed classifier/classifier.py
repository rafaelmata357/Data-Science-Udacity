#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  15 Set 2021                                 
# REVISED DATE :  25 Set 2021
# PURPOSE: Create a program to classify Dog breeds images using CNN 
#          
# 
# Frameworks used:
#
# - keras
# - OpenCV


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

def classify_human(image):

    ''' Function to classify human faces based on the project jupyter notebook provided by Udacity

        Params:
        -------
        img : string, path to the images

        Returns:

        face: image detected
    '''

    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    # load color (BGR) image
    img = cv2.imread(image)
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray)

    # print number of faces detected in the image
    print('Number of faces detected:', len(faces))

    # get bounding box for each detected face
    for (x,y,w,h) in faces:
    # add bounding box to color image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return cv_rgb


    def face_detector2(img_path, scale, minNeighbors):
    
    ''' Algorithm to detect faces using Open Cv library
    
        Params:
        ------
        img_path : string, path to the image file
        scale: float, scale parameter to zoom out the image by this factor
        minNeighbors: int, cadidate rectangles to found the face
               
        Returns:
        --------
        boolean variable indicating if a face is detected

    
    '''

    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scale,minNeighbors)
    
    if len(faces)>0 :
        # get bounding box for each detected face
        for (x,y,w,h) in faces:
    
        # add bounding box to color image
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
        # convert BGR image to RGB for plotting
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # display the image, along with bounding box
               
    return len(faces) > 0, cv_rgb
    else:
        return len(faces) > 0