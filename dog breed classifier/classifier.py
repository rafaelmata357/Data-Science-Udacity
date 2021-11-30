#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  15 Set 2021                                 
# REVISED DATE :  29 Nov 2021
# PURPOSE: Create a program to classify Dog breeds images using different CNN models already pre trained
#          
# 
# Frameworks used:
#
# - keras
# - OpenCV


# Imports python modules

import streamlit as st
import numpy as np
import os
from PIL import Image
import sys
import keras    
import cv2  
from keras.preprocessing import image  
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3, preprocess_input

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


def face_detector2(img_path, scale, minNeighbors, image):
    
    ''' Algorithm to detect faces using Open Cv library
    
        Params:
        ------
        img_path : string, path to the image file
        scale: float, scale parameter to zoom out the image by this factor
        minNeighbors: int, cadidate rectangles to found the face
        image: boolean, to return the image with a rectangle
               
        Returns:
        --------
        boolean variable indicating if a face is detected
        cv_rgb : image with boundary rectangle when an image is detected

    
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
               
  
    if image:
        return cv_rgb
    else:
        return len(faces) > 0


@st.cache()  #To load the model once
def import_resnet50_model():

    ''' Fucntion to define the  pre-trained ResNet-50 model to detect dogs 
        
        Paras: None

        Returns :

        Resnet50model: Pretrainned model
    
    '''

    ResNet50_model = ResNet50(weights='imagenet')

    return ResNet50_model

def ResNet50_predict_labels(img_path, ResNet50_model):
    
    ''' Function to returns prediction vector for image located at img_path, taken from Udacity example notebook

        Params:
        img_path : string, path to the image
        ResNet50_model : keras pretrainned ResNet Model

        Returns:
        label : int, index to the possible dog breed index 
    '''

    img = resnet_preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path, ResNet50_model):
    ''' Function that returns "True" if a dog is detected in the image stored at img_path
        Params:
        --------
        img_path: string, path to the image

    '''
    prediction = ResNet50_predict_labels(img_path,ResNet50_model)
    return ((prediction <= 268) & (prediction >= 151))

def path_to_tensor(img_path):

    ''' Function to process and Image and convert from 3D to 4D Tensor , function taken from Udacity example notebook
    
        Params:
        --------
        img_path: string, path to the image

        Returns:
        4D tensor
    
    '''

    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

#@st.cache()  #To load the model once
def load_Inception_model(path):
    ''' Function to create the Imagenet model and load the best weights trainned

        Params:
        ------
        path: string, path to the save weights

        Returns:
        --------
        Inception_model : Incpetionv3 pretrainned Keras model
    '''

    #Create the model according with the definition used in the jupyter notebook

    Inception_model = Sequential()
    Inception_model.add(GlobalAveragePooling2D(input_shape=(5,5,2048))) #Input size according with bottleneck trainning features
    Inception_model.add(Dropout(0.45))
    Inception_model.add(Dense(256, activation='relu'))
    Inception_model.add(Dropout(0.45))
    Inception_model.add(Dense(133, activation='softmax'))
    #Inception_model.summary()

    #Load the model with the best weights

    Inception_model.load_weights(path) 

    return Inception_model

@st.cache()  #To load the model once
def extract_Inception_bottleneck():

    ''' Function to extract the InceptionV3 model from keras and generate the bottleneck feature
       
        Params: None

        Returns:
        --------
        Inception_bottleneck : InceptionV3 keras model with no top layers
 
    '''

    Inception_bottleneck = InceptionV3(weights='imagenet', include_top=False)

    return Inception_bottleneck

def Inception_predict_breed(img_path, Inception_model,dog_names, Inception_bottleneck):

    ''' Fucntion to predict the dog breed using the trainned model with Inception and transfer learning

        Params:
        ---------
        img_path: string, path to the image file
        Inception_model: keras model 
        dog_names: list, list with the dog names

        Returns:
        ---------
        dog_breed: string, dog breed predicted
    '''
    
    # extract bottleneck features
    bottleneck_feature = Inception_bottleneck.predict(preprocess_input(path_to_tensor(img_path)))
  
    # obtain predicted vector
    predicted_vector = Inception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    dog_name = dog_names[np.argmax(predicted_vector)].split('/')[2]
    dog_name = dog_name.split('.')[1]
    return dog_name 

def classify_images(image_path, ResNet50_model):
    ''' Function to classify an image in three categories: Human, Dog, other
        If a dog is detected it also predict the dog breed
        
        This fuction uses different algorithms that are based on NN and CNN to make the predictions
          - face_detector2(img_path, scale, minNeighbors, show_image)
          - dog_detector(dog) 
          - Resnet50_predict_breed(img_path)
          
          
        Params:
        ---------
        
        image_path : string, file path of the image
        
        Returns:
        --------
        image_detected : string, type of image classified
        breed_detected : string, type of Dog breed detected, when apply
        
        
        
    '''
    

    scale = 1.35
    minNeighbors = 4
    
    image_detected = ''
    breed_detected = ''
    
    
    if dog_detector(image_path, ResNet50_model):                                   # Try to detect a Dog
        image_detected = 'Dog'
    elif face_detector2(image_path, scale, minNeighbors, False):  # Try to detect a human face
        image_detected = 'Human'
    else:
        image_detected = 'Other'
        
     
    return image_detected

