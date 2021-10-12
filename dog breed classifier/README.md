# Dog breed classifier using CNN

## Project Overview


```
Image classification is an important task nowadays,  plays an important role for various applications from the automobile industry,  medical analysis, security, automated perception in robots, among others use cases.


But  categorizing and assigning labels to image is not a simple task for computers, with the advance in Artificial Intelligence, machine learning and image datasets available to train different type of neural neural networks, now is possible to achive acceptable results in image classification.  


As part of the final capstone Udacity Data science nanodegree, in this project different kind of computing vision algorithms will be used to classify human and dog images and try to predict the dog breed,  different datasets with images containing dog breeds and human faces are used to train, validate  and test different Convolutional Neural Networks models using supervised learning with the keras framework.

As a final product a web app is deployed putting together a series of models to perform different tasks.
```


## Problem statement


```
The problem  to be solved is image classification, we need to classify an image in three categories: dogs, humans  and others, If a dog is detected in the image, an estimate of the dog's breed must be provided. If a human is detected also an estimate of the dog breed that is most resembling, if neither is detected in the image, an output message that indicates an error must be shown.  

The project is divided in different steps:

- Step 0: Import Datasets, analyze and process
- Step 1: Detect Humans using the OpenCV framework
- Step 2: Detect Dogs using the Resnet50 pretrainned CNN
- Step 3: Create different CNN models to Classify Dog Breeds 
        - A model from Scratch
        - With transfer learning using VGG16 model
        - With transfer learning using Inception V3 model
- Step 4: Evaluate models results
- Step 5: Write and test the algorithm to classify predict the dogs breeds using the best models
- Step 6: Write a web app

The expected project results is a web app with at least 60% accuracy, this app will accept any user-supplied image as input and provide a text message as output with the results. 
```


## Metrics


```
Based on the characteristics of the problem which is classification, diffrent metrics can be used: accuracy, precision, recall, f1 score, etc, to measure the performance of the  models.

In classification problem we have different results:
```

- $TP: True Positives$
- $TN: True Negatives$
- $FP: False Positives$
- $FN: False Negatives$

Due to the datastet characteristics I will use **accuracy** which is one of the most common evaluation metrics, accuracy is very useful when the target class is balanced, that is the total number of correct predictions  (True Positives) divided by the total number of predictions made for a dataset.

$Accuracy = \frac{TP}{TP + TN + FP + FN}$

The CNN must attain at least **60%** accuracy on the test set.


## Analysis


### The datasets

#### Data exploration

There are two datasets available to solve this problem

- A dataset with 8351 dog images and 133 breeds
- A dataset with 13233 human face images

All the images are in jpg format

The average size of the images is 138351 bytes
The minimum size is 4362 bytes
The maximun size is 7389073 bytes
The average pixel color
Nan values 0
