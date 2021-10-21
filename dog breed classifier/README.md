# Dog breed classifier using CNN

## Installation

```
$gh repo clone rafaelmata357/Data-Science-Udacity/dog breed classifier
```

## Libraries used

* streamlit

* numpy
* os
* PIL
* sys
* keras
* cv2
* sklearn
* json

The libraries details can be found in the requirements.txt file

## Files in the repository

**/haarcascade/haarcascade_frontalface_alt.xml** xml file with the pre-trained face detectors

**/saved_models/weights.best.Inception.hdf5** best weights for the inception model

**/test_images** Images used to thes the classifier

**app.py** the main progran with the web app

**utils.py** Utilities scripts used in the app

**classifier.py** CNN models to classify the images used in the web app

**dog_app.ipynb** Jupyter notebook used to train and test the different models

**Inception_fine_tuning.ipynb** Jupyter notebook used to improve the Inception model

**dogs.json** Json file with the dog breeds

**requirements.txt** Required libraries to use the web app

**Readme.md** This file


## How to interact with the project


From the repository root directory, run the following command:

```
 $streamlit run app.py
```

This launch the web app on the local machine, where the user can interact and test with different images

## Project Overview

Image classification is an important task nowadays, plays an important role for various applications from the automobile industry, medical analysis, security, automated perception in robots, among others use cases.

But categorizing and assigning labels to image is not a simple task for computers, with the advance in Artificial Intelligence, machine learning and image datasets available, now is possible to achive acceptable results in image classification.

As part of the final capstone project for [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd0259) from Udacity,  different kind of computing vision algorithms will be used to classify human and dog images and try to predict the dog breed, different datasets with images containing dog and human faces are used to train, validate and test different Convolutional Neural Networks models using supervised learning with the keras framework.

As a final product a **web app** is deployed putting together a series of models to perform different tasks.

## Problem statement

The problem to be solved is image classification, we need to classify an image in three categories: dogs, humans and others, If a dog is detected in the image an estimate of the dog breed must be provided. If a human is detected also an estimate must be given with the most resembling dog, if neither is detected in the image an output message that indicates an error must be shown.

The project is divided in different steps:

```
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
```

The expected project results is a web app with at least **60% accuracy**, this app will accept any user-supplied image as input and provide a text message as output with the results.

## Metrics

Based on the characteristics of the problem which is classification, diffrent metrics can be used: accuracy, precision, recall, f1 score, etc, to measure the performance of the models.

In classification these are the possible results:

- TP: True Positives
- TN: True Negatives
- FP: False Positives
- FN: False Negatives

Due to the datastet characteristics I will use **accuracy** which is one of the most common evaluation metrics, accuracy is very useful when the target class is balanced, that is the total number of correct predictions (True Positives) divided by the total number of predictions made for a dataset.

![Accuracy](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/accuracy.png)

The CNN must attain at least **60%** accuracy on the test set.

## Analysis

### The datasets

#### Data exploration

There are two datasets available to solve this problem

- A dataset with 8351 dog images and 133 breeds
- A dataset with 13233 human face images

All the images are in jpg format

Due to the nature of the dataset there is not Nan values

These are some statistics about the datasets

![Dataset stats](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/dataset_characteristics.png)

### Data visualization

This is the sample distribution

![Image file sample distribution](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/sample_distribution.png)

The average breeds sample size is between 60 to 70 images per dog, there is top samples above 90 and lower samples below 40, but the minimum 33.

---

Top and lower breed samples

![Lower breed samples](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/Low%20breed%20chart.png)

![Top breeds](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/Top%20breed%20chart.png)

---

Distribution of the file sizes in the two samples

![File size distribution](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/File_size_distribution.png)

The dog images sizes have a right skew distribution with 4% of the files bigger than 400K Bytes, the human files have a normal distribution

## Methodology

### Data Preprocessing

The images dataset are loaded through the use of the load_files function from the scikit-learn library and different variables for the models are generated

* `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
* `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels
* `dog_names` - list of string-valued dog breed names for translating labels

```
def load_dataset(path):
    """ Function to load the images and generate different variables

        Params:
        path : str, path to the images

        Returns:
        dog_files, dog_targes : numpy array with the files names and the encoded targets
    """

    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets
```

The dog dataset is splitted in three groups to train, validate and test the different models

The human dataset is loaded using the glob function

```
human_files = np.array(glob("../../../data/lfw/*/*"))
```

For the OpenCV face detector, it is standard procedure to convert the images to grayscale. The detectMultiScale function executes the classifier stored in face_cascade and takes the grayscale image as a parameter.

```
# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**For the keras CNN models these are the pre-processing steps:**

* Convert the input to a 4D tensor, with shape (nb_samples,rows,columns,channels),
  * nb_samples corresponds to the total number of images (or samples)
  * rows, columns, and channels correspond to the number of rows, columns, and channels for each image.
* The path_to_tensor function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.
* The function first loads the image and resizes it to a square image that is 224×224 pixels.
* Next, the image is converted to an array, which is then resized to a 4D tensor.
* In this case, since we are working with color images, each image has three channels, the returned tensor will always have shape (1,224,224,3).
* nb_samples is the number of 3D tensors (where each 3D tensor corresponds to a different image) in the dog images dataset!

```
from keras.preprocessing import image      
from tqdm import tqdm

def path_to_tensor(img_path):

    """ Function that takes a numpy array of string-valued image paths as input and returns a 4D tensor

        Params:
        img_path: string, path to the image

        Returns
        4D tensor with shape 1, 224, 224, 3)

    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

Moreover getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing. First, the RGB image is converted to BGR by reordering the channels and all pre-trained models have the additional normalization step that the mean pixel, This is implemented in the imported function **preprocess_input**

```
img = preprocess_input(path_to_tensor(img_path))
```

In addition for the transfer learning CNN models the images are rescaled dividing every pixel in every image by 255.

```
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

---

# Implementation

**Three computer methods are used to solve this problem:**

```
- OpenCV framework to recognize a human face
- A pre-trained ResNet-50 model to detect dogs
- CNN with transfer learning to classify dog breeds
```

### **Classifying human faces using OpenCV framework**

OpenCV's implementation of Haar feature-based cascade classifiers is used to detect human faces in images. OpenCV provides many pre-trained face detectors.

```
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)
```

Algorithm to detect human faces

```
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    ''' Function to detect is there is a human face in the image
  
        Params: img_path, string, path to the image
  
        Returns:
        A boolena variable indicating if a face is detected
  
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
  
    return len(faces) > 0

```



This algoritmn can also make a rectangle showing the face detected

![face](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/hilary.png)

The algorithm is tested with the two datasets (human faces images and dog images) and 100 random images, these are the results:

- Peformace of faces detected in the human files: **100.00%**
- Peformace of faces detected in the dog files: **11.00%**

### Detecting dogs using a pre-trained ResNet-50 model

The Resnet have been trained on ImageNet, a very large and popular dataset used for image classification and other vision tasks. ImageNet contains over 10 million images with 1000 different categories. Given an image, this pre-trained ResNet-50 model returns a prediction from the available categories in ImageNet.

First, the model is downloaded with weights that have been trained on ImageNet

```
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

**Making Predictions with ResNet-50**

Once the image is formatted is supplied to the Resnet-50 model and extract the predictions, this is accomplished using the `predict` method, which returns an array whose **𝑖**i-th entry is the model's predicted probability that the image belongs to the **𝑖**i-th ImageNet category.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

This is implemented in the `ResNet50_predict_labels` function below.

```
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

A dog detector is written using the `ResNet50_predict_labels` based on the range 151 to 268 which are the dog categories

def dog_detector(img_path): ''' Function thatreturns "True" if a dog is detected in the image stored at img_path

```
Params:
img_path, sting, path to the images

Returns:
A boolen variable if a dog is detected or not
'''

prediction = ResNet50_predict_labels(img_path)
return ((prediction <= 268) & (prediction >= 151))
```

The algorithm is tested with the two datasets (human faces images and dog images) and 100 random images, these are the results:

* Peformace of dogs detected in the human files: **0.00%**
* Peformace of dogs detected in the dog files: **100.00%**

### Create a CNN to Classify Dog Breeds using CNN

Three different CNN models are created using Keras:

* A model from Scratch
* Models with transfer learning
  * VGG16 model
  * Inception V3 model

**All the models created follow this methodology:**

* Pre process the data
* Create the model
  * Define the different convolutional layers (Average Pooling, Max Pooling, Conv2D)
  * Dropouts (define the dropout percentage)
  * Activation functions (Relu, softmax)
  * Dense layers (number of nodes)
* Compile the model
  * optimizer (rmsprop, adam, sgd)
  * loss function (categorical_crossentropy)
  * metrics (accuracy)
* Train the model
  * epochs
  * batch size
* Save the Model with the Best Validation Loss
* Load the Model with the Best Validation Loss
* Test the model
* Evaluate model performance with the test set and get the accuracy metrics

### A CNN model from scratch

In this model a CNN is created using a combination of different convolutional layers, filters, dropout and dense layers

```
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.4))
model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(133, activation='softmax'))

model.summary()
```

![Scratch model architecture](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/Scratch%20model%20architecture.png)

**Compile the model**

The model is compiled using the following parameters:

* optimizer = rmsprop
* loss = categorical_crossentropy
* metrics = accuracy

```
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Train the model**

Using the train and validation images the model is train and validated, using 20 epochs and a batchsize of 20

```
from keras.callbacks import ModelCheckpoint  

epochs = 20

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

**Load the model**

```
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

**Test the model**

```
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

After testing this model we get this performace:

* Test accuracy: **18.1818%**

### VGG16 model with transfer learning

Transfer learning is used to reduce training time without sacrificing accuracy

Obtain bottleneck features

```
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

From here all the steps are similar to the scrath model we described before

Model Architecture

The model uses the the pre-trained VGG-16 model, the last convolutional output of VGG-16 is fed as input to the model. A global average pooling layer and a fully connected layer is added, where the latter contains one node for each dog category equipped with a softmax function to predict the dog probability.

![VGG16 Model](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/VGG16%20model.png)

After compiling, training and testing the model, the accuracy result is: **39.83**%

### Inception V3 model with transfer learning

Another model available in keras is Inception V3, it has 159 layers and an architecture similar like the VGG16 is created to predict the dog breeds

Model architecture

* A global average pooling is added to the last convolutional layer of the inception model
* Dropout is added to avoid overfitting
* A dense layer with 256 nodes is added
* Another dropout is added
* A final dense layer with the 133 output corresponding to the 133 dog categories and softmax function is added to predict the breeds

```
Inception_model = Sequential()
Inception_model.add(GlobalAveragePooling2D(input_shape=train_Inception.shape[1:]))
Inception_model.add(Dropout(0.45))
Inception_model.add(Dense(256, activation='relu'))
Inception_model.add(Dropout(0.45))
Inception_model.add(Dense(133, activation='softmax'))
Inception_model.summary()
```

![Inception Model](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/inception%20model.png)

After training and testing the model, the accuracy is: **80.3828%**

## Refinement

Considering the previous results,  two algorithms will be improved:

* OpenCV to classify images
* Inception V3 to classify dog breeds

The Dog detector is not optimized because the classifying results is 100%

**OpenCV**

For the face_cascade.detectMultiScale two hyperparameters are tuned to improve the face detection:

* **scaleFactor** that controls how the input image is scaled prior to the detection, the image is scaled up or down, default = 1.1
* **minNeighbors** determines how robust each detection must be in order to be reported, the default is 3

After testing different values, these ones improve the algorithm accuracy:

* **scaleFactor** = 1.35
* **minNeighbors** = 4

```
face_cascade.detectMultiScale(gray,scale,minNeighbors)
```

**Inception V3**

The model Inception gets the best results with an accuracy of 80.38%, to improve this model two strategies are used:

* Use different hyperparameters:

  * Optimizer
  * epochs: [20, 30, 40]
* Use K-fold Cross Validation

Keras provides different algorithms to optimize the model, three of them are tested to see if the model improves the accuracy:

* adam
* rmsprop
* sgd

The optimizer is specified as a parameter when the model is compiled:

```
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

The optimizer has another hyperparameter like: learning rate, momentum, among others but the default values are used

In addition with these optimizers the model is trained with different epochs [30, 40, 50] to see if a better accuracy is obtained, the epocs are specified when the model is trained:

```

model.fit(inputs[train], targets[train], validation_data=(inputs[validate], targets[validate]), epochs=30, batch_size=25)
```

**Cross Validation**

The dataset as we saw before in the data visualization part, is not totally balanced, some breeds hava more than 90 samples, and others has less than 40, in this scenario a technique called k-folding cross validation could be used, instead of split the dataset in training and validation fix sets, different sets are splitted to ensure that all the training and validation set are relatively unbiased, when the process finishes  the model has been trained using most of the images and the model with the best accuracy result is chosen.

The steps to train the model with cross validation are:

* Merge the original training and validation dataset
  ```
  inputs = np.concatenate((train_Inception, valid_Inception), axis=0)
  targets = np.concatenate((train_targets, valid_targets), axis=0)
  ```
* Define the K-fold Cross Validator with the number of folds using the sklearn libray
  ```
  kfold = KFold(n_splits=num_folds, shuffle=True)
  ```
* Follow the steps to create, compile, train and evaluate the model with the different folds
* Select and save the model with the best weights to get the better accuracy

---

## Results

These are the results obtained once the refinement is applied to the human face detector and dog breed detector algorithms

**OpenCV Algorithm**

This is the improvement after changing the default parameters for the face_cascade.detectMultiScale algorithm

![opencv](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/open%20cv%20results.png)

An improvement of 10% is achieved, it means that a dog face is not detected as a human, even though in some dog images there is a human with a dog, this could explain why some human faces are in the dog images

**CNN Model Evaluation and Validation**

This is the performance of the different CNN models

![Models](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/models%20accuracy.png)

The model from scratch has the worst performance, it shows that a simple CNN can not have a good accuracy if its not trainned with enough images and the complexity of the network is increased but this requieres more computational power to trained the network.

For the two models with transfer learning, the Inception model has the best performance, two factors could explain the results:

* The pretrained model performance per se
* The architecture used on top of the model, where for the Inception model dropout is applied; this avoid overfitting and get better results

Once the Inception model is chosen we try to improve it, tuning the different hyperparameters (optimizer and epocs), here is the accuracy obtained:

![hyperparameter models](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/hyperparameter%20results.png)

From the previous results, the accuracy did not get a good improvement tuning the hyperparameter, less than a 2% improvement is achieved across all the possible combinations

From this point we select the model with **rmsprop** and try to improve using k-folding cross validation to train the model with all the possible image combinations, these are the results using a k-fold = **8**

![]()

![Validation](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/Validation%20accuracy.png)

The result shows that even though the dog image data set is not equally distributed, when trainning the model with different sample distributions the accuracy did not get a big improve, all the results are around 2% margin.

Applying the model with the best validation scores to the test set, the final result for the accuracy is:  **82.10** , which is very similar to the previous results, so the dataset is very stable to train the model, but to improve the results different approaches must be taken and further investigation is needed.

### Justification

The models with the best scores are selected to create the web app, so this are the models selected:

* To classify images: **face_cascade.detectMultiScale** with the tune parameters
* To classify dogs: **pre-trained ResNet-50**
* to classify dog breed: **Inception V3 model**

---

## Web app

A web app is created using the [Streamlit](https://streamlit.io/) framework, an easy way to build interactive data apps, and integrated all the models with the best results.

This is the general flowchart for the algorith used in the webapp:

![General Algorithm](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/Flow2.png)

This is how the web app looks like:

![Web app](https://github.com/rafaelmata357/Data-Science-Udacity/blob/master/dog%20breed%20classifier/test_images/web%20app.png)

---

## Conclusion

* CNN is an efficient computer algorithm  to classify images and detect different features, but doing a CNN from scratch requires a lot of time and power  to get an acceptable performance.
* Using transfer learning improves a lot a image classifier and reduce the training time required to get better results
* Different image scenarios  can be added to generalize the app functionality and detect other images or combinations like dog and human in the same picture or other kind of images like cats.
* Looking forward to improve the accuracy other techniques can be used like image augmentation
* A web app is a good form to integrate the different models and make it available to the final user


## Acknowledgements

The completion of this project was done thanks to the support of the Udacity team and instructors, the initial Jupyter notebook guide was key to develop and complete the project.

## License

The code follows this license: [https://creativecommons.org/licenses/by/3.0/us/](https://creativecommons.org/licenses/by/3.0/us/)


---
