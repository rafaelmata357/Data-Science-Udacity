# Understanding the Boston Airbnb market for 2016-2017 Period

## Installation:

Clone the repository to the local machine

`$ git clone https://github.com/rafaelmata357/Data-Science-Udacity/tree/master/Airbnb`

This notebook uses this libraries

- pandas
- numpy 
- matplotlib
- seaborn 
- nltk
- wordcloud
- sklearn
- folium

The Airnbn dataset for Boston can be found [here](https://www.kaggle.com/airbnb/boston)
The geojson file for Boston can be forun [here](http://data.insideairbnb.com/united-states/ma/boston/2020-10-24/visualisations/neighbourhoods.geojson)

The python version used: **3.8**


## Project Motivation

As part of the Udacity Data Science nanodegree this is an analyzis of the Boston´s airnbnb dataset using the CRISPM methodology and publish an article with the findings.


## Files in the repository

- airbnb_V2.ipynb : Jupyter notebook with the source code for the analysis
- README.md : This file


## How To Interact With Your Project 

- The Jupyter notebook in this repository follows the CRISPM methodology:

1. Business Understanding

    - Starting in august 2008 Airbnb  an American company  introduced an app for online lodging disrupting the market,  primarily for home stays  this app offers  the house owners the opportunity to rent available house rooms and give the guests different tourism experiences, this model has increased year over year and Boston is not the exception

2. Data Understanding
    - The dataset contains three tables:
        - Listings : with the listings offering, contains 3585 listings for 25 neighborehoods with more than 90 features 
        - Calendar : with the listings availability, contains more than 643000 entries
        - Reviews  : with the listings reviews, contains more than 68000 reviews
    - Different descriptive statistics are used for EDA and data visualization:
        - Histograms
        - WordClouds
        - Choroplet maps
        - Box plots
        - Heat Maps

3. Data preparation
    - A data cleaning process is followed including:
        - Removed columns with unique values
        - Inputing: the NaN values are filled with the most common value for categorical features, and the average for numerical features
        - Remove the outliers
        - Codify the categorical features
        - Use NLTK library to extract the main feature from the transit and neighborhood description columns

4. Data Modeling
    - To model a linear regression two methods are used:
        - Linear regression
        - Random Forest regression
    - The dataset is normalized using Min-Max scaler
    - The data is splitted in two parts:
        - A trainning dataset containing the 80% of the data
        - A test dataset containing the 20% of the data

5. Evaluate the Results
    - The results are evaluated using three metrics:
        - MAS Error
        - MSE Error
        - R2 Score

6. Deploy
    - The Deploy is an article posted in (medium)[https://medium.com/] and can be foud here[article](https://rafaelmata357.medium.com/a-guide-tour-to-the-boston-airbnbs-market-d7689ddf9d6c)


## Terms of use:

This notebook is not a formal study, is an initial analyzis and is made only to show how to use the CRISP Methodology to understand a dataset an answer question related to the Boston  Airbnb market, . 

## License:

The code follows this license: https://creativecommons.org/licenses/by/3.0/us/
