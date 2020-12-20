#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER   : Rafael Mata M.
# DATE CREATED :  04 December 2020                                 
# REVISED DATE :  18 December 2020
# PURPOSE: Create a program to create a web app than tracks the Covid daily information for different countries from JHU dataset
#
# 


# Imports python modules


import pandas as pd
import plotly.graph_objs as go
import datetime
from datetime import date, timedelta
import numpy as np

#URL to get the raw data JHU CSSE COVID-19 Dataset

URL_ACCUMULATED_CASES ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
URL_RECOVERED = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
URL_DEATHS = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

country_default = 'Costa Rica'


def get_and_cleandata(URL, start_date, end_date):

    '''
    Download the data from the JHU github repository and prepare the data to graph
      
    Args:
        URL : url to the github raw data from JHU updated daily
       
   
    
    Returns:
     dataset  : a pandas DF with the comple covid dataset  
     population: dataset with the population per country 
    '''
   
    dataset = pd.read_csv(URL,index_col=0)  #Se lee los datos de github en formato .csv
 
    columna = dataset.columns
    dataset.set_index(columna[0], inplace=True)  # Para regenerar el indice por pais
    dataset.drop(['Lat', 'Long'], axis=1, inplace=True)  # Para eliminar las colunnas de Lat y Long


    #population = pd.read_excel('population.xlsx', 'data', index_col=0, na_values=['NA'])
    subdata = dataset.groupby('Country/Region', axis=0).sum()        #Sum the daily data by country
    subdata =  columns_dataset_to_timestamp(subdata) #Change the columns format date to timestamp

    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    #Filter the dataset using the start date selected

    subdata = subdata.loc[:,start_date_obj:end_date_obj]
    
    #Sort datasets using last day data as as key
    #subdata = sort_dataset(subdata)
    
    return subdata

def columns_dataset_to_timestamp(dataset):
    '''
    From the Dataset this function take the columns in string format and conver to timestamp 
    
      
    Args:
        dataset : pandas datarframe dataset with string date format columns
       
    Returns:
         datset: with columns in timestamp format
    '''

    columns = list(dataset.columns)
    dataset.columns = pd.to_datetime(columns)

    return dataset

def get_daily_values(dataset):
    '''
    From the accumulated Dataset this function calculate the daily values
      
    Args:
       
        dataset : dataset with the accumulated cases
       
    Returns:
         daily_dataset: data frame with the daily values
    '''
    columns = dataset.columns
    daily_value = np.empty(len(dataset)) # create a temporary numpy array to store the daily values 
    daily_dataset = pd.DataFrame(index=dataset.index)
    for country in columns: 
        country_data = dataset[country] 
        for i in range(len(country_data)-1): 
            daily_value[-i-1] = country_data[-i-1] - country_data[-i-2] 
            daily_value[0] = country_data[0] 
        daily_dataset[country] = daily_value
    
    return daily_dataset

def rolling_window(dataset, window, countries):

    rolling = dataset[countries].rolling(window)
    
    return rolling



def return_figures(countries=country_default):
    """Creates four plotly visualizations, function based on original example from Udacity DataScience Nanodegree

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    # Read the datasets

    #Filter by country
    if not bool(countries):
        countries = country_default

    if not type(countries) is str:
        country = countries.to_dict()
        country = list(country.values())[0]
    else:
        country = 'Costa Rica'
    
    today = date.today()
    yesterday = today - timedelta(days=1)
    end_date = yesterday.strftime("%Y-%m-%d") 
    start_date = '2020-03-01'
    accumulated_dataset = get_and_cleandata(URL_ACCUMULATED_CASES, start_date, end_date)
    daily_dataset = get_daily_values(accumulated_dataset.T) #Calculate the daily values from accumuated dataset
    #recovered_dataset, population = get_and_cleandata(URL_RECOVERED, start_date, end_date)

    rolling_daily = daily_dataset[country].rolling('14D') 
       
    death_dataset = get_and_cleandata(URL_DEATHS, start_date, end_date)
    daily_death_dataset = get_daily_values(death_dataset.T)

    rolling_daily_deaths = daily_death_dataset[country].rolling('14D') 


  


    accumulated_dataset = accumulated_dataset.loc[country]
    #recovered_dataset = recovered_dataset.loc[countries]
    death_dataset = death_dataset.loc[country]

  # first chart plots the Covid accumulated Cases 
  # as a line chart
    
    graph_one = []
    
    x_val =  accumulated_dataset.index.strftime('%d/%m').tolist()
    y_val = accumulated_dataset.tolist()  
    graph_one.append(
        go.Scatter(
        x = x_val,
        y = y_val,
        mode = 'lines',
       )
      )

    #Costa Rica 

   
    layout_one = dict(title = 'Covid-19 Accumulated cases for {}'.format(country),
                xaxis = dict(title = 'Day',),
                yaxis = dict(title = 'Cases'),
                )

# second chart plots ararble land for 2015 as a bar chart    
    graph_two = []
    x_val = death_dataset.index.strftime('%d/%m').tolist()
    y_val = death_dataset.tolist()

    graph_two.append(
       go.Scatter(
        x = x_val,
        y = y_val,
        mode = 'lines',
       )
    )

    layout_two = dict(title = 'Accumulated Covid Death Cases for {}'.format(country),
                xaxis = dict(title = 'Day',),
                yaxis = dict(title = 'Deaths'),
                )


# third chart plots percent of population that is rural from 1990 to 2015
    graph_three = []
    df_weekly = daily_dataset.groupby(daily_dataset.index.isocalendar().week)[country].sum()

    graph_three.append(
      go.Bar(
      x = df_weekly.index.tolist(),
      y = df_weekly.tolist(),
      )
    )

    layout_three = dict(title = 'Accumulated cases by week for {}'.format(country),
                xaxis = dict(title = 'Week',),                  
                yaxis = dict(title = 'Cases'),
                )
    
# fourth chart Death cases by week
    graph_four = []
    df_weekly = daily_death_dataset.groupby(daily_death_dataset.index.isocalendar().week)[country].sum()

    graph_four.append(
      go.Bar(
      x = df_weekly.index.tolist(),
      y = df_weekly.tolist(),
      )
    )

    layout_four = dict(title = 'Covid Death cases by week for {}'.format(country),
                xaxis = dict(title = 'Week'),
                yaxis = dict(title = 'Deaths'),
                )

   # fith chart 14 days accumalated cases rolling window
    graph_five = []

    
  
    x_val =  rolling_daily.sum().index.strftime('%d/%m').tolist()
    y_val = rolling_daily.sum().tolist() 
    graph_five.append(
        go.Scatter(
        x = x_val,
        y = y_val,
        mode = 'lines',
       )
      )

    layout_five = dict(title = 'Accumulated cases 14 days window for {}'.format(country),
                xaxis = dict(title = 'Day',),
                yaxis = dict(title = 'Cases'),
                )

    # sixth chart 14 days accumalated cases rolling window

    graph_six = []
 
    x_val =  rolling_daily_deaths.sum().index.strftime('%d/%m').tolist()
    y_val =  rolling_daily_deaths.sum().tolist() 
    graph_six.append(
        go.Scatter(
        x = x_val,
        y = y_val,
        mode = 'lines',
       )
      )

    layout_six = dict(title = 'Accumulated death cases 14 days window for {}'.format(country),
                xaxis = dict(title = 'Day',),
                yaxis = dict(title = 'Cases'),
                )
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))
    figures.append(dict(data=graph_five, layout=layout_five))
    figures.append(dict(data=graph_six, layout=layout_six))
    
    return figures