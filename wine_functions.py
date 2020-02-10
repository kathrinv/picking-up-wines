import config
from bs4 import BeautifulSoup
import requests
import mysql.connector 
from mysql.connector import errorcode
import json
import datetime
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import re
import os
import time
import random
import nltk

from os import system
from math import floor
from copy import deepcopy
import itertools

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import tree 
from sklearn.preprocessing import OneHotEncoder
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score, f1_score, precision_score,\
                            recall_score, roc_auc_score, mean_squared_error,\
                            classification_report, confusion_matrix, roc_curve, auc



from sklearn.base import clone
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import xgboost as xgb


from sklearn.externals.six import StringIO 
from IPython.display import Image  
import logging
import pydotplus


# Webscraping Functions

def scrape_wine_details(url):
    driver.get(url)
    sequence = [x/10 for x in range(20, 30)]
    time.sleep(random.choice(sequence))
    see_more_reviews = driver.find_element_by_css_selector('a.viewMoreModule_link.viewMoreReviews')
    try:
        see_more_reviews.click()
    finally:
        wine_name = driver.find_element_by_class_name('pipName').text
        vintage = re.findall("[12]\d{3}", wine_name)
        vintage = 'None' if vintage == [] else vintage[-1]
        wine_type = driver.find_element_by_class_name('prodItemInfo_varietal').text
        location = driver.find_element_by_class_name('prodItemInfo_originText').text
#         country = re.findall(r"(?<=, )[a-zA-Z ]+$|[a-zA-Z ]+$", location)
#         country = 'None' if country == [] else country[0]
#         region = location[:-len(country)-1]
        critics = driver.find_elements_by_xpath('/html/body/div[1]/main/section[2]/div[2]/div[1]/div[2]/ul[1]')
        critic_names, critic_ratings = get_critic_names_and_ratings(critics)
        price = driver.find_element_by_class_name('productPrice_price-reg').text
        if price == '':
            price = driver.find_element_by_class_name('prodItemStock_soldOut-smallText').text
            price = price.replace('$', '')
        price = price.replace(',', '')
        price = float(price.replace(' ', '.'))
        sale_price = driver.find_element_by_class_name('productPrice_price-sale').text
        try:
            sale_price = sale_price.replace(',', '')
            sale_price = float(sale_price.replace(' ', '.'))
        except:
            sale_price = price
        discount = driver.find_element_by_class_name('productPrice_savings-percentage').text
        try:
            discount = float(discount)
        except:
            discount = 0
        try:
            user_ratings = driver.find_elements_by_class_name('averageRating_average')
            for rating in user_ratings:
                if rating.text == '':
                    user_rating = -1
                else:
                    user_rating = float(rating.text)
        except:
            user_ratings = -1
        try:
            user_rating_cnt = driver.find_elements_by_class_name('averageRating_number')
            for cnt in user_rating_cnt:
                if cnt.text == '':
                    num_reviews = 0
                else:
                    num_reviews = int(cnt.text)
                num_reviews = int(cnt.text.replace(' Ratings', ''))
        except:
            num_reviews = 0
        try:
            sizes = driver.find_elements_by_css_selector('span.prodAlcoholVolume')
            for size in sizes:
                if size.text == 0:
                    bottle_size = "Unknown"
                else:
                    bottle_size  = size.text.replace(' /', '')
        except:
            bottle_size = "Unknown"
        try:
            alcohol_content = driver.find_elements_by_css_selector('span.prodAlcoholPercent_percent')
            for alch in alcohol_content:
                if len(alch.text) == 0:
                    alcohol = -1
                else:
                    alcohol  = float(alch.text)
                    break
        except:
            alcohol = -1
        attributes = driver.find_elements_by_xpath('/html/body/div[1]/main/section[3]/section[2]/ul/li')
        wine_attributes_dict = get_wine_attributes(attributes)
        notes = driver.find_element_by_class_name('viewMoreModule_text').text
        try:
            critic_notes_raw = driver.find_elements_by_class_name('pipProfessionalReviews_review')
            critic_notes = [cn.text for cn in critic_notes_raw]
        except:
            critic_notes = []
        winery = driver.find_element_by_class_name('pipWinery_headlineLink').text
        wine_dict = {'wine_name': wine_name, 'vintage': vintage, 'wine_type': wine_type,
                    'country': location, 'region': location, 'critic_names': critic_names,
                    'critic_ratings': critic_ratings, 'price': price, 'sale_price': sale_price,
                    'discount': discount, 'user_rating': user_rating,
                    'num_reviews': num_reviews, 'bottle_size': bottle_size,
                     'alcohol': alcohol, 'wine_attributes': wine_attributes_dict,
                     'notes': notes, 'critic_notes': critic_notes, 'winery': winery}
        return wine_dict

    
    
def format_for_db(url, wine_dict):
    wine_details = []
    critic_details = []
        
    wine_details_tuple = (url, wine_dict['wine_name'], wine_dict['vintage'],\
            wine_dict['country'], wine_dict['region'], wine_dict['wine_type'],\
            wine_dict['wine_attributes']['category'],\
            wine_dict['alcohol'], wine_dict['price'], wine_dict['sale_price'],\
            wine_dict['discount'], wine_dict['bottle_size'], wine_dict['user_rating'],\
            wine_dict['num_reviews'], wine_dict['winery'], wine_dict['notes'],\
            wine_dict['wine_attributes']['screw_cap'],\
            wine_dict['wine_attributes']['boutique'],\
            wine_dict['wine_attributes']['great_gift'],\
            wine_dict['wine_attributes']['green_wine'],\
            wine_dict['wine_attributes']['collectible'])
    wine_details.append(wine_details_tuple)
    
    if wine_dict['critic_names'] == [] or (wine_dict['critic_names'][0] == '' and len(wine_dict['critic_names']) == 1):
        critic_details = []
    else:
        for idx, critic in enumerate(wine_dict['critic_names']):
            critic_details_tuple = (url, wine_dict['wine_name'],\
                             critic, wine_dict['critic_ratings'][idx],\
                             wine_dict['critic_notes'][idx])
            critic_details.append(critic_details_tuple)
    
    return wine_details, critic_details


def get_critic_names_and_ratings(critics):
    critic_names = []
    critic_ratings = []
    if critics[0] == 0:
        critic_names = []
        critic_ratings = []
        return critic_names, critic_ratings
    else:
        ratings_and_names = critics[0].text.split('\n')
        for idx, i in enumerate(ratings_and_names):
            if idx % 2 == 0:
                critic_names.append(i)
            else:
                critic_ratings.append(i)
        return critic_names, critic_ratings

    
def get_wine_attributes(attributes):
    wine_attributes_dict = {'category':  "Unknown",
                            'boutique': False,
                            'great_gift': False,
                            'green_wine': False,
                            'collectible': False,
                            'screw_cap': False}
    wine_attributes = [i.get_attribute('title') for i in attributes]
#     print(wine_attributes)
    if 'Boutique' in wine_attributes:
        wine_attributes_dict['boutique'] = True
    if 'Great Gift' in wine_attributes:
        wine_attributes_dict['great_gift'] = True
    if 'Green Wine' in wine_attributes:
        wine_attributes_dict['green_wine'] = True
    if 'Collectible' in wine_attributes:
        wine_attributes_dict['collectible'] = True
    if 'Screw Cap' in wine_attributes:
        wine_attributes_dict['screw_cap'] = True
    if 'Red Wine' in wine_attributes:
        wine_attributes_dict['category'] = 'Red Wine'
    if 'White Wine' in wine_attributes:
        wine_attributes_dict['category'] = 'White Wine'
    if 'Pink and Rosé' in wine_attributes:
        wine_attributes_dict['category'] = 'Rose'
    if 'Sparkling & Champagne' in wine_attributes:
        wine_attributes_dict['category'] = 'Sparkling'
    return wine_attributes_dict


# Functions to set up relational AWD database and populate data

def create_wine_details_table():
    """One-time table creation to house wine data"""
    cnx = mysql.connector.connect(
        host = config.host,
        user = config.user,
        password = config.passwd,
        database = config.database)
    cursor = cnx.cursor()
    cursor.execute("""CREATE TABLE details(
                      url TEXT,
                      name TEXT,
                      vintage VARCHAR(4),
                      country TEXT,
                      region TEXT,
                      type TEXT,
                      category TEXT,
                      alcohol FLOAT,
                      base_price FLOAT,
                      current_price FLOAT,
                      discount FLOAT,
                      size TEXT,
                      user_rating FLOAT,
                      num_reviews NUMERIC,
                      winery TEXT,
                      notes TEXT,
                      screw_cap BOOLEAN,
                      boutique BOOLEAN,
                      great_gift BOOLEAN,
                      green_wine BOOLEAN,
                      collectible BOOLEAN)""")
    cnx.commit()
    cursor.close()

def create_critic_rating_table()
    """One-time table creation to house wine critical acclaim data"""
    cnx = mysql.connector.connect(
        host = config.host,
        user = config.user,
        password = config.passwd,
        database = config.database)
    cursor = cnx.cursor()
    cursor.execute("""CREATE TABLE critical_acclaim(
                      wine_url TEXT,
                      wine_name TEXT,
                      critic_name TEXT,
                      critic_rating NUMERIC,
                      critic_notes TEXT)""")
    cnx.commit()
    cursor.close() 
   
    
def upload_to_db_wines(wine_details):  
    """
    Sends a list of tuples containing wine details
    to the AWS DB
    """
    cnx = mysql.connector.connect(
        host = config.host,
        user = config.user,
        password = config.passwd,
        database = config.database)
    cursor = cnx.cursor()
    stmt = """INSERT INTO details (
                  url,
                  name,
                  vintage,
                  country,
                  region,
                  type,
                  category,
                  alcohol,
                  base_price,
                  current_price,
                  discount,
                  size,
                  user_rating,
                  num_reviews,
                  winery,
                  notes,
                  screw_cap,
                  boutique,
                  great_gift,
                  green_wine,
                  collectible) 
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s)
                  """
    for i in wine_details:
        cursor.execute(stmt, i)
        cnx.commit()
    cursor.close()
    
    
def upload_to_db_critics(critic_details):
    """
    Sends a list of tuples containing wine critic
    reviews and ratings . to the AWS DB
    """
    cnx = mysql.connector.connect(
        host = config.host,
        user = config.user,
        password = config.passwd,
        database = config.database)
    cursor = cnx.cursor()
    stmt = """INSERT INTO critical_acclaim (
                  wine_url,
                  wine_name,
                  critic_name,
                  critic_rating,
                  critic_notes)
                  VALUES (%s, %s, %s, %s, %s)
                  """
    for i in critic_details:
        cursor.execute(stmt, i)
        cnx.commit()
    cursor.close()

    
# Database Query to Pull Complete Data

def get_complete_wine_data():
    """
    Accesses the AWS database and joins the wine details table
    and the critical acclaim table to obtain information
    about each wine. Details from the critical acclaim table
    are lost since there could be more than one rating per wine.
    """
    cnx = mysql.connector.connect(
        host = config.host,
        user = config.user,
        password = config.passwd,
        database = config.database)
    cursor = cnx.cursor()
    q = """
        SELECT d.*, AVG(c.critic_rating)  FROM details d
        LEFT JOIN critical_acclaim c
        ON c.wine_url = d.url
        GROUP BY d.name;
        """
    
    cursor.execute(q)
    results = pd.DataFrame(cursor.fetchall())
    results.columns = [x[0] for x in cursor.description]
    cursor.close()
    return results