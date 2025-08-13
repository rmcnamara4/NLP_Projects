import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import spacy 
nlp = spacy.load('en_core_web_sm', disable = ['ner', 'parser'])

from sklearn.model_selection import train_test_split

import re 
import swifter 

from word2number import w2n

from src.preprocessing_utils import *

def preprocess():
    train_temp = pd.read_csv('./data/train.csv', header = None) 
    test = pd.read_csv('./data/test.csv', header = None)

    train_temp.columns = ['sentiment', 'title', 'review']
    test.columns = ['sentiment', 'title', 'review']

    train_temp = train_temp.drop(columns = 'title')
    test = test.drop(columns = 'title')

    train, _ = train_test_split(train_temp, train_size = 700_000, random_state = 432, stratify = train_temp.sentiment)
    train, val = train_test_split(train, train_size = 0.8, random_state = 432, stratify = train.sentiment)

    stop_words = set(stopwords.words('english'))

    X_train_nltk = train.review.swifter.apply(preprocess_nltk)
    X_train_spacy = preprocess_spacy(train.review.tolist())

    X_val_nltk = val.review.swifter.apply(preprocess_nltk)
    X_val_spacy = preprocess_spacy(val.review.tolist())

    X_test_nltk = test.review.swifter.apply(preprocess_nltk)
    X_test_spacy = preprocess_spacy(test.review.tolist())

    y_train = train.sentiment
    y_val = val.sentiment
    y_test = test.sentiment

    replace_dict = {1: 0, 2: 1}

    X_train_nltk.to_csv('./data/X_train_nltk.csv', header = True, index = False)
    pd.Series(X_train_spacy).to_csv('./data/X_train_spacy.csv', header = True, index = False)
    y_train.replace(replace_dict).to_csv('./data/y_train.csv', header = True, index = False)

    X_val_nltk.to_csv('./data/X_val_nltk.csv', header = True, index = False)
    pd.Series(X_val_spacy).to_csv('./data/X_val_spacy.csv', header = True, index = False)
    y_val.replace(replace_dict).to_csv('./data/y_val.csv', header = True, index = False)

    X_test_nltk.to_csv('./data/X_test_nltk.csv', header = True, index = False)
    pd.Series(X_test_spacy).to_csv('./data/X_test_spacy.csv', header = True, index = False)
    y_test.replace(replace_dict).to_csv('./data/y_test.csv', header = True, index = False)

    # save it to the common data folder as well for future usage 
    X_train_nltk.to_csv('../../data/amazon_sentiment/X_train_nltk.csv', header = True, index = False)
    y_train.replace(replace_dict).to_csv('../../data/amazon_sentiment/y_train.csv', header = True, index = False)

    X_val_nltk.to_csv('../../data/amazon_sentiment/X_val_nltk.csv', header = True, index = False)
    y_val.replace(replace_dict).to_csv('../../data/amazon_sentiment/y_val.csv', header = True, index = False)

    X_test_nltk.to_csv('./data/X_test_nltk.csv', header = True, index = False)
    y_test.replace(replace_dict).to_csv('../../data/amazon_sentiment/y_test.csv', header = True, index = False)