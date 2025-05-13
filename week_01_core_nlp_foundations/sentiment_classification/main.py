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

from src.preprocessing import preprocess

if __name__ == '__main__': 
    preprocess()