from datasets import load_dataset 
import nltk 

import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split

from collections import Counter 
import random
import math 

from src.evaluation import * 
from src.generation import * 
from src.ngram_models import * 
from src.utils import * 

if __name__ == '__main__': 
    text_data = load_dataset('ptb_text_only', split = 'train') 
    text_data = pd.DataFrame(text_data).values.tolist()

    train_data, test_data = train_test_split(text_data, test_size = 0.2, random_state = 10)

    train_tokens = preprocess_ptb(train_data) 
    test_tokens = preprocess_ptb(test_data) 

    unigram_model = build_unigram_model(train_tokens)
    bigram_model = build_bigram_model(train_tokens)
    trigram_model = build_trigram_model(train_tokens)

    print('Unigram test perplexity:', calculate_perplexity(unigram_model, test_tokens, 1))
    print('Bigram test perplexity:', calculate_perplexity(bigram_model, test_tokens, 2))
    print('Trigram test perplexity:', calculate_perplexity(trigram_model, test_tokens, 3))

    smoothed_trigram_prob, V = trigram_model_leplace(train_tokens)
    print('Leplace smoothed trigram perplexity:', calculate_trigram_smoothed_perplexity(smoothed_trigram_prob, test_tokens, V))

    print('Fallback Perplexity:', calculate_fallback_perplexity(trigram_model, bigram_model, unigram_model, test_tokens))