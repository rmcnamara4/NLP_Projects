from datasets import load_dataset 
import nltk 

import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split

from collections import Counter 
import random
import math 
import json
import os 

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

    unigram_perplexity = calculate_perplexity(unigram_model, test_tokens, 1)
    bigram_perplexity = calculate_perplexity(bigram_model, test_tokens, 2) 
    trigram_perplexity = calculate_perplexity(trigram_model, test_tokens, 3) 

    print('Unigram test perplexity:', unigram_perplexity)
    print('Bigram test perplexity:', bigram_perplexity)
    print('Trigram test perplexity:', trigram_perplexity)

    smoothed_trigram_prob, V = trigram_model_leplace(train_tokens)
    laplace_trigram_perplexity = calculate_trigram_smoothed_perplexity(smoothed_trigram_prob, test_tokens, V)
    print('Leplace smoothed trigram perplexity:', laplace_trigram_perplexity)

    fallback_perplexity = calculate_fallback_perplexity(trigram_model, bigram_model, unigram_model, test_tokens)
    print('Fallback Perplexity:', fallback_perplexity)

    results = {
        'unigram_perplexity': round(unigram_perplexity, 2), 
        'bigram_perplexity': round(bigram_perplexity, 2), 
        'trigram_perplexity': round(trigram_perplexity, 2), 
        'laplace_trigram_perplexity': round(laplace_trigram_perplexity, 2), 
        'fallback_perplexity': round(fallback_perplexity, 2)
    }

    os.makedirs('results/', exist_ok = True) 
    with open('results/perplexity_results.json', 'w') as f: 
        json.dump(results, f, indent = 2)

