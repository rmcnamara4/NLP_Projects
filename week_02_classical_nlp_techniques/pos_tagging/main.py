import pandas as pd 
import numpy as np 
import json

import nltk 
from nltk.corpus import brown 
from collections import defaultdict, Counter 

import math 

nltk.download('brown') 
nltk.download('universal_target') 

from src.evaluation import evaluate
from src.viterbi_algorithm import viterbi
from src.hmm_utils import get_emission_probability, convert_counts_to_probabilities

if __name__ == '__main__': 
    tagged_sentences = brown.tagged_sents(tagset = 'universal') 

    train_size = int(len(tagged_sentences) * 0.8) 
    train_sentences = tagged_sentences[:train_size]
    test_sentences = tagged_sentences[train_size:]

    words = [word.lower() for sent in train_sentences for word, _ in sent]
    tags = [tag for sent in train_sentences for _, tag in sent]

    unique_words = set(words)
    unique_tags = set(tags)

    word_counts = Counter(words)
    tag_counts = Counter(tags)

    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = dict(tag_counts)
    initial_counts = defaultdict(int)

    for sentence in train_sentences:
        prev_tag = None
        for i, (word, tag) in enumerate(sentence):
            word = word.lower()
            emission_counts[tag][word] += 1

            if i == 0:
                initial_counts[tag] += 1
            else:
                transition_counts[prev_tag][tag] += 1

            prev_tag = tag

    transition_probabilities = convert_counts_to_probabilities(transition_counts)
    emission_probabilities = convert_counts_to_probabilities(emission_counts)
    initial_probabilities = {
        k: v / sum(initial_counts.values()) 
        for k, v in initial_counts.items()
    }

    accuracy = evaluate(test_sentences, initial_probabilities, transition_probabilities, emission_probabilities, unique_tags)

    with open('./results/pos_tagging_results.json', 'w') as f: 
        json.dump({
            'accuracy': accuracy,
            'transition_probabilities': transition_probabilities,
            'emission_probabilities': emission_probabilities,
            'initial_probabilities': initial_probabilities
        }, f, indent = 4)

    print(f'Accuracy: {accuracy*100:.4f}%')

