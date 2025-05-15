import spacy
import spacy.cli
from spacy.tokens import DocBin

import nltk
from nltk.corpus.reader import ConllCorpusReader

from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_f1_score

import time

from collections import Counter

from seqeval.metrics import classification_report, f1_score

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

import re

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score, PredefinedSplit
from sklearn.metrics import make_scorer
from scipy.stats import loguniform

from src.crf_utils import * 
from src.spacy_utils import * 

if __name__ == '__main__': 
    # SpaCy pre-trained model 
    spacy.cli.download('en_core_web_trf') 
    nlp = spacy.load('en_core_web_trf')

    # load data
    train = ConllCorpusReader('./data/', 'eng.train', ['words', 'pos', 'ignore', 'chunk'])
    test_a = ConllCorpusReader('./data/', 'eng.testa', ['words', 'pos', 'ignore', 'chunk'])
    test_b = ConllCorpusReader('./data/', 'eng.testb', ['words', 'pos', 'ignore', 'chunk'])

    train_sentences = train.iob_sents()
    test_sentences = test_a.iob_sents() + test_b.iob_sents()

    # remove sentences that are empty strings
    train_sentences = [sent for sent in train_sentences if len(sent) > 0]
    test_sentences = [sent for sent in test_sentences if len(sent) > 0]

    pred, true_labels = evaluate_spacy_ner(nlp, test_sentences, label_map)

    # print performance
    print('SpaCy Pre-Trained Model Performance:') 
    print(classification_report(true_labels, pred))
    print()

    # CRF Model 
    # get X_train and y_train
    X_train = [create_sentence_features(sentence) for sentence in train_sentences]
    y_train = [get_labels(sentence) for sentence in train_sentences]

    # define and fit model
    crf_model = CRF(
        algorithm = 'lbfgs',
        c1 = 0.1,
        c2 = 0.1,
        max_iterations = 100,
        all_possible_transitions = True
    )
    crf_model.fit(X_train, y_train)

    # get model performance on the test set
    X_test = [create_sentence_features(sentence) for sentence in test_sentences]
    y_test = [get_labels(sentence) for sentence in test_sentences]

    y_pred = crf_model.predict(X_test)
    
    print('CRF Model Performance:') 
    print(classification_report(y_test, y_pred))
    print()

    # tune model 
    crf_model = CRF(
        algorithm = 'lbfgs',
        max_iterations = 100,
        all_possible_transitions = True
    )

    # define parameter space
    param_space = {
        'c1': loguniform(0.01, 1),
        'c2': loguniform(0.01, 1)
    }

    def custom_score(estimator, X, y):
        y_pred = estimator.predict(X)
        return flat_f1_score(y, y_pred, average = 'weighted')
    
    # set up random search 
    crf_grid = RandomizedSearchCV(
        crf_model,
        param_space,
        scoring = custom_score,
        cv = 3,
        verbose = 3,
        error_score = 'raise', 
        n_jobs = 6
    )

    crf_grid.fit(X_train, y_train)

    print('CRF best params:', crf_grid.best_params_)
    print('CRF best score:', crf_grid.best_score_)
    print()

    # evaluate best model 
    best_crf = crf_grid.best_estimator_
    y_pred = best_crf.predict(X_test)

    print('Best CRF Model Performance:')
    print(classification_report(y_test, y_pred))



