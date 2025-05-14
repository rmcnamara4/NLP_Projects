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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import re 
import swifter 
import pickle

from word2number import w2n

import mlflow 
mlflow.autolog(disable = True) 

import ast 
import warnings
warnings.filterwarnings('ignore') 

import optuna 
from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback, CatBoostPruningCallback

from src.preprocessing import preprocess
from src.metrics import * 
from src.mlflow import * 

if __name__ == '__main__': 
    # preprocess dataset
    preprocess()

    # load data
    X_train = pd.read_csv('./data/X_train_nltk.csv').squeeze()
    X_val = pd.read_csv('./data/X_val_nltk.csv').squeeze()
    X_test = pd.read_csv('./data/X_test_nltk.csv').squeeze()

    y_train = pd.read_csv('./data/y_train.csv').squeeze()
    y_val = pd.read_csv('./data/y_val.csv').squeeze()
    y_test = pd.read_csv('./data/y_test.csv').squeeze()

    # drop NaN values
    inds = X_train.isna()
    X_train = X_train[~inds]
    y_train = y_train[~inds]

    inds = X_val.isna()
    X_val = X_val[~inds]
    y_val = y_val[~inds]

    inds = X_test.isna()
    X_test = X_test[~inds]
    y_test = y_test[~inds]

    # set up MLFlow for tracking
    mlflow.set_tracking_uri('./experiments/') 

    experiment_name = 'sentiment-analysis-amazon-reviews'
    mlflow.set_experiment(experiment_name) 
    experiment = mlflow.get_experiment_by_name(experiment_name) 

    experiment_id = experiment.experiment_id
    print('Experiment ID:', experiment_id) 

    # define tfidf suggestions
    tfidf_suggestions = {
        'max_df': lambda trial: trial.suggest_float('max_df', 0.5, 1.0),
        'min_df': lambda trial: trial.suggest_float('min_df', 0.01, 0.05),
        'ngram_range': lambda trial: trial.suggest_categorical('ngram_range', [(1, 1), (1, 2), (2, 2)]),
        'max_features': lambda trial: trial.suggest_int('max_features', 1000, 7000, step = 250)
    }

    # tune Logistic Regression 
    model_suggestions = {
        'solver': lambda trial: trial.suggest_categorical('solver', ['saga']),
        'penalty': lambda trial: trial.suggest_categorical('penalty', ['l1', 'l2']),
        'C': lambda trial: trial.suggest_float('C', 1e-4, 1e3, log = True),
        'max_iter': lambda trial: trial.suggest_int('max_iter', 300, 300, step = 1),
        'random_state': lambda trial: trial.suggest_int('random_state', 100, 400, step = 1)
    }

    run_id = log_mlflow(tfidf_suggestions, model_suggestions, 'lr', X_train, y_train, X_val, y_val, X_test, y_test, 25, 'lr', experiment_id = experiment_id)

    # tune xgboost 
    model_suggestions = {
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12, step = 1),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 1e-5, 0.1, log = True),
        'subsample': lambda trial: trial.suggest_float('subsample', 0.5, 1),
        'alpha': lambda trial: trial.suggest_float('alpha', 0, 10),
        'lambda': lambda trial: trial.suggest_float('lambda', 0, 10),
        'gamma': lambda trial: trial.suggest_float('gamma', 0, 10),
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 500, step = 1),
        'random_state': lambda trial: trial.suggest_int('random_state', 100, 400, step = 1)
    }

    run_id = log_mlflow(tfidf_suggestions, model_suggestions, 'xgb', X_train, y_train, X_val, y_val, X_test, y_test, 25, 'xgboost', experiment_id = experiment_id)

    # tune lightgbm 
    model_suggestions = {
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12, step = 1),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 1e-5, 0.1, log = True),
        'feature_fraction': lambda trial: trial.suggest_float('feature_fraction', 0.5, 1),
        'bagging_fraction': lambda trial: trial.suggest_float('bagging_fraction', 0.5, 1),
        'lambda_l1': lambda trial: trial.suggest_float('lambda_l1', 0, 100),
        'lambda_l2': lambda trial: trial.suggest_float('lambda_l1', 0, 100),
        'boosting_type': lambda trial: trial.suggest_categorical('boosting_type', ['gbdt']),
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 500, step = 1),
        'random_state': lambda trial: trial.suggest_int('random_state', 100, 400, step = 1),
        'verbose': lambda trial: trial.suggest_int('verbose', -1, -1, step = 1)
    }

    run_id = log_mlflow(tfidf_suggestions, model_suggestions, 'lgbm', X_train, y_train, X_val, y_val, X_test, y_test, 25, 'lightgbm', experiment_id = experiment_id)

    # tune catboost
    model_suggestions = {
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 12, step = 1),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 1e-5, 0.1, log = True),
        'l2_leaf_reg': lambda trial: trial.suggest_float('l2_leaf_reg', 1, 100),
        'bagging_temperature': lambda trial: trial.suggest_float('bagging_temperature', 0, 1),
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 500, step = 1),
        'random_state': lambda trial: trial.suggest_int('random_state', 100, 400, step = 1),
        'verbose': lambda trial: trial.suggest_int('verbose', 0, 0, step = 1)
    }

    run_id = log_mlflow(tfidf_suggestions, model_suggestions, 'cat', X_train, y_train, X_val, y_val, X_test, y_test, 25, 'catboost', experiment_id = experiment_id)

    # load best CatBoost model and evaluate on test set 
    # search through mlflow runs and select the run with the best F1 score
    runs_df = mlflow.search_runs(
        experiment_ids = [experiment_id],
        filter_string = "tags.mlflow.runName = 'catboost'",
        order_by = ["metrics.val_f1 DESC"]
    )

    best_run_id = runs_df.iloc[0]["run_id"]

    # load the model using the best run id
    model_uri = f"runs:/{best_run_id}/catboost"
    model = mlflow.sklearn.load_model(model_uri)

    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, pred_proba, pred, set = 'test')

    print('Test Accuracy:', metrics['test_accuracy'])
    print('Test Precision:', metrics['test_precision'])
    print('Test Recall:', metrics['test_recall'])
    print('Test Specificity:', metrics['test_specificity'])
    print('Test F1:', metrics['test_f1'])
    print('Test AUROC:', metrics['test_auroc'])
    print('Test AUPRC:', metrics['test_auprc'])

    create_confusion_matrix(y_test, pred, path = f'./artifacts/cat/test_confusion_matrix.png', set = 'Test')
    plot_roc_curve(y_test, pred_proba, path = f'./artifacts/cat/test_roc_curve.png', set = 'Test')
    plot_pr_curve(y_test, pred_proba, path = f'./artifacts/cat/test_pr_curve.png', set = 'Test')