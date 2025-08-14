import mlflow 
from mlflow.tracking import MlflowClient

import optuna 
import json

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from metrics import *
import ast 


def create_model(model_name, model_params):
  """
  Creates and returns a classification model based on the specified model name and hyperparameters.

  Args:
      model_name (str): The name of the model to create. One of:
          - 'lr'   : Logistic Regression
          - 'xgb'  : XGBoost Classifier
          - 'lgbm' : LightGBM Classifier
          - 'cat'  : CatBoost Classifier
      model_params (dict): Dictionary of hyperparameters to initialize the model with.

  Returns:
      model (sklearn/base.BaseEstimator): An instance of the specified classification model,
                                          initialized with the provided parameters.

  Notes:
      - Adds default settings for `n_jobs` or `thread_count` where applicable for parallelism.
      - Evaluation metric is preset to AUC for tree-based models.
  """
  if model_name == 'lr':
    model = LogisticRegression(**model_params, n_jobs = 6)
  elif model_name == 'xgb':
    model = XGBClassifier(**model_params, eval_metric = 'auc', n_jobs = 6)
  elif model_name == 'lgbm':
    model = LGBMClassifier(**model_params, metric = 'auc', n_jobs = 6)
  elif model_name == 'cat':
    model = CatBoostClassifier(**model_params, eval_metric = 'AUC', thread_count = 6)
  return model

def create_objective(tfidf_suggestions, model_suggestions, model_name, X_train, y_train, X_val, y_val, experiment_id):
  """
  Creates an Optuna objective function for hyperparameter optimization using a TF-IDF + ML model pipeline.

  The returned objective function builds a pipeline with TF-IDF and a classifier, fits it on training data,
  evaluates it on validation data, logs the run to MLflow, and returns the validation F1 score.

  Args:
      tfidf_suggestions (Dict[str, Callable[[optuna.Trial], Any]]):
          Dictionary of hyperparameter suggestion functions for TF-IDF vectorizer.
      model_suggestions (Dict[str, Callable[[optuna.Trial], Any]]):
          Dictionary of hyperparameter suggestion functions for the classifier.
      model_name (str):
          Name of the model to use in the pipeline ('lr', 'xgb', 'lgbm', 'cat').
      X_train (pd.Series or array-like):
          Training feature data (text).
      y_train (pd.Series or array-like):
          Training labels.
      X_val (pd.Series or array-like):
          Validation feature data (text).
      y_val (pd.Series or array-like):
          Validation labels.
      experiment_id (str):
          MLflow experiment ID to log the nested runs under.

  Returns:
      Callable[[optuna.Trial], float]:
          An objective function compatible with Optuna that returns validation F1 score.
  """
  def objective(trial):
    tfidf_params = {key: func(trial) for key, func in tfidf_suggestions.items()}
    model_params = {key: func(trial) for key, func in model_suggestions.items()}

    tfidf = TfidfVectorizer(**tfidf_params, lowercase = False, tokenizer = str.split)
    model = create_model(model_name, model_params)

    pipe = Pipeline([
        ('tfidf', tfidf),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)

    y_pred_proba = pipe.predict_proba(X_val)[:, 1]
    y_pred = pipe.predict(X_val)

    metrics = calculate_metrics(y_val, y_pred_proba, y_pred, set = 'val')

    run_name = f'trial_{trial.number}'
    with mlflow.start_run(run_name = run_name, experiment_id = experiment_id, nested = True) as run:
      mlflow.log_param('tfidf_params', tfidf_params)
      mlflow.log_param('model_params', model_params)
      mlflow.log_metrics(metrics)
      trial.set_user_attr('mlflow_run_id', run.info.run_id)

    return metrics['val_f1']
  return objective

def find_model_artifact(run_id): 
    """
    Finds the relative artifact path of a logged MLflow model for a given run ID.

    The returned artifact path can be used in `mlflow.<flavor>.load_model` calls
    to load the model without needing to hardcode the artifact subdirectory name
    (e.g., "model", "catboost", "sklearn-model").

    Args:
        run_id (str):
            The unique MLflow run ID from which to locate the logged model artifact.

    Returns:
        str:
            The relative path to the model's artifact directory (e.g., "model",
            "catboost", "xgboost-model"), which contains the 'MLmodel' file.

    Raises:
        FileNotFoundError:
            If no artifact directory containing an 'MLmodel' file is found for the given run ID.

    Notes:
        This is useful when:
        - You have retrieved the best run via `mlflow.search_runs` but do not know
          the exact artifact subdirectory name for the model.
        - The function is flavor-agnostic and works for any MLflow-supported model type.
    """
    client = MlflowClient()
    # look at top-level artifact dirs
    for art in client.list_artifacts(run_id):
        if not art.is_dir:
            continue
        # does this folder have an MLmodel file?
        children = client.list_artifacts(run_id, art.path)
        if any(c.path.endswith('MLmodel') for c in children):
            return art.path
    raise FileNotFoundError(f'No MLflow model artifact found for run {run_id}')

def log_mlflow(tfidf_suggestions, model_suggestions, model_name, X_train, y_train, X_val, y_val, X_test, y_test, n_trials, run_name, experiment_id):
  """
  Runs hyperparameter tuning using Optuna, evaluates the best model, and logs all results to MLflow.

  This function:
    - Defines and optimizes an Optuna objective using a TF-IDF + classifier pipeline.
    - Logs the best parameters and validation metrics.
    - Re-trains the model on the full training set using the best parameters.
    - Evaluates on train, validation, and test sets.
    - Logs metrics, ROC & PR curves, confusion matrices, and the final model to MLflow.

  Args:
      tfidf_suggestions (Dict[str, Callable[[optuna.Trial], Any]]):
          Dictionary of TF-IDF hyperparameter search spaces for Optuna.
      model_suggestions (Dict[str, Callable[[optuna.Trial], Any]]):
          Dictionary of model hyperparameter search spaces for Optuna.
      model_name (str):
          One of 'lr', 'xgb', 'lgbm', or 'cat' — specifies the classifier to use.
      X_train, y_train:
          Training data and labels.
      X_val, y_val:
          Validation data and labels.
      X_test, y_test:
          Test data and labels.
      n_trials (int):
          Number of Optuna trials to run.
      run_name (str):
          Name of the parent MLflow run.
      experiment_id (str):
          ID of the MLflow experiment where results should be logged.

  Returns:
      str: The MLflow run ID of the parent run.

  Notes:
      - Uses nested MLflow runs for each trial during tuning.
      - Uses stratified performance metrics (AUROC, AUPRC, F1, etc.).
      - Saves visualizations and logs them as MLflow artifacts.
      - Logs the final trained model with input-output signature.
  """
  objective = create_objective(tfidf_suggestions, model_suggestions, model_name, X_train, y_train, X_val, y_val, experiment_id)
  direction = 'maximize'
  study = optuna.create_study(direction = direction)

  with mlflow.start_run(run_name = run_name, experiment_id = experiment_id) as run_outer:
    study.optimize(objective, n_trials = n_trials)

    mlflow.log_metric('best_optimize_metric', study.best_value)
    mlflow.set_tag('best_trial_number', study.best_trial.number)

    best_run_id = study.best_trial.user_attrs['mlflow_run_id']
    run = mlflow.get_run(best_run_id)

    best_tfidf_params = ast.literal_eval(run.data.params['tfidf_params'])
    best_model_params = ast.literal_eval(run.data.params['model_params'])
    mlflow.log_param('best_tfidf_params', best_tfidf_params)
    mlflow.log_param('best_model_params', best_model_params)

    tfidf = TfidfVectorizer(**best_tfidf_params, lowercase = False, tokenizer = str.split)
    model = create_model(model_name, best_model_params)

    pipe = Pipeline([
        ('tfidf', tfidf),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)

    train_pred_proba = pipe.predict_proba(X_train)[:, 1]
    train_pred = pipe.predict(X_train)

    os.makedirs(f'./artifacts/{model_name}', exist_ok = True)

    train_metrics = calculate_metrics(y_train, train_pred_proba, train_pred, set = 'Train')
    with open(f'./artifacts/{model_name}/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent = 2)
    plot_roc_curve(y_train, train_pred_proba, path = f'./artifacts/{model_name}/train_roc_curve.png', set = 'Train')
    plot_pr_curve(y_train, train_pred_proba, path = f'./artifacts/{model_name}/train_pr_curve.png', set = 'Train')
    create_confusion_matrix(y_train, train_pred, path = f'./artifacts/{model_name}/train_confusion_matrix.png', set = 'Train')

    mlflow.log_metrics(train_metrics)
    mlflow.log_artifact(f'./artifacts/{model_name}/train_roc_curve.png')
    mlflow.log_artifact(f'./artifacts/{model_name}/train_pr_curve.png')
    mlflow.log_artifact(f'./artifacts/{model_name}/train_confusion_matrix.png')

    val_pred_proba = pipe.predict_proba(X_val)[:, 1]
    val_pred = pipe.predict(X_val)

    val_metrics = calculate_metrics(y_val, val_pred_proba, val_pred, set = 'Val')
    with open(f'./artifacts/{model_name}/val_metrics.json', 'w') as f:
        json.dump(val_metrics, f, indent = 2)
    plot_roc_curve(y_val, val_pred_proba, path = f'./artifacts/{model_name}/val_roc_curve.png', set = 'Val')
    plot_pr_curve(y_val, val_pred_proba, path = f'./artifacts/{model_name}/val_pr_curve.png', set = 'Val')
    create_confusion_matrix(y_val, val_pred, path = f'./artifacts/{model_name}/val_confusion_matrix.png', set = 'Val')

    mlflow.log_metrics(val_metrics)
    mlflow.log_artifact(f'./artifacts/{model_name}/val_roc_curve.png')
    mlflow.log_artifact(f'./artifacts/{model_name}/val_pr_curve.png')
    mlflow.log_artifact(f'./artifacts/{model_name}/val_confusion_matrix.png')

    test_pred_proba = pipe.predict_proba(X_test)[:, 1]
    test_pred = pipe.predict(X_test)

    test_metrics = calculate_metrics(y_test, test_pred_proba, test_pred, set = 'Test')
    with open(f'./artifacts/{model_name}/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent = 2)
    plot_roc_curve(y_test, test_pred_proba, path = f'./artifacts/{model_name}/test_roc_curve.png', set = 'Test')
    plot_pr_curve(y_test, test_pred_proba, path = f'./artifacts/{model_name}/test_pr_curve.png', set = 'Test')
    create_confusion_matrix(y_test, test_pred, path = f'./artifacts/{model_name}/test_confusion_matrix.png', set = 'Test')

    mlflow.log_metrics(test_metrics)
    mlflow.log_artifact(f'./artifacts/{model_name}/test_roc_curve.png')
    mlflow.log_artifact(f'./artifacts/{model_name}/test_pr_curve.png')
    mlflow.log_artifact(f'./artifacts/{model_name}/test_confusion_matrix.png')

    signature = mlflow.models.infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(pipe, run_name, signature = signature)

  return run_outer.info.run_id