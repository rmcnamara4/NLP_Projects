import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from metrics import * 
import json

def save_plots(y_true, y_pred_proba, y_pred, path = None, set = 'Train'): 
    metrics = calculate_metrics(y_true, y_pred_proba, y_pred, set = set) 
    with open(os.path.join(path, f'{set.lower()}_metrics.json'), 'w') as f: 
        json.dump(metrics, f, indent = 4)
    plot_roc_curve(y_true, y_pred_proba, path = os.path.join(path, f'{set.lower()}_roc_curve.png'), set = set) 
    plot_pr_curve(y_true, y_pred_proba, path = os.path.join(path, f'{set.lower()}_auprc_curve.png'), set = set)
    create_confusion_matrix(y_true, y_pred, path = os.path.join(path, f'{set.lower()}_confusion_matrix.png'), set = set)
