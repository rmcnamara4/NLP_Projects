import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from metrics import * 

def save_plots(y_true, y_pred_proba, y_pred, path = None, set = 'Train'): 
    metrics = calculate_metrics(y_true, y_pred_proba, y_pred, set = set) 
    plot_roc_curve(y_true, y_pred_proba, path = path, set = set) 
    plot_pr_curve(y_true, y_pred_proba, path = path, set = set)
    create_confusion_matrix(y_true, y_pred, path = path, set = set)
