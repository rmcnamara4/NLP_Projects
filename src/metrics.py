import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

def calculate_metrics(y_true, y_pred_proba, y_pred, set = 'train'):
  """
  Calculates common classification evaluation metrics.

  Args:
      y_true (array-like): Ground truth binary labels (0 or 1).
      y_pred_proba (array-like): Predicted probabilities for the positive class.
      y_pred (array-like): Predicted binary class labels.
      set (str, optional): Identifier for the dataset split (e.g., 'train', 'val', 'test').
                            Used to prefix the returned metric keys. Default is 'train'.

  Returns:
      dict: A dictionary containing the following metrics with keys prefixed by `set`:
          - accuracy: Proportion of correct predictions.
          - precision: Proportion of positive predictions that are correct.
          - recall: Proportion of actual positives correctly predicted.
          - specificity: Proportion of actual negatives correctly predicted.
          - f1: Harmonic mean of precision and recall.
          - auroc: Area under the ROC curve.
          - auprc: Area under the Precision-Recall curve.
  """
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  specificity = recall_score(y_true, y_pred, pos_label = 0)
  f1 = f1_score(y_true, y_pred)
  auroc = roc_auc_score(y_true, y_pred_proba)
  auprc = average_precision_score(y_true, y_pred_proba)

  metrics = {
      f'{set}_accuracy': accuracy,
      f'{set}_precision': precision,
      f'{set}_recall': recall,
      f'{set}_specificity': specificity,
      f'{set}_f1': f1,
      f'{set}_auroc': auroc,
      f'{set}_auprc': auprc
  }

  return metrics

def plot_roc_curve(y_true, y_pred_proba, path = None, set = 'Train'):
    """
    Plots the ROC curve and computes the AUC.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred_proba (array-like): Predicted probabilities for the positive class.
        title (str): Title of the plot.

    Returns: None
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize = (6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle = '--', color = 'gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{set} ROC Curve')
    plt.legend(loc = 'lower right')
    plt.tight_layout()

    if path is None: 
       plt.show()
    else: 
        plt.savefig(path)
        plt.close()

def plot_pr_curve(y_true, y_pred_proba, path = None, set = 'Train'):
    """
    Plots the Precision-Recall curve and computes the average precision.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred_proba (array-like): Predicted probabilities for the positive class.
        title (str): Title of the plot.

    Returns: None
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize = (6, 5))
    plt.plot(recall, precision, label = f'AUPRC = {ap_score:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{set} Recall-Precision Curve')
    plt.legend(loc = 'lower left')
    plt.tight_layout()
    
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

def create_confusion_matrix(y_true, y_pred, path = None, set = 'Train'):
  """
  Generates and saves a confusion matrix plot for classification predictions.

  Args:
      y_true (array-like): Ground truth binary or multiclass labels.
      y_pred (array-like): Predicted class labels.
      path (str): File path to save the confusion matrix plot.
      set (str, optional): Label for the dataset split (e.g., 'Train', 'Val', 'Test').
                            Used in the plot title. Default is 'Train'.

  Returns:
      None. Saves the confusion matrix plot to the specified path.
  """
  ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
  plt.title(f'{set} Confusion Matrix')

  if path is None: 
     plt.show()
  else: 
     plt.savefig(path) 
     plt.close()

def plot_loss(train_losses, val_losses, path = None, title = 'Loss over Epochs'):
    """
    Plots training and validation loss over epochs.

    Args:
        train_losses (list of float): Training loss for each epoch.
        val_losses (list of float): Validation loss for each epoch.
        title (str, optional): Title of the plot. Defaults to 'Loss over Epochs'.
        save_path (str, optional): If provided, saves the plot to this path. Otherwise shows the plot.
    """
    plt.figure(figsize = (8, 6))
    plt.plot(train_losses, label = 'Train Loss', marker = 'o')
    plt.plot(val_losses, label = 'Val Loss', marker = 'o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.xticks(ticks = range(0, len(train_losses)), labels = range(1, len(train_losses) + 1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if path is None:
        plt.show()
    else: 
        plt.savefig(path)
        plt.close()

