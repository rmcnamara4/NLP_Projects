import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def train_one_epoch(model, loader, criterion, optimizer, device, print_every = 200):
  """
  Trains the given model for one epoch.

  Args:
      model (torch.nn.Module): The PyTorch model to train.
      loader (DataLoader): DataLoader providing batches of (input_ids, labels, lengths).
      criterion (torch.nn.Module): Loss function (e.g., nn.BCEWithLogitsLoss()).
      optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
      device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
      print_every (int, optional): How often to print batch loss during training. Default is 200.

  Returns:
      tuple: 
          - float: Average training loss over all batches.
          - float: AUROC score on the training set.
          - float: AUPRC score on the training set.
  """
  model.train()
  total_loss = 0.0

  pred_proba = []
  true_labels = []
  for i, (ids, labels, lengths) in enumerate(loader):
    ids = ids.to(device)
    labels = labels.to(device)
    lengths = lengths.to(device)

    optimizer.zero_grad()
    outputs = model(ids, lengths)

    loss = criterion(outputs.squeeze(), labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    if (i + 1) % print_every == 0:
      print(f'Batch {i + 1} Average Loss: {loss.item():.4f}')

    pred_proba.append(outputs.squeeze())
    true_labels.append(labels)

  pred_proba = torch.cat(pred_proba, dim = 0).detach().cpu().numpy()
  true_labels = torch.cat(true_labels, dim = 0).detach().cpu().numpy()

  auroc = roc_auc_score(true_labels, pred_proba)
  auprc = average_precision_score(true_labels, pred_proba)

  return total_loss / len(loader), auroc.item(), auprc.item()

def evaluate_one_epoch(model, criterion, loader, device):
  """
  Evaluates the model on a given dataset.

  Args:
      model (torch.nn.Module): The trained PyTorch model to evaluate.
      loader (DataLoader): DataLoader providing batches of (input_ids, labels, lengths).

  Returns:
      tuple:
          - np.ndarray: Predicted probabilites for the positive class. 
          - np.ndarray: Ground truth binary labels. 
          - float: Average evaluation loss across all batches.
          - float: AUROC score on the evaluation dataset.
          - float: AUPRC score on the evaluation dataset.
  """
  model.eval()
  total_loss = 0.0

  pred_proba = []
  true_labels = []
  with torch.no_grad():
    for i, (ids, labels, lengths) in enumerate(loader):
      ids = ids.to(device)
      labels = labels.to(device)
      lengths = lengths.to(device)

      outputs = model(ids, lengths)
      loss = criterion(outputs.squeeze(), labels)

      total_loss += loss.item()

      pred_proba.append(outputs.squeeze())
      true_labels.append(labels)

    pred_proba = torch.cat(pred_proba, dim = 0).detach().cpu().numpy()
    true_labels = torch.cat(true_labels, dim = 0).detach().cpu().numpy()

    auroc = roc_auc_score(true_labels, pred_proba)
    auprc = average_precision_score(true_labels, pred_proba)

  return pred_proba, true_labels, total_loss / len(loader), auroc.item(), auprc.item()

def train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, device, epochs = 10, patience = 2, print_every = 200): 
    best_val_auc = 0.0
    counter = 0

    train_losses, train_aurocs, train_auprcs = [], [], []
    val_losses, val_aurocs, val_auprcs = [], [], []
    for i in range(epochs):
        print(f'Epoch {i + 1}')
        print('=================================')
        train_avg_loss, train_auroc, train_auprc = train_one_epoch(model, train_loader, criterion, optimizer, device, print_every)
        _, _, val_avg_loss, val_auroc, val_auprc = evaluate_one_epoch(model, criterion, val_loader, device)

        train_losses.append(train_avg_loss)
        train_aurocs.append(train_auroc)
        train_auprcs.append(train_auprc)

        val_losses.append(val_avg_loss)
        val_aurocs.append(val_auroc)
        val_auprcs.append(val_auprc)
        print('=================================')

        if val_auroc > best_val_auc:
            best_val_auc = val_auroc
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {i + 1}")
                break
        
        print(f'Train Loss: {train_avg_loss:.4f}, Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}')
        print(f'Val Loss: {val_avg_loss:.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:4f}')
        print()

    return best_model_state, train_losses, val_losses, train_aurocs, val_aurocs, train_auprcs, val_auprcs
