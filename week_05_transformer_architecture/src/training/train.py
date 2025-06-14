import torch 
import logging 
import os 

from src.training.checkpoint import save_checkpoint, load_checkpoint

def train_one_epoch(model, dataloader, optimizer, criterion, device, print_every = 50):
  """
  Trains the model for one epoch over the provided DataLoader.

  This function:
    - Sets the model to training mode.
    - Loops through batches, performs forward pass, computes loss, and updates model parameters.
    - Applies gradient clipping to prevent exploding gradients.
    - Optionally prints loss at regular intervals.

  Args:
      model (nn.Module): The Transformer model being trained.
      dataloader (DataLoader): PyTorch DataLoader providing batches of (src_input, tgt_input, tgt_output).
      optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
      criterion (nn.Module): Loss function used for training (e.g., nn.CrossEntropyLoss).
      device (torch.device): Device to run training on ('cpu' or 'cuda').
      print_every (int, optional): Interval (in steps) to print batch loss. Defaults to 50.

  Returns:
      float: Average training loss over the epoch.
  """
  model.train()

  total_loss = 0
  for i, (src_input, tgt_input, tgt_output) in enumerate(dataloader):
    src_input = src_input.to(device)
    tgt_input = tgt_input.to(device)
    tgt_output = tgt_output.to(device)

    optimizer.zero_grad()

    output, self_attn, cross_attn = model(src_input, tgt_input)

    output = output.view(-1, output.shape[-1])
    tgt_output = tgt_output.reshape(-1).long()

    loss = criterion(output, tgt_output)
    total_loss += loss.item()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
    optimizer.step()

    if i % print_every == 0:
      logging.info(f'Batch {i} Loss: {loss:.4f}')

    torch.cuda.empty_cache()

  return total_loss / len(dataloader)

def evaluate_one_epoch(model, dataloader, criterion, device):
  """
  Evaluates the model for one epoch over the provided validation/test DataLoader.

  This function:
    - Sets the model to evaluation mode.
    - Disables gradient calculation for efficiency.
    - Computes the loss over the entire validation or test set.

  Args:
      model (nn.Module): The Transformer model being evaluated.
      dataloader (DataLoader): PyTorch DataLoader providing batches of (src_input, tgt_input, tgt_output).
      criterion (nn.Module): Loss function used for evaluation (e.g., nn.CrossEntropyLoss).
      device (torch.device): Device to run evaluation on ('cpu' or 'cuda').

  Returns:
      float: Average loss across the evaluation dataset.
  """
  model.eval()

  total_loss = 0.0
  with torch.no_grad():
    for i, (src_input, tgt_input, tgt_output) in enumerate(dataloader):
      src_input, tgt_input, tgt_output = src_input.to(device), tgt_input.to(device), tgt_output.to(device)

      output, self_attn, cross_attn = model(src_input, tgt_input)

      output = output.view(-1, output.shape[-1])
      tgt_output = tgt_output.reshape(-1).long()

      loss = criterion(output, tgt_output)
      total_loss += loss.item()

      torch.cuda.empty_cache()

  return total_loss / len(dataloader)

def train(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs, print_every = 50, patience = 2, checkpoint_path = None, resume = False):
  """
  Trains the Transformer model using a training and validation loop with early stopping.

  This function:
    - Trains the model for a specified number of epochs.
    - Evaluates the model on a validation set after each epoch.
    - Tracks training and validation loss.
    - Applies early stopping if validation loss does not improve for a given number of epochs.
    - Restores the model to the best-performing state.

  Args:
      model (nn.Module): The Transformer model to train.
      train_dataloader (DataLoader): DataLoader for the training set.
      val_dataloader (DataLoader): DataLoader for the validation set.
      optimizer (torch.optim.Optimizer): Optimizer used for training.
      criterion (nn.Module): Loss function for training and validation (e.g., nn.CrossEntropyLoss).
      device (torch.device): Device to train the model on ('cpu' or 'cuda').
      epochs (int): Maximum number of training epochs.
      print_every (int, optional): How frequently to print training loss during an epoch. Default is 50.
      patience (int, optional): Number of consecutive epochs with no validation improvement before early stopping. Default is 2.
      checkpoint_path (str, optional): File path to save or resume checkpoint.
      resume (bool): Whether to resume from checkpoint.

  Returns:
      tuple: A tuple containing:
          - train_losses (list of float): Training loss for each epoch.
          - val_losses (list of float): Validation loss for each epoch.
  """
  train_losses = []
  val_losses = []

  best_val_loss = float('inf')
  no_improve_epochs = 0
  start_epoch = 0 

  if resume and checkpoint_path and os.path.exists(checkpoint_path): 
    start_epoch, best_val_loss, train_losses, val_losses = load_checkpoint(model, optimizer, checkpoint_path, device)
    logging.info(f"Resuming training from epoch {start_epoch + 1} with best_val_loss = {best_val_loss:.4f}")

  best_model_state = model.state_dict()

  for epoch in range(start_epoch, epochs):
    logging.info(f'Epoch {epoch + 1} / {epochs}')
    logging.info('-' * 30)

    train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device, print_every = print_every)
    val_loss = evaluate_one_epoch(model, val_dataloader, criterion, device)

    logging.info(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      no_improve_epochs = 0
      best_model_state = model.state_dict()

      if checkpoint_path: 
        save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path, train_losses = train_losses, val_losses = val_losses)

    else:
      no_improve_epochs += 1
      logging.info(f'No improvement in validation loss for {no_improve_epochs} epochs.')

      if no_improve_epochs >= patience:
        logging.info(f'Early stopping triggered after {epoch + 1} epochs.')
        break

  model.load_state_dict(best_model_state)

  return train_losses, val_losses