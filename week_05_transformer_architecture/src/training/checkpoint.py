import torch
import logging

import os

def save_checkpoint(model, optimizer, epoch, best_val_loss, path, train_losses = None, val_losses = None):
    """
    Save model, optimizer, epoch, and best validation loss to a file.

    Args:
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer used during training.
        epoch (int): The current epoch number.
        best_val_loss (float): The best validation loss so far.
        path (str): Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok = True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses, 
        'val_losses': val_losses
    }
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device = 'cuda'):
    """
    Load model, optimizer, epoch, and best validation loss from a checkpoint.

    Args:
        model (nn.Module): The model to load weights into.
        optimizer (Optimizer): The optimizer to load state into.
        path (str): Path to the checkpoint file.
        device (str): Device to map the checkpoint to ('cuda' or 'cpu').

    Returns:
        tuple: (epoch (int), best_val_loss (float), train_losses (list), val_losses (list))
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    logging.info(f"Loaded checkpoint from {path}")
    return epoch, best_val_loss, train_losses, val_losses