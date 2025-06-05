import torch
import logging

def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model and optimizer state to a file.

    Args:
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer used during training.
        epoch (int): The current epoch number.
        path (str): Destination file path.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """
    Load model and optimizer state from a checkpoint.

    Args:
        model (nn.Module): The model to load weights into.
        optimizer (Optimizer): The optimizer to restore state for.
        path (str): Checkpoint file path.
        device (str): 'cuda' or 'cpu'.

    Returns:
        int: The epoch at which the checkpoint was saved.
    """
    checkpoint = torch.load(path, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logging.info(f"Checkpoint loaded from {path}")
    return checkpoint.get('epoch', 0)