import torch 
import os

def save_model_history(model, train_losses, val_losses, train_auprcs, val_auprcs, config): 
    """
    Save the model and its training history to the specified paths in the config.
    
    Args:
        model (torch.nn.Module): The trained model.
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_auprcs (list): List of training AUPRCs.
        val_auprcs (list): List of validation AUPRCs.
        config (dict): Configuration dictionary containing paths for saving.
    """
    os.makedirs(config['paths']['model_dir'], exist_ok = True)
    
    torch.save(model.state_dict(), config['paths']['model_file'])
    torch.save(train_losses, config['paths']['train_loss_file'])
    torch.save(val_losses, config['paths']['val_loss_file'])
    torch.save(train_auprcs, config['paths']['train_auprc_file'])
    torch.save(val_auprcs, config['paths']['val_auprc_file'])
    
    logging.info('Model and training history saved successfully.')