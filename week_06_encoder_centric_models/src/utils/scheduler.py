from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_scheduler(optimizer, mode = 'min', factor = 0.5, patience = 2):
    return ReduceLROnPlateau(
        optimizer, 
        mode = mode, 
        factor = factor, 
        patience = patience
    )