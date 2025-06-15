from torch.optim import Adam, SGD, RMSprop, AdamW

def get_optimizer(model, name, lr = 1e-3): 
    if name.lower() == 'adam': 
        return Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    elif name.lower() == 'sgd':
        return SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    elif name.lower() == 'rmsprop':
        return RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    elif name.lower() == 'adamw':
        return AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    else: 
        raise ValueError(f"Unsupported optimizer: {name}. Supported optimizers are: Adam, SGD, RMSprop, AdamW.")
    
