import numpy as np 
import torch 

def get_class_weights(labels, strategy = 'balanced'): 
    if strategy.lower() == 'None': 
        return None 
    
    elif strategy == 'balanced': 
        labels = np.array(labels) 
        class_counts = np.bincount(labels) 

        if np.any(class_counts == 0): 
            raise ValueError(f'One or more classes have 0 samples: {class_counts}')
        class_weights = 1.0 / class_counts 
        class_weights = class_weights / np.sum(class_weights)

        return torch.tensor(class_weights, dtype = torch.float32)
    
    elif isinstance(strategy, (list, tuple)) and len(strategy) == 2: 
        return torch.tensor(strategy, dtype = torch.float32) 
    
    else: 
        raise ValueError(f'Unsupported class weight strategy: {strategy}')
