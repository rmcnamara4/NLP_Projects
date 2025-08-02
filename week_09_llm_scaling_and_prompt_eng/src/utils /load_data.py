from datasets import load_dataset

def load_data(cfg): 
    if cfg.config_name is not None: 
        dataset = load_dataset(cfg.dataset_name, cfg.config_name) 
    else: 
        dataset = load_dataset(cfg.dataset_name) 
    
    dataset = dataset[cfg.split]
    dataset = dataset.shuffle(seed = cfg.seed).select(range(cfg.length))

    return dataset