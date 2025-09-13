from datasets import load_dataset

def load_data(cfg): 
    """
    Loads and preprocesses a dataset using Hugging Face's `load_dataset` utility based on a configuration object.

    This function handles optional dataset configuration names, shuffles the dataset with a fixed seed,
    and selects a subset of examples for use.

    Args:
        cfg (object): Configuration object containing the following attributes:
            - dataset_name (str): Name of the dataset to load.
            - config_name (str or None): Optional configuration name (e.g., 'plain_text').
            - split (str): Dataset split to load (e.g., 'train', 'test', 'validation').
            - seed (int): Random seed used for shuffling.
            - length (int): Number of examples to select from the shuffled dataset.

    Returns:
        datasets.Dataset: A Hugging Face Dataset object with the specified split, shuffled and truncated.
    """
    if cfg.config_name is not None: 
        dataset = load_dataset(cfg.dataset_name, cfg.config_name) 
    else: 
        dataset = load_dataset(cfg.dataset_name) 
    
    dataset = dataset[cfg.split]
    dataset = dataset.shuffle(seed = cfg.seed).select(range(cfg.length))

    return dataset