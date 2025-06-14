def set_seed(seed: int = 42):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    This function controls the sources of randomness that affect model initialization,
    data shuffling, dropout behavior, and other stochastic processes in machine learning
    workflows. It also configures PyTorch to use deterministic operations where possible.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False