import yaml

def load_config(path):
    """
    Load configuration settings from a YAML file.

    This function reads a YAML file and parses it into a Python dictionary,
    which can be used to configure model parameters, training settings, file paths, etc.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)