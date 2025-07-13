import json 
import os 

def save_results(results, save_dir): 
    """
    Saves evaluation metrics to a JSON file in the specified directory.

    Args:
        results (dict): A dictionary containing evaluation metrics (e.g., ROUGE scores).
        save_dir (str): The directory where the `metrics.json` file will be saved.
                        If the directory does not exist, it will be created.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok = True) 
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f: 
        json.dump(results, f, indent = 4) 

def save_predictions(predictions_dict, references_dict, save_dir): 
    """
    Saves model predictions and their corresponding references to a JSON file.

    Each entry in the output JSON is keyed by the article ID and contains:
        - 'prediction': The generated summary.
        - 'reference': The ground truth summary.

    Args:
        predictions_dict (dict): Dictionary mapping article IDs to predicted summaries.
        references_dict (dict): Dictionary mapping article IDs to reference summaries.
        save_dir (str): Directory where the output file (`predictions.json`) will be saved.
                        The directory is created if it does not exist.

    Returns:
        None
    """
    combined = {}
    for aid in predictions_dict: 
        combined[aid] = {
            'prediction': predictions_dict[aid], 
            'reference': references_dict.get(aid, '') 
        }

    os.makedirs(save_dir, exist_ok = True)
    with open(os.path.join(save_dir, 'predictions.json'), 'w') as f:
        json.dump(combined, f, indent = 2)