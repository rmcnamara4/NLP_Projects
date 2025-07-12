import json 
import os 

def save_results(results, save_dir): 
    os.makedirs(save_dir, exist_ok = True) 
    with open(save_dir + 'metrics.json', 'w') as f: 
        json.dump(results, f, indent = 4) 

def save_predictions(predictions_dict, references_dict, save_dir): 
    combined = {}
    for aid in predictions_dict: 
        combined[aid] = {
            'prediction': predictions_dict[aid], 
            'reference': references_dict.get(aid, '') 
        }

    os.makedirs(save_dir, exist_ok = True)
    with open(save_dir + 'predictions.json', 'w') as f:
        json.dump(combined, f, indent = 2)