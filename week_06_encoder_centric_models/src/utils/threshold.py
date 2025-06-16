from sklearn.metrics import fbeta_score
import json

def find_best_threshold(probs, labels, step = 0.01, beta = 1): 
    best_threshold = 0.5
    best_score = -1

    for thresh in np.arange(0, 1 + step, step): 
        preds = (probs >= thresh).astype(int) 
        score = fbeta_score(labels, preds, beta = beta) 
        if score > best_score: 
            best_score = score
            best_threshold = thresh
    return best_threshold, best_score 

def save_threshold(threshold, path): 
    with open(path, 'w') as f: 
        json.dump({'threshold': threshold}, f) 

def load_threshold(path):
    with open(path, 'r') as f: 
        data = json.load(f)
    return data['threshold']