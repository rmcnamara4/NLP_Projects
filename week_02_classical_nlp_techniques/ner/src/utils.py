from sklearn_crfsuite.metrics import flat_f1_score

def cast_to_py(obj):
    if isinstance(obj, dict):
        return {k: cast_to_py(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [cast_to_py(i) for i in obj]
    elif hasattr(obj, "item"):  # NumPy scalar
        return obj.item()
    else:
        return obj
    
def custom_score(estimator, X, y):
    y_pred = estimator.predict(X)
    return flat_f1_score(y, y_pred, average = 'weighted')