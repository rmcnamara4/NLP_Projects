def calculate_accuracy(predictions, references): 
    """
    Calculates the accuracy between predicted and reference values with a tolerance.

    A prediction is considered correct if both the prediction and reference are not None,
    and their absolute difference is less than 1e-3.

    Args:
        predictions (list of float or None): List of predicted values.
        references (list of float or None): List of reference (ground truth) values.

    Returns:
        tuple:
            - accuracy (float): The proportion of correct predictions.
            - correct (list of int): List indicating correctness per prediction (1 = correct, 0 = incorrect).
    """
    correct = []
    for pred, ref in zip(predictions, references): 
        if pred is not None and ref is not None: 
            if abs(pred - ref) < 1e-3: 
                correct.append(1) 
            else: 
                correct.append(0) 
        else: 
            correct.append(0)

    accuracy = sum(correct) / len(correct) 

    return accuracy, correct
