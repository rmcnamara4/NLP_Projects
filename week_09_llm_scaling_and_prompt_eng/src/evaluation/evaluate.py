def calculate_accuracy(predictions, references): 
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
