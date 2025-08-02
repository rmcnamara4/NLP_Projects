import pandas as pd 
from datetime import datetime
import os

def save_results(model_name, prompt_version, questions, answers, generated_text, predictions, correct_vec, output_file): 
    result = pd.DataFrame({
        'example_id': [f'{i:04d}' for i in range(len(questions))], 
        'question': questions, 
        'expected_answer': answers, 
        'generated_text': generated_text,
        'model_prediction': predictions, 
        'is_correct': correct_vec, 
        'model_name': [model_name] * len(questions), 
        'prompt_version': [prompt_version] * len(questions), 
        'timestamp': [datetime.now().isoformat()] * len(questions)
    })

    os.makedirs(os.path.dirname(output_file), exist_ok = True) 
    result.to_csv(output_file, header = True, index = False) 

    