import re

def extract_final_answer(text): 
    text = re.sub(r'<<.*>?>>', '', text) 
    numbers = re.findall(r'\d+', text) 
    
    if numbers: 
        return int(numbers[-1])
    return None