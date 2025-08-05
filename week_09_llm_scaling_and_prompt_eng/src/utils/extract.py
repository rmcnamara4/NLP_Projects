import re

def extract_final_answer(text):
    # Remove any artifacts like <<...>>
    text = re.sub(r'<<.*?>>', '', text)

    # Improved regex to catch full numbers like 1000, 105,000.00, etc.
    matches = re.findall(r'\b\d[\d,]*\.?\d*\b', text)

    if matches:
        cleaned = matches[-1].replace(',', '')
        try:
            return float(cleaned)
        except:
            return None
    
    return None