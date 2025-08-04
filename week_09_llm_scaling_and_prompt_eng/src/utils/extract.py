import re

def extract_final_answer(text):
    # Remove any artifacts like <<...>>
    text = re.sub(r'<<.*?>>', '', text)

    # Match numbers with optional commas and decimals (e.g., 37,500.00 or 25)
    matches = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+', text)

    if matches:
        # Clean commas and convert last match to float
        cleaned = matches[-1].replace(',', '')
        return float(cleaned)
    
    return None