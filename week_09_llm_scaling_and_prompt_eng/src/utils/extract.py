import re

def extract_final_answer(text):
    """
    Extracts the final numeric answer from an LLM-generated response to a math problem.

    The function removes artifacts (e.g., <<...>>) and uses regex to find the last
    numeric value in the response. It supports integers and floats with commas.

    Args:
        text (str): The full generated text output from the language model.

    Returns:
        float or None: The extracted numeric answer as a float, or None if parsing fails.
    """
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