import re 

import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from word2number import w2n
from nltk.tokenize import word_tokenize

import spacy 
nlp = spacy.load('en_core_web_sm', disable = ['ner', 'parser'])

def replace_numbers(text):
  """
  Replaces all numeric digits and written numbers in a text string with the <NUM> token.

  Args:
      text (str): A string of text to process.

  Returns:
      str: The input text with standalone digits and number words replaced by <NUM>.

  Notes:
      - Uses regex to detect digit-based numbers (e.g., "123").
      - Uses `word2number` to identify written numbers (e.g., "twenty").
      - Any word that fails both checks remains unchanged.
  """
  words = text.split()
  converted_words = []
  for word in words:
    if re.match(r'\b\d+\b', word):
      converted_words.append('<NUM>')
    else:
      try:
        _ = w2n.word_to_num(word)
        converted_words.append('<NUM>')
      except:
        converted_words.append(word)
  return ' '.join(converted_words)


def replace_numbers_batch(texts):
    """
    Replaces all numeric digits and written numbers with the <NUM> token in a batch of text strings.

    Args:
        texts (List[str]): A list of text strings to process.

    Returns:
        List[str]: A list of processed text strings with numbers replaced by <NUM>.

    Notes:
        - Applies lowercase transformation before processing.
        - Matches digit-based numbers using regex.
        - Converts written numbers (e.g., "five", "twenty") using `word2number`.
        - If conversion fails, retains the original word.
        - Handles unexpected cases by printing the problematic word (for debugging).
    """
    processed_texts = []

    for text in texts:
        words = text.lower().split()
        converted_words = []
        for word in words:
            if re.match(r"\b\d+\b", word):  # Match numeric digits
                converted_words.append("<NUM>")
            else:
                try:
                    _ = w2n.word_to_num(word)  # Convert written number
                    converted_words.append("<NUM>")
                except ValueError:
                    converted_words.append(word)
                except IndexError:
                    print(f'Problematic word: {word}')
                    converted_words.append(word)
        processed_texts.append(" ".join(converted_words))

    return processed_texts

stop_words = set(stopwords.words('english'))

def preprocess_nltk(text):
  """
  Applies standard NLP preprocessing steps using NLTK for a single text string.

  Args:
      text (str): The raw input text to preprocess.

  Returns:
      str: The preprocessed text with normalized tokens.

  Steps:
      - Replaces all numeric digits and written numbers with the <NUM> token.
      - Converts text to lowercase for uniformity.
      - Removes punctuation and non-word characters.
      - Tokenizes the text into individual words using NLTK's tokenizer.
      - Removes stopwords from the token list.
      - Applies lemmatization to reduce words to their base form.
      - Returns the cleaned text as a single space-joined string.
  """
  lemmatizer = WordNetLemmatizer()
  text = replace_numbers(text)
  text = text.lower()
  text = re.sub(r"[^\w\s]", "", text)
  tokens = word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
  return ' '.join(tokens)

def preprocess_spacy(texts):
  """
  Applies NLP preprocessing to a batch of text documents using spaCy.

  Args:
      texts (List[str]): A list of raw text strings to preprocess.

  Returns:
      List[str]: A list of cleaned and lemmatized text strings.

  Steps:
      - Replaces numeric digits and written numbers with the <NUM> token.
      - Processes texts in parallel using spaCy's pipeline.
      - For each document:
          - Removes stopwords.
          - Keeps only alphabetic tokens (filters out punctuation/numbers).
          - Applies lemmatization to each remaining token.
      - Returns a list of preprocessed strings, one for each input text.

  Notes:
      - Uses spaCy's efficient `nlp.pipe()` for batching and parallel processing.
      - The `replace_numbers_batch` function should handle both digit and written numbers.
  """
  texts = replace_numbers_batch(texts)
  processed_texts = [
      ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha]) for doc in nlp.pipe(texts, batch_size = 1000, n_process = -1)
  ]
  return processed_texts