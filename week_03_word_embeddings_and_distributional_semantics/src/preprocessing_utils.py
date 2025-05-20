import re 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))

custom_tokenizer = RegexpTokenizer(r'[a-zA-Z\d<NUM>./]+(?:-[a-zA-Z\d<NUM>./]+)*|<NUM>\.<NUM>/<NUM>\.<NUM>|<NUM>\.<NUM>/<NUM>|<NUM>/<NUM>\.<NUM>|<NUM>/<NUM>|<NUM>\.<NUM>|<NUM>|\w+|')

def replace_numbers(text):
  """
  Replaces standalone numbers and specific numeric patterns in a text string with placeholders.

  - Replaces decimal points between digits (e.g., "6.8") with '<DOT>' temporarily.
  - Replaces slashes between digits (e.g., "120/80") with '<SLASH>' temporarily.
  - Replaces all standalone whole numbers with the token '<NUM>'.
  - Restores the decimal points and slashes after substitution.

  Args:
      text (str): Input text to process.

  Returns:
      str: Text with numeric values standardized using the '<NUM>' placeholder.
    """
  text = re.sub(r'(?<=\d)\.(?=\d)', r'<DOT>', text)
  text = re.sub(r'(?<=\d)/(?=\d)', r'<SLASH>', text)
  text = re.sub(r'\b\d+\b', r'<NUM>', text)
  text = text.replace('<DOT>', '.').replace('<SLASH>', '/')
  return text

def clean_text(text, bpe = False):
  """
  Cleans and tokenizes biomedical or clinical text using custom preprocessing steps.

  Steps:
  1. Converts text to lowercase.
  2. Replaces numeric patterns using `replace_numbers()`.
  3. Tokenizes using a custom regular expression tokenizer designed to preserve biomedical formats.
  4. Removes stop words.
  5. Removes most punctuation (retains periods and slashes).
  6. Lemmatizes tokens unless they contain the '<NUM>' placeholder.

  Args:
      text (str): Raw input text to clean and tokenize.

  Returns:
      List[str]: A list of cleaned and lemmatized tokens.
  """
  text = text.lower()
  text = replace_numbers(text)
  tokens = custom_tokenizer.tokenize(text)
  tokens = [t for t in tokens if t not in stop_words]
  tokens = [t for t in tokens if t not in string.punctuation or t in ('.', '/')]
  if not bpe: 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) if '<NUM>' not in t else t for t in tokens]
  return tokens

