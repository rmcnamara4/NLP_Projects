import re

def get_word_shape(word):
  """
  Generate a simplified word shape pattern to capture orthographic features.

  The function encodes each character in the word using:
      - 'X' for uppercase letters
      - 'x' for lowercase letters
      - 'd' for digits
      - 's' for special characters or punctuation

  Consecutive identical symbols are collapsed to a single character to generalize the shape.

  Example:
      - "USA123!" → "Xds"
      - "Apple" → "Xx"
      - "hello123" → "xd"

  Args:
      word (str): The input word.

  Returns:
      str: The collapsed word shape string.
  """
  shape = ''
  for char in word:
    if char.isupper():
      shape += 'X'
    elif char.islower():
      shape += 'x'
    elif char.isdigit():
      shape += 'd'
    else:
      shape += 's'

  shape = re.sub(r'(.)\1+', r'\1', shape)
  return shape

def create_word_features(sentence, i):
  """
  Extracts a set of contextual and lexical features for a word at position `i` in a sentence.

  Args:
      sentence (List[Tuple[str, str]]): A list of (word, POS) tuples representing a sentence.
      i (int): Index of the target word in the sentence.

  Returns:
      dict: A dictionary of features for the CRF model, including:
          - Lexical features: word lowercased, suffix/prefix, casing, digit, shape
          - POS tag and first two characters of the tag
          - Features of previous and next word (if available)
          - Special flags for beginning (`BOS`) or end (`EOS`) of sentence
  """
  word, pos_tag = sentence[i][0], sentence[i][1]

  features = {
      'bias': 1.0,
      'word.lower()': word.lower(),
      'word[-3:]': word[-3:],
      'word[3:]': word[3:],
      'word.isupper()': word.isupper(),
      'word.islower()': word.islower(),
      'word.istitle()': word.istitle(),
      'word.isdigit()': word.isdigit(),
      'word.shape': get_word_shape(word),
      'pos_tag': pos_tag,
      'pos_tag[:2]': pos_tag[:2]
  }

  if i > 0:
    prev_word, prev_pos_tag = sentence[i - 1][0], sentence[i - 1][1]
    features.update({
        '-1:word.lower()': prev_word.lower(),
        '-1:pos_tag': prev_pos_tag,
        '-1:shape': get_word_shape(prev_word)
    })
  else:
    features['BOS'] = True

  if i < len(sentence) - 1:
    next_word, next_pos_tag = sentence[i + 1][0], sentence[i + 1][1]
    features.update({
        '+1:word.lower()': next_word.lower(),
        '+1:pos_tag': next_pos_tag,
        '+1:word.shape': get_word_shape(next_word)
    })
  else:
    features['EOS'] = True

  return features

def create_sentence_features(sentence):
  """
  Generates a list of feature dictionaries for each word in a sentence.

  Args:
      sentence (List[Tuple[str, str, str]]): A list of (word, POS, label) tuples.

  Returns:
      List[Dict[str, Any]]: A list of dictionaries, each containing features for a single word.
                            Features are generated using the `create_word_features` function.
  """
  return [create_word_features(sentence, i) for i in range(len(sentence))]

def get_labels(sentence):
  """
  Extracts the label sequence from a tagged sentence.

  Args:
      sentence (List[Tuple[str, str, str]]): A list of (word, POS, label) tuples.

  Returns:
      List[str]: A list of labels corresponding to each token in the sentence.
  """
  return [label for _, _, label in sentence]

