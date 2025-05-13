def preprocess_ptb(series):
  tokenized_sentences = []
  for text in series:
    tokens = text[0].split()
    tokens = ['<s>'] + tokens + ['</s>']
    tokenized_sentences.append(tokens)
  return [token for sentence in tokenized_sentences for token in sentence]

def generate_ngrams(tokens, n):
  '''
  Generates n-grams from a list of tokens.

  Args:
    - tokens (List[str]): A list of string tokens.
    - n (int): The number of items in each n-gram (e.g., 2 for bigrams).

  Returns:
    - List[Tuple[str]]: A list of n-gram tuples.

  Example:
    >>> generate_ngrams(["the", "dog", "wagged", "his", "tail"], 2)
    [('the', 'dog'), ('dog', 'wagged'), ('wagged', 'his'), ('his', 'tail')]
  '''
  return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]