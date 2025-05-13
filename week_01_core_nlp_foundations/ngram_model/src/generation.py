import random 
from src.ngram_models import build_unigram_model, build_bigram_model, build_trigram_model, build_ngram_model

def generate_unigram_text(model, length = 10):
  """
  Generates text from a unigram model by sampling words based on their probabilities.

  Args:
      model (Dict[str, float]): A unigram probability model.
      length (int): Number of words to generate. If None or 0, generation continues until </s> is seen.

  Returns:
      str: A generated sentence.
  """
  sentence = []
  if length:
    for _ in range(length):
      next_word = random.choices(list(model.keys()), weights = list(model.values()))[0]
      sentence.append(next_word)
  else:
    while True:
      next_word = random.choices(list(model.keys()), weights = list(model.values()))[0]
      sentence.append(next_word)
      if next_word == '</s>':
        break
  return ' '.join(sentence)

def generate_bigram_text(model, seed_word, length = 10):
  """
  Generates text from a bigram model using a single seed word.

  Args:
      model (Dict[Tuple[str, str], float]): A bigram probability model.
      seed_word (str): The starting word of the sentence.
      length (int): Total number of words to generate. If None or 0, continues until </s>.

  Returns:
      str: A generated sentence beginning with the seed word.
  """
  sentence = [seed_word]
  if length:
    for _ in range(length - 1):
      candidates = {k[1]:v for k, v in model.items() if k[0] == sentence[-1]}
      if not candidates:
        break
      next_word = random.choices(list(candidates.keys()), weights = list(candidates.values()))[0]
      sentence.append(next_word)
  else:
    while True:
      candidates = {k[1]:v for k, v in model.items() if k[0] == sentence[-1]}
      if not candidates:
        break
      next_word = random.choices(list(candidates.keys()), weights = list(candidates.values()))[0]
      sentence.append(next_word)
      if next_word == '</s>':
        break
  return ' '.join(sentence)

def generate_trigram_text(model, seed_words, length = 10):
  """
  Generates text from a trigram model using two seed words.

  Args:
      model (Dict[Tuple[str, str, str], float]): A trigram probability model.
      seed_words (Tuple[str, str]): The starting two words.
      length (int): Total number of words to generate. If None or 0, continues until </s>.

  Returns:
      str: A generated sentence beginning with the seed words.
  """
  sentence = list(seed_words)
  if length:
    for _ in range(length - 2):
      candidates = {k[2]:v for k, v in model.items() if k[:2] == tuple(sentence[-2:])}
      if not candidates:
        break
      next_word = random.choices(list(candidates.keys()), weights = list(candidates.values()))[0]
      sentence.append(next_word)
  else:
    while True:
      candidates = {k[2]:v for k, v in model.items() if k[:2] == tuple(sentence[-2:])}
      if not candidates:
        break
      next_word = random.choices(list(candidates.keys()), weights = list(candidates.values()))[0]
      sentence.append(next_word)
      if next_word == '</s>':
        break
  return ' '.join(sentence)

def generate_ngram_text(model, seed_words, n, length = 10):
  """
  Dispatch function to generate text from an n-gram model.

  Args:
      model (Dict): An n-gram model (unigram, bigram, or trigram).
      seed_words (Union[str, Tuple[str, str]]): Seed word(s) used to start the sentence.
      n (int): The type of model (1, 2, or 3).
      length (int): Number of tokens to generate.

  Returns:
      None
  """
  if n == 1:
    print(generate_unigram_text(model, length))
  elif n == 2:
    print(generate_bigram_text(model, seed_words, length))
  elif n == 3:
    print(generate_trigram_text(model, seed_words, length))
  else:
    raise ValueError('Only n = 1, 2, or 3 are supported.')