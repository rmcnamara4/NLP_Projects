from collections import Counter
from src.utils import generate_ngrams

def build_unigram_model(tokens):
  """
  Builds a unigram (1-gram) language model from a list of tokens.

  Args:
      tokens (List[str]): A list of word tokens.

  Returns:
      Dict[str, float]: A dictionary mapping each word to its probability
      (frequency normalized by the total number of tokens).
  """
  unigram_counts = Counter(tokens)
  denominator = len(tokens)
  return {
      word: unigram_counts[word] / denominator
      for word in unigram_counts
  }

def build_bigram_model(tokens):
  """
  Builds a bigram (2-gram) language model from a list of tokens.

  Args:
      tokens (List[str]): A list of word tokens.

  Returns:
      Dict[Tuple[str, str], float]: A dictionary mapping bigrams to their conditional
      probabilities P(w2 | w1), computed as count(w1, w2) / count(w1).
  """
  bigram_counts = Counter(generate_ngrams(tokens, 2))
  unigram_counts = Counter(tokens)
  return {
      (w1, w2): bigram_counts[(w1, w2)] / unigram_counts[w1]
      for (w1, w2) in bigram_counts
  }

def build_trigram_model(tokens):
  """
  Builds a trigram (3-gram) language model from a list of tokens.

  Args:
      tokens (List[str]): A list of word tokens.

  Returns:
      Dict[Tuple[str, str, str], float]: A dictionary mapping trigrams to their conditional
      probabilities P(w3 | w1, w2), computed as count(w1, w2, w3) / count(w1, w2).
  """
  trigram_counts = Counter(generate_ngrams(tokens, 3))
  bigram_counts = Counter(generate_ngrams(tokens, 2))
  return {
      (w1, w2, w3): trigram_counts[(w1, w2, w3)] / bigram_counts[(w1, w2)]
      for (w1, w2, w3) in trigram_counts
  }

def build_ngram_model(tokens, n):
  """
  Dispatches to the appropriate n-gram model builder based on the value of n.

  Args:
      tokens (List[str]): A list of word tokens.
      n (int): The n-gram size (1, 2, or 3).

  Returns:
      Dict: The n-gram model (unigram, bigram, or trigram).
  """
  if n == 1:
    return build_unigram_model(tokens)
  elif n == 2:
    return build_bigram_model(tokens)
  elif n == 3:
    return build_trigram_model(tokens)
  else:
    raise ValueError('Only n = 1, 2, or 3 are supported.')
  
def trigram_model_leplace(tokens):
  '''
  Constructs a smoothed trigram language model using Laplace (add-one) smoothing.

  Arguments:
    - tokens: A list of tokens from a preprocessed corpus

  Returns:
    - smoothed_trigram_prob: A function that takes (w1, w2, w3) and returns the Laplace-smoothed trigram probability
    - V: The vocabulary size (number of unique tokens)

  How it works:
    - Counts trigram and bigram frequencies from the token list
    - Applies Laplace smoothing to avoid zero probabilities for unseen trigrams:
        P(w3 | w1, w2) = (count(w1, w2, w3) + 1) / (count(w1, w2) + V)
    - This helps the model assign a small probability to unseen sequences

  Example:
    prob_fn, vocab_size = trigram_model_leplace(tokens)
    prob = prob_fn('the', 'cat', 'sat')
  '''
  trigrams = generate_ngrams(tokens, 3)
  bigrams = generate_ngrams(tokens, 2)

  trigram_counts = Counter(trigrams)
  bigram_counts = Counter(bigrams)

  V = len(set(tokens))

  def smoothed_trigram_prob(w1, w2, w3):
    trigram = (w1, w2, w3)
    bigram = (w1, w2)

    return (trigram_counts.get(trigram, 0) + 1) / (bigram_counts.get(bigram, 0) + V)

  return smoothed_trigram_prob, V