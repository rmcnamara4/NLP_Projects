import math 

def calculate_perplexity(model, tokens, n):
  '''
  Calculates the perplexity of an n-gram model on a given token sequence.

  Perplexity measures how well a probability model predicts a sequence.
  Lower perplexity indicates better predictive performance.

  Arguments:
    - model: A dictionary representing the n-gram model (e.g., unigram, bigram, trigram)
    - tokens: A list of tokens (words) from the text to evaluate
    - n: The order of the n-gram (1 for unigram, 2 for bigram, etc.)

  Returns:
    - perplexity: A float value representing the model's perplexity on the token sequence

  Notes:
    - If an n-gram is not found in the model, a small probability (1e-6) is used.
    - Uses base-2 logarithm for computing log probability.
  '''
  N = len(tokens)
  log_prob_sum = 0

  for t in range(len(tokens) - n + 1):
    if n == 1:
      ngram = tokens[t]
    else:
      ngram = tuple(tokens[t:t+n])
    prob = model.get(ngram, 1e-6)
    log_prob_sum += math.log2(prob)

  return 2 ** (-log_prob_sum / N)

def calculate_fallback_perplexity(trigram_model, bigram_model, unigram_model, tokens):
  '''
  Calculates the perplexity of a fallback n-gram model on a given token sequence.

  This function uses a backoff approach:
  - It first tries to find the trigram probability
  - If not found, it falls back to the bigram
  - If the bigram is also missing, it falls back to the unigram
  - If the unigram is missing, it uses a small default probability (1e-6)

  Arguments:
    - trigram_model: A dictionary of trigram probabilities
    - bigram_model: A dictionary of bigram probabilities
    - unigram_model: A dictionary of unigram probabilities
    - tokens: A list of tokens to evaluate

  Returns:
    - perplexity: A float representing how well the fallback model predicts the token sequence

  Notes:
    - Uses log base 2
    - The perplexity is normalized by the total number of tokens
  '''
  n = len(tokens)
  log_prob_sum = 0
  for i in range(2, n):
    trigram = (tokens[i - 2], tokens[i - 1], tokens[i])
    bigram = (tokens[i - 1], tokens[i])
    unigram = tokens[i]
    prob = trigram_model.get(trigram, bigram_model.get(bigram, unigram_model.get(unigram, 1e-6)))
    log_prob_sum += math.log2(prob)

  return 2 ** (-log_prob_sum / n)

def calculate_trigram_smoothed_perplexity(smoothed_trigram_prob, test_tokens, V):
  '''
  Calculates the perplexity of a test sequence using a smoothed trigram model.

  Arguments:
    - smoothed_trigram_prob: A function that returns the probability of a trigram
                              (e.g., from trigram_model_leplace)
    - test_tokens: A list of tokens from the test corpus
    - V: The vocabulary size used during smoothing (not directly used here, but kept for consistency)

  Returns:
    - Perplexity: A float value representing how well the model predicts the test set.
                  Lower is better; higher implies the model is less confident.

  Notes:
    - Uses log base 2 for probability accumulation
    - Starts from index 2 since trigrams require 2 previous tokens
    - Applies the standard formula:
        Perplexity = 2 ^ [ - (1/N) * ∑ log2 P(w_i | w_{i-2}, w_{i-1}) ]
    - Assumes the input model already applies Laplace smoothing

  Example:
    prob_fn, V = trigram_model_leplace(train_tokens)
    ppl = calculate_trigram_smoothed_perplexity(prob_fn, test_tokens, V)
  '''
  n = len(test_tokens)
  log_prob_sum = 0
  for i in range(2, n):
    w1, w2, w3 = test_tokens[i - 2], test_tokens[i - 1], test_tokens[i]
    prob = smoothed_trigram_prob(w1, w2, w3)
    log_prob_sum += math.log2(prob)
  return 2 ** (-log_prob_sum / n)