def convert_counts_to_probabilities(counts):
  """
  Convert nested count dictionaries to probability distributions.

  Parameters:
      counts (dict): A nested dictionary where each outer key maps to a dictionary 
                      of counts (e.g., {state: {next_state: count, ...}}).

  Returns:
      dict: A nested dictionary with the same structure, but with normalized 
            probabilities instead of raw counts.
            (e.g., {state: {next_state: prob, ...}})
  """
  probs = {}
  for key, counts in counts.items():
    total = sum(counts.values())
    probs[key] = {k: v / total for k, v in counts.items()}
  return probs

def get_emission_probability(emission_probs, tag, word, smoothing = 1e-6):
  """
  Retrieve the emission probability of a word given a tag, with optional smoothing.

  Parameters:
      emission_probs (dict): A nested dictionary of emission probabilities,
                              where emission_probs[tag][word] gives P(word | tag).
      tag (str): The POS tag or state.
      word (str): The observed word/token.
      smoothing (float, optional): A small fallback probability used if the word
                                    is not found under the given tag. Default is 1e-6.

  Returns:
      float: The emission probability P(word | tag), or the smoothing value if the word is unseen.

  Notes:
      - This function uses add-ε smoothing to handle unseen word-tag pairs.
      - If the word is not found in the emission dictionary for the given tag,
        the function returns the smoothing value instead.
  """
  return emission_probs[tag].get(word, smoothing)