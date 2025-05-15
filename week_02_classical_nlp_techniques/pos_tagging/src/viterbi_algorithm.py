import numpy as np 
import math

from src.hmm_utils import get_emission_probability

def viterbi(sentence, initial_probs, transition_probs, emission_probs, unique_tags):
  """
  Apply the Viterbi algorithm to find the most likely sequence of tags for a given sentence.

  Parameters:
      sentence (list of str): A list of words representing the observed sequence.
      initial_probs (dict): A dictionary of initial probabilities for each tag (P(tag_0)).
      transition_probs (dict): A nested dictionary representing transition probabilities 
                                between tags (P(tag_t | tag_{t-1})).
      emission_probs (function): A function that returns the emission probability 
                                  P(word | tag), typically with smoothing support.
      unique_tags (set or list of str): The set of all possible tags used in the model.

  Returns:
      tuple:
          - list of (word, tag) pairs representing the most likely tag sequence.
          - float: the probability of the best tag sequence.
          
  Notes:
      - Uses log probabilities to avoid numerical underflow during multiplication.
      - Applies add-one smoothing (1e-6) for unseen emission and transition probabilities.
      - Backpointers are used to reconstruct the best tag sequence from the dynamic programming table.
  """
  words = [word.lower() for word in sentence]
  n = len(sentence)
  m = len(unique_tags)
  tags_list = list(unique_tags)

  v_table = np.zeros((m, n))
  backpointer = np.zeros((m, n), dtype = int)

  for i, tag in enumerate(tags_list):
    v_table[i, 0] = math.log(initial_probs.get(tag, 1e-6)) + math.log(get_emission_probability(emission_probs, tag, words[0]))

  for i in range(1, n):
    for j in range(m):
      temp_values = []
      for k in range(m):
        temp_values.append(
            v_table[k, i - 1] + math.log(transition_probs.get(tags_list[k], {}).get(tags_list[j], 1e-6)) + math.log(get_emission_probability(emission_probs, tags_list[j], words[i]))
        )
      v_table[j, i] = max(temp_values)
      backpointer[j, i] = np.argmax(temp_values)

  best_path_prob = max(v_table[:, -1])
  best_path_pointer = np.argmax(v_table[:, -1])

  best_path = [best_path_pointer]
  for i in range(n - 1, 0, -1):
    best_path.append(int(backpointer[best_path[-1], i]))

  best_tags = [tags_list[i] for i in best_path[::-1]]

  return list(zip(sentence, best_tags)), math.exp(best_path_prob)