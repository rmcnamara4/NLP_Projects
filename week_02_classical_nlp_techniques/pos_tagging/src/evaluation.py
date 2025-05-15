from src.viterbi_algorithm import viterbi

def evaluate(test_data, initial_probs, transition_probs, emission_probs, unique_tags):
  """
  Evaluate the accuracy of a sequence tagging model using the Viterbi algorithm.

  Parameters:
      test_data (list of list of (str, str)): A list of sentences, where each sentence is a 
                                              list of (word, true_tag) pairs.
      initial_probs (dict): Initial tag probabilities P(tag_0).
      transition_probs (dict): Transition probabilities P(tag_t | tag_{t-1}).
      emission_probs (function): A function that returns emission probabilities P(word | tag).
      unique_tags (set or list): The full set of possible tags.

  Returns:
      float: The overall tagging accuracy (correct predictions / total tags).

  Notes:
      - Uses the Viterbi decoder to generate predicted tag sequences.
      - Assumes test data uses the same tag set and vocabulary as the training data.
  """
  correct, total = 0, 0
  for sentence in test_data:
    words, true_tags = zip(*sentence)

    predictions, probability = viterbi(words, initial_probs, transition_probs, emission_probs, unique_tags)
    predicted_tags = [tag for _, tag in predictions]

    correct += sum(pred == true for pred, true in zip(predicted_tags, true_tags))
    total += len(true_tags)

  return correct / total