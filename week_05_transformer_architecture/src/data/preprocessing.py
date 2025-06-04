import re 
from collections import Counter

import torch 
from torchtext.vocab import vocab

def tokenize(text):
  """
  Tokenize the input text into lowercase words and punctuation.

  This function uses a regular expression to extract words and contractions 
  (e.g., "don't", "it's") as well as standalone punctuation marks from the input string.
  All tokens are converted to lowercase.

  Args:
      text (str): The input string to tokenize.

  Returns:
      list[str]: A list of lowercase tokens (words and punctuation).
  """
  tokens = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text.lower())
  return tokens

def numericalize(tokens, vocab, add_sos = True, add_eos = True):
  """
  Convert a list of tokens into a tensor of token IDs using a given vocabulary.

  Optionally prepends a <SOS> (start-of-sequence) token and/or appends an <EOS> 
  (end-of-sequence) token to the sequence.

  Args:
      tokens (list[str]): List of tokens from a sentence.
      vocab (dict or Vocab): A vocabulary mapping tokens to integer IDs.
      add_sos (bool, optional): Whether to add the <SOS> token at the beginning. Defaults to True.
      add_eos (bool, optional): Whether to add the <EOS> token at the end. Defaults to True.

  Returns:
      torch.Tensor: Tensor of token IDs with dtype torch.int.
  """
  ids = [vocab['<SOS>']] if add_sos else []
  ids.extend([vocab[token] for token in tokens])

  if add_eos:
    ids.append(vocab['<EOS>'])

  return torch.tensor(ids, dtype = torch.int)

def build_vocab(sentences):
  """
  Build a vocabulary from a list of tokenized sentences.

  This function counts the frequency of each token across all sentences and 
  constructs a vocabulary object, including special tokens for padding, unknown words,
  start-of-sequence, and end-of-sequence.

  Args:
      sentences (list[list[str]]): A list of tokenized sentences (each sentence is a list of strings).

  Returns:
      torchtext.vocab.Vocab: A vocabulary object mapping tokens to indices.
  """
  counter = Counter()
  for sent in sentences:
    counter.update(sent)
  return vocab(counter, specials = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'])