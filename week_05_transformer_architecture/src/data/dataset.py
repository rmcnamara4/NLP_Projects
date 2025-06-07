import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split
import pandas as pd 
from datasets import load_dataset

import swifter

from src.data.preprocessing import numericalize, tokenize

def load_tokenized_data(src_lang, tgt_lang, dataset_name = 'opus_books'): 
  """
  Loads and tokenizes a specific split of a Hugging Face dataset.

  Args:
      src_lang (str): Source language code (e.g., 'en').
      tgt_lang (str): Target language code (e.g., 'fr').
      dataset_name (str): Name of the Hugging Face dataset. Default is 'opus_books'.

  Returns:
      tuple: A tuple containing three elements:
          - (train_source_tokens, train_target_tokens)
          - (val_source_tokens, val_target_tokens)
          - (test_source_tokens, test_target_tokens)

      Each element is a tuple of tokenized source and target sequences for that split.
  """
  dataset = load_dataset(dataset_name, f'{src_lang}-{tgt_lang}')
  data = pd.DataFrame(dataset['train']['translation'])

  source = data[src_lang]
  target = data[tgt_lang]

  eval_source, test_source, eval_target, test_target = train_test_split(source, target, test_size = 0.2, random_state = 10)
  train_source, val_source, train_target, val_target = train_test_split(eval_source, eval_target, test_size = 0.2, random_state = 10)

  train_source_tokens = train_source.swifter.apply(tokenize).values
  train_target_tokens = train_target.swifter.apply(tokenize).values

  val_source_tokens = val_source.swifter.apply(tokenize).values
  val_target_tokens = val_target.swifter.apply(tokenize).values

  test_source_tokens = test_source.swifter.apply(tokenize).values
  test_target_tokens = test_target.swifter.apply(tokenize).values

  return (train_source_tokens, train_target_tokens), (val_source_tokens, val_target_tokens), (test_source_tokens, test_target_tokens)
  

def collate_fn(batch, source_vocab, target_vocab, train = True):
  """
  Collate function for preparing mini-batches of sequence pairs for a sequence-to-sequence model.

  This function:
    - Pads source and target sequences in the batch to the same length using their respective <PAD> tokens.
    - Splits target sequences into input and output for teacher forcing during training.
    - Prepares an initial target input tensor with only the <SOS> token for inference when `train` is False.

  Args:
      batch (list of tuples): A list of (src_ids, target_ids) tensor pairs from the dataset.
      source_vocab (Vocab): Vocabulary object for the source language. Used to determine padding index.
      target_vocab (Vocab): Vocabulary object for the target language. Used to determine padding and <SOS> token.
      train (bool, optional): Whether the model is in training mode. Defaults to True.
                              If True, returns teacher-forced target input and target output.
                              If False, returns a <SOS>-only start for inference.

  Returns:
      tuple:
          - src_input (Tensor): Padded source input tensor of shape (batch_size, max_src_len).
          - target_input (Tensor): Target input tensor (for decoder), teacher-forced or <SOS> depending on mode.
          - target_output (Tensor): Target output tensor for loss computation.
  """
  src_ids, target_ids = zip(*batch)

  src_ids = pad_sequence(src_ids, batch_first = True, padding_value = source_vocab['<PAD>'])
  target_ids = pad_sequence(target_ids, batch_first = True, padding_value = target_vocab['<PAD>'])

  if train:
    src_input = src_ids
    target_input = target_ids[:, :-1]
    target_output = target_ids[:, 1:]
  else:
    src_input = src_ids
    target_input = torch.full(
        (src_ids.size(0), 1),
        fill_value = target_vocab['<SOS>'],
        dtype = torch.long
    )
    target_output = target_ids

  return src_input, target_input, target_output

class TranslationDataset(Dataset):
  """
  Custom Dataset class for sequence-to-sequence translation tasks.

  This dataset pairs source and target text sequences and converts them 
  into numerical tensors using provided vocabularies.

  Args:
      src_text (list[list[str]]): List of tokenized source language sentences.
      target_text (list[list[str]]): List of tokenized target language sentences.
      src_vocab (dict or Vocab): Vocabulary mapping source tokens to indices.
      target_vocab (dict or Vocab): Vocabulary mapping target tokens to indices.
  """
  def __init__(self, src_text, target_text, src_vocab, target_vocab):
    self.src_text = src_text
    self.target_text = target_text
    self.src_vocab = src_vocab
    self.target_vocab = target_vocab

  def __len__(self):
    """
    Returns the number of samples in the dataset.

    Returns:
        int: Number of sentence pairs.
    """
    return len(self.src_text)

  def __getitem__(self, idx):
    """
    Retrieves and numericalizes the source and target sentence pair at the given index.

    Args:
        idx (int): Index of the sentence pair.

    Returns:
        tuple: A pair of tensors (src_ids, target_ids) representing the numericalized 
                source and target sentences.
    """
    src_ids = numericalize(self.src_text[idx], self.src_vocab)
    target_ids = numericalize(self.target_text[idx], self.target_vocab)
    return src_ids, target_ids