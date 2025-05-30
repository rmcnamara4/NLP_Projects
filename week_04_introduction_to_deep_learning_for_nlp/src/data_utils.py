import torch 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
  """
  A PyTorch Dataset for handling tokenized text data and corresponding labels.

  This dataset:
  - Converts tokens in each sequence to their corresponding integer indices using a provided vocabulary.
  - Substitutes out-of-vocabulary tokens with the <UNK> index.
  - Returns input-target pairs suitable for training classification models.

  Args:
      X (List[List[str]]): A list of tokenized text sequences.
      y (List[int] or List[float]): A list of labels corresponding to each sequence.
      stoi (Dict[str, int]): A dictionary mapping tokens to integer indices.

  Returns:
      Tuple[torch.Tensor, int or float]: A tuple containing the tensor of token indices for a sequence
                                          and its associated label.
  """
  def __init__(self, X, y, stoi):
    self.X = X
    self.y = y
    self.stoi = stoi

  def __len__(self):
    """Returns the number of samples in the dataset."""
    return len(self.y)

  def __getitem__(self, idx):
    """
    Retrieves the tokenized and indexed representation of the sample at position `idx`.

    Args:
        idx (int): Index of the sample to retrieve.

    Returns:
        Tuple[torch.Tensor, int or float]: A tensor of token indices and the corresponding label.
    """
    text = self.X[idx]
    ids = torch.tensor([self.stoi.get(t, self.stoi['<UNK>']) for t in text])
    labels = self.y[idx]

    return ids, labels
  
def collate_fn(batch, stoi):
    """
    Custom collate function for DataLoader to handle variable-length text sequences.

    This function:
    - Pads sequences in the batch to the same length using the <PAD> token index.
    - Converts labels to a float tensor.
    - Computes the original lengths of each sequence before padding.

    Args:
        batch (List[Tuple[torch.Tensor, int or float]]): A list of (sequence_tensor, label) tuples.
        stoi (Dict[str, int]): A dictionary mapping tokens to integer indices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - padded_texts: Tensor of shape (batch_size, max_seq_len) with padded sequences.
            - labels: Tensor of shape (batch_size,) containing float labels.
            - lengths: Tensor of shape (batch_size,) with original sequence lengths.
    """
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in texts])
    padded_texts = pad_sequence(texts, batch_first = True, padding_value = stoi['<PAD>'])
    return padded_texts, torch.tensor(labels, dtype = torch.float), lengths
  
