import torch.nn as nn
from src.model_utils import * 

##########################################################################################
# Base LTTM Classifier 
##########################################################################################

class LSTMClassifier(nn.Module):
  """
  A PyTorch neural network for sequence classification using an LSTM.

  This model consists of:
  - An embedding layer to convert token indices into dense vectors.
  - An LSTM layer to process the sequence of embeddings.
  - A fully connected layer to produce final output logits from the last hidden state.

  Args:
      vocab_size (int): Size of the vocabulary.
      embed_dim (int): Dimensionality of the embedding vectors.
      hidden_dim (int): Number of hidden units in the LSTM.
      output_dim (int): Dimensionality of the output (e.g., 1 for binary classification).
  """
  def __init__(self, stoi, embed_dim, hidden_dim, output_dim):
    super(LSTMClassifier, self).__init__()
    self.embedding = nn.Embedding(len(stoi), embed_dim, padding_idx = stoi['<PAD>'])
    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first = True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x, lengths):
    """
    Forward pass of the model.

    Args:
        x (Tensor): Tensor of token indices with shape (batch_size, seq_len).
        lengths (Tensor): Lengths of each sequence in the batch for packing.

    Returns:
        Tensor: Output logits from the final fully connected layer with shape (batch_size, output_dim).
    """
    embedded = self.embedding(x)
    packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first = True, enforce_sorted = False)
    out, (hidden, cell) = self.lstm(packed)
    return self.fc(hidden[-1])
  
##########################################################################################
# LSTM Classifier with Attention
##########################################################################################

class LSTMClassifierWithAttention(nn.Module):
  """
  An LSTM-based text classification model with an attention mechanism.

  This model embeds input tokens, processes them with an LSTM,
  and uses an attention mechanism to weight the LSTM outputs before
  making a prediction through a fully connected layer.

  Args:
      vocab_size (int): Size of the vocabulary.
      embed_dim (int): Dimensionality of the embedding vectors.
      hidden_dim (int): Number of hidden units in the LSTM.
      output_dim (int): Number of output classes (1 for binary classification).
  """
  def __init__(self, stoi, embed_dim, hidden_dim, output_dim):
    super(LSTMClassifierWithAttention, self).__init__()
    self.embedding = nn.Embedding(len(stoi), embed_dim, padding_idx = stoi['<PAD>'])
    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first = True)
    self.attn = Attention(hidden_dim)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x, lengths):
    """
    Forward pass of the model.

    Args:
        x (Tensor): Tensor of token indices of shape (batch_size, seq_len).
        lengths (Tensor): Lengths of each sequence in the batch (used for packing).

    Returns:
        Tensor: Output logits of shape (batch_size, output_dim).
    """
    embedded = self.embedding(x)
    packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first = True, enforce_sorted = False)
    out, (hidden, cell) = self.lstm(packed)

    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first = True)
    mask = create_mask(lengths)

    context, weights = self.attn(out, mask)

    return self.fc(context)
  