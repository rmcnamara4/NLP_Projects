import torch 
import torch.nn as nn
import torch.nn.functional as F 

class Attention(nn.Module):
  """
  Applies an attention mechanism over LSTM outputs to produce a context vector.
  
  This module computes attention scores for each timestep in the LSTM output,
  applies a softmax to get weights, and uses these weights to calculate a weighted
  sum of the LSTM outputs.
  
  Args:
      hidden_dim (int): Dimensionality of the LSTM hidden state.
  """
  def __init__(self, hidden_dim):
    super(Attention, self).__init__()
    self.attn = nn.Linear(hidden_dim, 1)

  def forward(self, lstm_outputs, mask = None):
    """
    Compute the attention-weighted context vector from LSTM outputs.

    Args:
        lstm_outputs (Tensor): Tensor of shape (batch_size, seq_len, hidden_dim) 
                                representing outputs from the LSTM.
        mask (Tensor, optional): Boolean mask of shape (batch_size, seq_len) indicating 
                                  valid (non-padded) tokens.

    Returns:
        context (Tensor): Weighted sum of LSTM outputs. Shape (batch_size, hidden_dim).
        weights (Tensor): Attention weights for each token. Shape (batch_size, seq_len).
    """
    scores = self.attn(lstm_outputs).squeeze(-1)

    if mask is not None:
      scores = scores.masked_fill(~mask, float('-inf'))

    weights = F.softmax(scores, dim = 1)
    context = torch.bmm(weights.unsqueeze(1), lstm_outputs)

    return context.squeeze(1), weights

def create_mask(lengths, max_len=None):
    """
    Creates a boolean mask for padded sequences based on sequence lengths.

    This mask is used to identify the valid (non-padded) positions in each sequence,
    typically for attention mechanisms or loss calculations.

    Args:
        lengths (Tensor): A 1D tensor of shape (batch_size,) containing the lengths of each sequence.
        max_len (int, optional): The maximum sequence length to pad to. If None, uses the max value from `lengths`.

    Returns:
        Tensor: A boolean mask of shape (batch_size, max_len) where True indicates a valid token position.
    """
    batch_size = lengths.size(0)
    max_len = max_len or lengths.max()
    return torch.arange(max_len).expand(batch_size, max_len).to(lengths.device) < lengths.unsqueeze(1)