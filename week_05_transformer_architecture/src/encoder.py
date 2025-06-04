import torch.nn as nn
from src.encoder_layer import EncoderLayer

class Encoder(nn.Module):
  """
  Stacked Transformer encoder composed of multiple EncoderLayer modules.

  Each layer applies multi-head self-attention followed by a position-wise 
  feedforward network with residual connections and layer normalization.

  Args:
      n_layers (int): Number of encoder layers to stack.
      d_model (int): Dimensionality of input and output features.
      d_k (int): Dimensionality of keys in attention.
      d_v (int): Dimensionality of values in attention.
      n_heads (int): Number of attention heads.
      ffn_hidden (int): Hidden size of the feedforward network.
      dropout (float): Dropout rate used in attention and FFN layers.
      max_len (int): Maximum sequence length (optional, passed to layers if used).
      device (str): Device to place the encoder on (e.g., 'cpu' or 'cuda').
  """
  def __init__(self, n_layers, d_model, d_k, d_v, n_heads, ffn_hidden, dropout = 0.2, max_len = 512, device = 'cpu'):
    super(Encoder, self).__init__()
    self.n_layers = n_layers
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ffn_hidden = ffn_hidden
    self.dropout = dropout
    self.max_len = max_len
    self.device = device

    self.layers = nn.ModuleList([
        EncoderLayer(d_model, d_k, d_v, n_heads, ffn_hidden, dropout, max_len, device) for _ in range(n_layers)
    ])

  def forward(self, x, encoder_input_ids = None, padding_idx = None):
    """
    Forward pass through the encoder stack.

    Applies each encoder layer sequentially to the input.

    Args:
        x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model).
        encoder_input_ids (Tensor, optional): Input token IDs used to generate padding masks.
        padding_idx (int, optional): Index of padding token in the vocabulary.

    Returns:
        Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    for layer in self.layers:
      x = layer(x, encoder_input_ids, padding_idx)

    return x