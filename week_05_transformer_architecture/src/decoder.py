import torch.nn as nn
from src.decoder_layer import DecoderLayer

class Decoder(nn.Module):
  """
  Stacked Transformer decoder composed of multiple DecoderLayer modules.

  The decoder attends to its own past outputs (via masked self-attention)
  and to the encoder's output (via cross-attention), layer by layer.

  Args:
      n_layers (int): Number of decoder layers to stack.
      d_model (int): Dimensionality of input and output features.
      d_k (int): Dimensionality of keys in attention.
      d_v (int): Dimensionality of values in attention.
      n_heads (int): Number of attention heads.
      ffn_hidden (int): Hidden size of the feedforward network.
      dropout (float): Dropout rate used in attention and FFN layers.
      max_len (int): Maximum sequence length.
      return_attn (bool): Whether to return attention weights for analysis.
      device (str): Device to place the decoder on (e.g., 'cpu' or 'cuda').
  """
  def __init__(self, n_layers, d_model, d_k, d_v, n_heads, ffn_hidden, dropout = 0.2, max_len = 512, return_attn = True, device = 'cpu'):
    super(Decoder, self).__init__()
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ffn_hidden = ffn_hidden
    self.dropout = dropout
    self.max_len = max_len
    self.device = device
    self.return_attn = return_attn

    self.layers = nn.ModuleList([
        DecoderLayer(d_model, d_k, d_v, n_heads, ffn_hidden, dropout, max_len, device) for _ in range(n_layers)
    ])

  def forward(self, x, encoder_output, decoder_input_ids = None, encoder_input_ids = None, padding_idx = None):
    """
    Forward pass through the decoder stack.

    Applies each decoder layer sequentially. Supports returning self-attention
    and cross-attention weights from all layers.

    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_model), typically target embeddings.
        encoder_output (Tensor): Output from the encoder (batch_size, src_seq_len, d_model).
        decoder_input_ids (Tensor, optional): Input token IDs used to generate decoder padding/causal masks.
        encoder_input_ids (Tensor, optional): Encoder input IDs for cross-attention masking.
        padding_idx (int, optional): Index of padding token in the vocabulary.

    Returns:
        - If return_attn is True:
            Tuple[Tensor, List[Tensor], List[Tensor]]:
            - Final decoder output (batch_size, seq_len, d_model)
            - List of self-attention weights for each layer
            - List of cross-attention weights for each layer
        - If return_attn is False:
            Tensor: Final decoder output (batch_size, seq_len, d_model)
    """
    all_self_attn = []
    all_cross_attn = []

    for layer in self.layers:
      x, self_attn, cross_attn = layer(x, encoder_output, decoder_input_ids, encoder_input_ids, padding_idx)
      if self.return_attn:
        all_self_attn.append(self_attn)
        all_cross_attn.append(cross_attn)

    if self.return_attn:
      return x, all_self_attn, all_cross_attn
    else:
      return x