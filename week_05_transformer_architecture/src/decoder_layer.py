import torch.nn as nn 
import torch 
from src.multihead_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
  """
  Transformer decoder layer composed of masked self-attention, cross-attention,
  and a position-wise feedforward network, each followed by residual connections 
  and layer normalization. Designed to handle causal masking and optional padding masks.

  Args:
      d_model (int): Dimensionality of input and output features.
      d_k (int): Dimensionality of keys in attention.
      d_v (int): Dimensionality of values in attention.
      n_heads (int): Number of attention heads.
      ffn_hidden (int): Number of hidden units in the feedforward network.
      dropout (float): Dropout probability applied after attention and FFN layers.
      max_len (int): Maximum sequence length (not used here, but commonly for positional encodings).
      device (str): Device to place tensors and modules ('cpu' or 'cuda').
  """
  def __init__(self, d_model, d_k, d_v, n_heads, ffn_hidden, dropout = 0.2, max_len = 512, device = 'cpu'):
    super(DecoderLayer, self).__init__()
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.dropout = dropout

    self.self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
    self.cross_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)

    self.ffn = nn.Sequential(
        nn.Linear(d_model, ffn_hidden),
        nn.ReLU(),
        nn.Linear(ffn_hidden, d_model)
    )

    self.drop_emb = nn.Dropout(dropout)
    self.drop_ffn = nn.Dropout(dropout)

  def forward(self, x, encoder_output, decoder_input_ids = None, encoder_input_ids = None, padding_idx = None):
    """
    Forward pass through the decoder layer.

    Applies:
    - Masked multi-head self-attention (with causal and padding masks)
    - Multi-head cross-attention over the encoder outputs
    - Position-wise feedforward transformation
    - Residual connections and layer normalization

    Args:
        x (Tensor): Decoder input of shape (batch_size, seq_len, d_model).
        encoder_output (Tensor): Encoder output of shape (batch_size, src_len, d_model).
        decoder_input_ids (Tensor, optional): Token IDs of decoder input (used to compute padding mask).
        encoder_input_ids (Tensor, optional): Token IDs of encoder input (used to compute padding mask).
        padding_idx (int, optional): Index of padding token.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 
            - Output tensor of shape (batch_size, seq_len, d_model)
            - Self-attention weights (batch_size, n_heads, seq_len, seq_len)
            - Cross-attention weights (batch_size, n_heads, seq_len, src_len)
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device = x.device)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    if decoder_input_ids is not None and padding_idx is not None:
      pad_mask = (decoder_input_ids != padding_idx).unsqueeze(1).unsqueeze(2)
      mask = causal_mask & pad_mask
    else:
      mask = causal_mask

    attn_mask = mask.masked_fill(~mask, float('-inf')).float()

    self_attn_output, self_attn_weights = self.self_attn(x, x, x, attn_mask)
    decoder_hidden = self.norm1(self_attn_output + x)

    if encoder_input_ids is not None and padding_idx is not None:
        enc_pad_mask = (encoder_input_ids != padding_idx).unsqueeze(1).unsqueeze(2)
        cross_attn_mask = enc_pad_mask.masked_fill(~enc_pad_mask, float('-inf')).float()
    else:
        cross_attn_mask = None

    cross_attn_output, cross_attn_weights = self.cross_attn(decoder_hidden, encoder_output, encoder_output, mask = cross_attn_mask)
    cross_attn_output = self.norm2(cross_attn_output + decoder_hidden)

    output = self.drop_ffn(self.ffn(cross_attn_output))
    output = self.norm3(output + cross_attn_output)

    return output, self_attn_weights, cross_attn_weights