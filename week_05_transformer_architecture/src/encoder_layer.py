import torch.nn as nn
from src.multihead_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
  """
  A Transformer encoder layer consisting of multi-head self-attention followed by a feedforward network.
  
  Each sub-layer has residual connections and layer normalization, as described in "Attention is All You Need".

  Args:
      d_model (int): Dimension of the input and output embeddings.
      d_k (int): Dimension of the attention key vectors.
      d_v (int): Dimension of the attention value vectors.
      n_heads (int): Number of attention heads.
      ffn_hidden (int): Hidden layer size of the feedforward network.
      dropout (float, optional): Dropout rate to apply after attention and FFN. Default is 0.2.
      max_len (int, optional): Maximum sequence length. Default is 512 (not used in this implementation).
      device (str, optional): Device to place the model ('cpu' or 'cuda'). Default is 'cpu'.
    """
  def __init__(self, d_model, d_k, d_v, n_heads, ffn_hidden, dropout = 0.2, max_len = 512, device = 'cpu'):
    super(EncoderLayer, self).__init__()

    self.attention = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)

    self.ffn = nn.Sequential(
        nn.Linear(d_model, ffn_hidden),
        nn.ReLU(),
        nn.Linear(ffn_hidden, d_model)
    )

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    self.drop_emb = nn.Dropout(dropout)
    self.drop_ffn = nn.Dropout(dropout)

  def forward(self, x, encoder_input_ids = None, padding_idx = None):
    """
    Perform a forward pass through the encoder layer.

    Applies multi-head self-attention followed by a position-wise feedforward network,
    with residual connections and layer normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        encoder_input_ids (torch.Tensor, optional): Input IDs for constructing the padding mask.
        padding_idx (int, optional): Padding token ID for masking.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]

    if encoder_input_ids is not None and padding_idx is not None:
      pad_mask = (encoder_input_ids != padding_idx).unsqueeze(1).unsqueeze(2)
      attn_mask = pad_mask.masked_fill(~pad_mask, float('-inf')).float()
    else:
      attn_mask = None

    attention_output, _ = self.attention(x.float(), x.float(), x.float(), mask = attn_mask)
    attention_output = self.norm1(attention_output + x)

    output = self.ffn(attention_output)
    output = self.drop_ffn(output)
    output = self.norm2(output + attention_output)

    return output