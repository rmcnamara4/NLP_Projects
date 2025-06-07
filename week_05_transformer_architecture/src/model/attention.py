import torch.nn as nn
import torch 
import torch.nn.functional as F 
import math 

class MultiHeadAttention(nn.Module):
  """
  Multi-head attention mechanism as described in "Attention is All You Need".

  This module projects the input query, key, and value tensors into multiple
  attention heads, performs scaled dot-product attention in parallel, and
  concatenates the results before projecting them back to the original dimension.

  Args:
      d_model (int): Dimensionality of the model (input and output features).
      d_k (int): Dimensionality of key vectors per head.
      d_v (int): Dimensionality of value vectors per head.
      n_heads (int): Number of attention heads.
      dropout (float): Dropout probability applied to attention outputs.
  """
  def __init__(self, d_model, d_k, d_v, n_heads, dropout = 0.2):
    super(MultiHeadAttention, self).__init__()
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.dropout = dropout

    self.q_proj = nn.Linear(d_model, d_k * n_heads)
    self.k_proj = nn.Linear(d_model, d_k * n_heads)
    self.v_proj = nn.Linear(d_model, d_v * n_heads)

    self.o_proj = nn.Linear(d_v * n_heads, d_model)

    self.drop_attn = nn.Dropout(dropout)

  def forward(self, query, key, value, mask = None):
    """
    Forward pass for multi-head attention.

    Args:
        query (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        key (Tensor): Key tensor of shape (batch_size, seq_len, d_model).
        value (Tensor): Value tensor of shape (batch_size, seq_len, d_model).
        mask (Tensor, optional): Attention mask of shape 
            (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len).
            Values should be 0 for tokens to attend to, and -inf for masked tokens.

    Returns:
        Tuple[Tensor, Tensor]:
            - output (Tensor): Final attention output of shape (batch_size, seq_len, d_model).
            - weights (Tensor): Raw weighted values before final linear projection 
              (used for visualization or attention analysis).
    """
    batch_size = query.shape[0]

    Q = self.q_proj(query)
    K = self.k_proj(key)
    V = self.v_proj(value)

    Q = Q.reshape(batch_size, query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
    K = K.reshape(batch_size, key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
    V = V.reshape(batch_size, value.shape[1], self.n_heads, self.d_v).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
      scores = scores + mask
    scores = F.softmax(scores, dim = -1)

    weights = torch.matmul(scores, V).transpose(1, 2).contiguous().reshape(batch_size, query.shape[1], self.d_v * self.n_heads)
    output = self.o_proj(weights)
    output = self.drop_attn(output)

    return output, weights