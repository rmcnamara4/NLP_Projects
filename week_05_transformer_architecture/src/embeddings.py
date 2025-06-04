import math 
import torch 

def get_positional_embeddings(seq_len, d_model, device):
  """
  Generate sinusoidal positional embeddings for a given sequence length and model dimensionality.

  This function implements the fixed positional encoding as described in the original
  Transformer paper: "Attention is All You Need" (Vaswani et al., 2017). It creates a tensor
  of shape (1, seq_len, d_model) where each position is encoded with a combination of sine and
  cosine functions of different frequencies.

  Args:
      seq_len (int): The length of the input sequence.
      d_model (int): The dimensionality of the model (embedding size).
      device (torch.device): The device to move the resulting tensor to (e.g., 'cpu' or 'cuda').

  Returns:
      torch.Tensor: A tensor of shape (1, seq_len, d_model) containing the positional encodings.
  """
  positional_embeddings = []
  for pos in range(seq_len):
    for dim in range(d_model):
      if dim % 2 == 0:
        positional_embeddings.append(math.sin(pos / (10_000 ** (2 * dim / d_model))))
      else:
        positional_embeddings.append(math.cos(pos / (10_000 ** (2 * dim / d_model))))

  return torch.tensor(positional_embeddings).reshape(seq_len, d_model).unsqueeze(0).to(device)