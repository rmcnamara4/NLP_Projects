import torch.nn as nn
from src.encoder import Encoder 
from src.decoder import Decoder
from src.position import get_positional_embeddings

class TransformerModel(nn.Module):
  """
  Full Transformer model for sequence-to-sequence tasks such as machine translation.

  This model includes:
  - Source and target token embeddings
  - Positional embeddings (shared sinusoidal encoding)
  - Encoder and decoder stacks composed of multiple Transformer layers
  - Final linear projection layer to map decoder outputs to vocabulary logits

  Args:
      source_vocab (dict): Mapping of source language tokens to indices.
      target_vocab (dict): Mapping of target language tokens to indices.
      d_model (int): Dimensionality of model embeddings and internal representations.
      d_k (int): Dimensionality of key vectors per attention head.
      d_v (int): Dimensionality of value vectors per attention head.
      n_heads (int): Number of attention heads in multi-head attention.
      ffn_hidden (int): Hidden layer size in feed-forward sublayers.
      n_layers (int): Number of encoder and decoder layers.
      encoder_dropout (float): Dropout rate applied in encoder.
      decoder_dropout (float): Dropout rate applied in decoder.
      max_len (int): Maximum sequence length for positional embeddings.
      return_attn (bool): Whether to return attention weights for visualization.
      device (str): Device to place tensors on ('cpu' or 'cuda').
  """
  def __init__(self, source_vocab, target_vocab, d_model, d_k, d_v, n_heads, ffn_hidden, n_layers, encoder_dropout = 0.2, decoder_dropout = 0.2, max_len = 512, return_attn = True, device = 'cpu'):
    super(TransformerModel, self).__init__()
    self.source_vocab = source_vocab
    self.target_vocab = target_vocab
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ffn_hidden = ffn_hidden
    self.encoder_dropout = encoder_dropout
    self.decoder_dropout = decoder_dropout
    self.max_len = max_len
    self.device = device
    self.return_attn = return_attn

    self.encoder_embedding_layer = nn.Embedding(len(source_vocab), d_model)
    self.decoder_embedding_layer = nn.Embedding(len(target_vocab), d_model)

    pos_embeddings = get_positional_embeddings(max_len, d_model, device)
    self.register_buffer('positional_embeddings', pos_embeddings)

    self.encoder = Encoder(n_layers, d_model, d_k, d_v, n_heads, ffn_hidden, encoder_dropout, max_len, device)
    self.decoder = Decoder(n_layers, d_model, d_k, d_v, n_heads, ffn_hidden, decoder_dropout, max_len, return_attn, device)

    self.encoder_emb_drop = nn.Dropout(encoder_dropout)
    self.decoder_emb_drop = nn.Dropout(decoder_dropout)

    self.output_fc = nn.Linear(d_model, len(target_vocab))

  def forward(self, source, target):
    """
    Forward pass of the Transformer model.

    Args:
        source (Tensor): Input tensor of token indices for the source sequence,
                          shape (batch_size, source_seq_len).
        target (Tensor): Input tensor of token indices for the target sequence,
                          shape (batch_size, target_seq_len).

    Returns:
        Union[
            Tensor,
            Tuple[Tensor, List[Tensor], List[Tensor]]
        ]:
            - If return_attn is False:
                - logits (Tensor): Output tensor of shape (batch_size, target_seq_len, vocab_size).
            - If return_attn is True:
                - logits (Tensor): Output predictions.
                - self_attn (List[Tensor]): Self-attention weights for each decoder layer.
                - cross_attn (List[Tensor]): Cross-attention weights for each decoder layer.
    """
    encoder_embeddings = self.encoder_embedding_layer(source)
    encoder_pos_embeddings = self.positional_embeddings[:, :source.shape[1], :]
    encoder_embeddings = encoder_embeddings + encoder_pos_embeddings
    encoder_embeddings = self.encoder_emb_drop(encoder_embeddings)

    encoder_output = self.encoder(encoder_embeddings, encoder_input_ids = source, padding_idx = self.source_vocab['<PAD>'] if '<PAD>' in self.source_vocab else None)
    print('Encoder Output shape:', encoder_output.shape)

    decoder_embeddings = self.decoder_embedding_layer(target)
    decoder_pos_embeddings = self.positional_embeddings[:, :target.shape[1], :]
    decoder_embeddings = decoder_embeddings + decoder_pos_embeddings
    decoder_embeddings = self.decoder_emb_drop(decoder_embeddings)

    decoder_output = self.decoder(decoder_embeddings, encoder_output, decoder_input_ids = target, encoder_input_ids = source, padding_idx = self.source_vocab['<PAD>'] if '<PAD>' in self.source_vocab else None)
    if self.return_attn:
      output, self_attn, cross_attn = decoder_output
      return self.output_fc(output), self_attn, cross_attn
    else:
      return self.output_fc(decoder_output)
