import torch 
import logging
from nltk.translate.bleu_score import corpus_bleu

def length_penalty(length, alpha = 0.6):
  """
  Computes a length penalty to normalize sequence log-probabilities during beam search.

  This function is used to prevent the model from favoring shorter sequences by penalizing
  shorter outputs less and longer outputs more based on their lengths.

  Args:
      length (int): The length of the sequence.
      alpha (float, optional): Hyperparameter controlling penalty strength. Default is 0.6.

  Returns:
      float: The computed length penalty factor.
  """
  return ((5 + length) / 6) ** alpha

def beam_search_decode(model, source, beam_width = 5, max_len = 100, alpha = 0.6, device = 'cuda'):
  """
  Performs beam search decoding using the provided Transformer model.

  This function generates a translation by expanding multiple candidate sequences at each time step,
  keeping only the top sequences ranked by their accumulated log-probabilities (adjusted by a length penalty).
  It returns the most likely output token ID sequence.

  Args:
      model (nn.Module): The trained Transformer model with encoder and decoder.
      source (torch.Tensor): A tensor containing token IDs of the source sentence (shape: [seq_len]).
      beam_width (int, optional): Number of candidate sequences to keep during decoding. Default is 5.
      max_len (int, optional): Maximum length of generated sequence. Default is 100.
      alpha (float, optional): Length penalty hyperparameter. Higher values favor longer sequences. Default is 0.6.
      device (str, optional): Device to perform computation on ('cuda' or 'cpu'). Default is 'cuda'.

  Returns:
      list: The best decoded token ID sequence as a list of integers.
  """
  sos_id = model.target_vocab['<SOS>']
  eos_id = model.target_vocab['<EOS>']

  live_sequences = [(torch.tensor([sos_id]), 0.0)]
  completed_sequences = []

  source = source.unsqueeze(0)

  encoder_embeddings = model.encoder_embedding_layer(source)
  encoder_pos_embeddings = model.positional_embeddings[:, :source.shape[1], :]
  encoder_embeddings = encoder_embeddings + encoder_pos_embeddings
  encoder_embeddings = model.encoder_emb_drop(encoder_embeddings)

  encoder_output = model.encoder(encoder_embeddings, encoder_input_ids = source, padding_idx = model.source_vocab['<PAD>'])

  for step in range(max_len):
    all_candidates = []
    for seq, score in live_sequences:
      if seq[-1].item() == eos_id:
        completed_sequences.append((seq, score))
        continue

      seq = seq.to(device)
      seq = seq.unsqueeze(0)

      decoder_embeddings = model.decoder_embedding_layer(seq)
      decoder_pos_embeddings = model.positional_embeddings[:, :seq.shape[1], :]
      decoder_embeddings = decoder_embeddings + decoder_pos_embeddings
      decoder_embeddings = model.decoder_emb_drop(decoder_embeddings)

      decoder_output = model.decoder(decoder_embeddings, encoder_output, decoder_input_ids = seq, padding_idx = model.target_vocab['<PAD>'])
      if model.return_attn:
        decoder_output = decoder_output[0]

      logits = model.output_fc(decoder_output[:, -1, :])
      probs = torch.log_softmax(logits, dim = -1).squeeze(0)

      topk_log_probs, topk_indices = torch.topk(probs, beam_width)

      for i in range(beam_width):
        token = topk_indices[i].unsqueeze(0)
        candidate_seq = torch.cat([seq.squeeze(0), token])
        candidate_score = score + topk_log_probs[i].item()
        all_candidates.append((candidate_seq, candidate_score))

    live_sequences = sorted(all_candidates, key = lambda x: x[1], reverse = True)[:beam_width]

    if not live_sequences:
      break

  all_final_sequences = live_sequences + completed_sequences
  all_final_sequences = [(seq, prob / length_penalty(len(seq) - 1)) for seq, prob in all_final_sequences]
  all_final_sequences = sorted(all_final_sequences, key = lambda x: x[1], reverse = True)

  return all_final_sequences[0][0].tolist()

def decode(ids, vocab):
  """
  Converts a list of token IDs back into their corresponding tokens using the vocabulary.

  This function performs inverse tokenization and stops decoding at the <EOS> token 
  if it exists in the sequence.

  Args:
      ids (list or Tensor): Sequence of token IDs to decode.
      vocab (Vocab): Vocabulary object with an `get_itos()` method that returns index-to-token mapping.

  Returns:
      list: List of decoded tokens as strings, truncated at <EOS> if present.
  """
  itos = {k: v for k, v in enumerate(vocab.get_itos())}
  tokens = [itos[t] for t in ids]
  if '<EOS>' in tokens:
    tokens = tokens[:tokens.index('<EOS>')]
  return tokens

def evaluate_bleu(model, dataloader, smoother, beam_width = 5, max_len = 100, alpha = 0.6, device = 'cuda'):
  """
  Evaluates the translation quality of a Transformer model using corpus-level BLEU score.

  This function runs the model in evaluation mode, decodes predictions using beam search,
  and compares them to the ground truth targets to compute the BLEU score. It prints a few
  example translations and the final BLEU score.

  Args:
      model (nn.Module): Trained Transformer model for sequence-to-sequence translation.
      dataloader (DataLoader): DataLoader yielding source and target sequences.
      smoother (function): Smoothing function from nltk.translate.bleu_score.
      beam_width (int, optional): Beam width used in beam search decoding. Default is 5.
      max_len (int, optional): Maximum length of generated sequences. Default is 100.
      alpha (float, optional): Length penalty factor in beam search. Default is 0.6.
      device (str, optional): Device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.

  Returns:
      float: The corpus-level BLEU score of the model’s predictions.
  """
  model.eval()

  predictions = []
  reference = []
  with torch.no_grad():
    for i, (src_input, tgt_input, tgt_output) in enumerate(dataloader):
      if i % 1 == 0:
        logging.info(f'Batch [{i}] / [{len(dataloader)}]')

      batch_size = src_input.shape[0]
      for j in range(batch_size):
        src = src_input[j].to(device)
        if tgt_output.dim() == 1:
            tgt = tgt_output.tolist()
        else:
            tgt = tgt_output[j].tolist()

        pred_ids = beam_search_decode(model, src, beam_width, max_len, alpha, device)

        tgt_tokens = decode(tgt, model.target_vocab)
        pred_tokens = decode(pred_ids, model.target_vocab)

        predictions.append(pred_tokens)
        reference.append([tgt_tokens])

        if i == 0 and j < 5:
          logging.info(f'Example {j + 1}:')
          logging.info('Reference:' + ' '.join(tgt_tokens))
          logging.info('Hypothesis:' + ' '.join(pred_tokens))
          logging.info('\n')

  bleu = corpus_bleu(reference, predictions, smoothing_function = smoother)
  logging.info(f'Corpus BLEU: {bleu:.4f}')

  return bleu
