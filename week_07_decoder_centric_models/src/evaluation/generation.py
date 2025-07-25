from tqdm import tqdm
import torch 

def generate_summaries(cfg, model, dataloader, tokenizer, device = 'cuda'): 
    """
    Generates chunk-level summaries using a trained language model.

    This function performs batched inference over the input dataloader and applies generation 
    techniques such as beam search, sampling, and repetition penalties based on the provided config.
    The function assumes each input corresponds to a chunk of an article and associates predictions 
    with their article ID.

    Args:
        cfg: A config object or dict containing generation parameters such as:
            - max_new_tokens (int): Maximum number of tokens to generate.
            - num_beams (int): Beam width for beam search.
            - do_sample (bool): Whether to sample (used for nucleus/top-k sampling).
            - top_p (float): Top-p (nucleus) sampling threshold.
            - top_k (int): Top-k sampling threshold.
            - temperature (float): Sampling temperature.
            - repetition_penalty (float): Penalty for repeated tokens.
            - length_penalty (float): Length penalty during generation.
            - no_repeat_ngram_size (int): Prevent repeating n-grams of this size.
            - early_stopping (bool, optional): Whether to stop early in beam search.
        model (torch.nn.Module): The language model to use for generation.
        dataloader (DataLoader): A PyTorch DataLoader that yields input_ids, attention_mask, and article_id.
        tokenizer (PreTrainedTokenizer): The tokenizer used to decode predictions.
        device (str, optional): Device to run inference on. Defaults to 'cuda'.

    Returns:
        dict: A dictionary mapping article IDs to lists of chunk-level summary strings.
              Format: {article_id: [summary_chunk1, summary_chunk2, ...]}
    """
    model.eval()
    model.to(device)

    all_preds = {}

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Safety check: match shapes
            assert input_ids.shape == attention_mask.shape, \
                f'Shape mismatch: {input_ids.shape} vs {attention_mask.shape}'

            # Generate
            generated_ids = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = cfg.max_new_tokens,
                num_beams = cfg.num_beams,
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                early_stopping = cfg.get('early_stopping', False),
                do_sample = cfg.do_sample, 
                top_p = cfg.get('top_p', None),
                top_k = cfg.get('top_k', None),
                temperature = cfg.get('temperature', None), 
                repetition_penalty = cfg.repetition_penalty, 
                length_penalty = cfg.length_penalty,
                no_repeat_ngram_size = cfg.no_repeat_ngram_size
            )

            # Decode predictions
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)

            for pred, aid in zip(decoded_preds, batch['article_id']):
                if 'TL;DR:' in pred:
                    pred = pred.split('TL;DR:')[1].strip()
                else:
                    pred = pred.strip()  # Fallback in case split fails

                if aid not in all_preds:
                    all_preds[aid] = []
                all_preds[aid].append(pred)

    return all_preds

def resummarize_chunks(cfg, all_preds, model, tokenizer, device = 'cuda'): 
    """
    Combines chunk-level summaries into a full-text summary per article using a second-stage 
    generation pass.

    This function:
    - Concatenates the chunk-level summaries for each article.
    - Prepends a prompt and appends a TL;DR token to guide the model.
    - Applies generation with either beam search or sampling, based on the given config.
    - Extracts the final generated summary after the 'TL;DR:' prompt.

    Args:
        cfg: A config object or dict containing generation parameters such as:
            - max_new_tokens (int): Maximum number of tokens to generate.
            - num_beams (int): Beam width for beam search.
            - do_sample (bool): Whether to use sampling.
            - top_p (float): Nucleus sampling threshold.
            - top_k (int): Top-k sampling threshold.
            - temperature (float): Sampling temperature.
            - repetition_penalty (float): Repetition penalty for decoding.
            - length_penalty (float): Length penalty during generation.
            - no_repeat_ngram_size (int): N-gram repetition restriction.
            - early_stopping (bool, optional): Whether to stop early in beam search.
        all_preds (dict): A dictionary mapping article IDs to lists of chunk summaries.
                          Format: {article_id: [chunk_summary1, chunk_summary2, ...]}
        model (torch.nn.Module): The language model used for summarization.
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer used to encode/decode text.
        device (str, optional): Device for computation ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        dict: A dictionary mapping article IDs to their final resummarized string.
              Format: {article_id: final_summary}
    """
    final_summaries = {}

    model.to(device)
    model.eval()

    with torch.no_grad():
        for aid, chunk_summaries in tqdm(all_preds.items()):
            combined_text = " ".join(chunk_summaries)
            combined_ids = tokenizer.encode(combined_text, return_tensors = 'pt', add_special_tokens = False).squeeze()

            if len(combined_ids) > tokenizer.model_max_length - cfg.max_new_tokens - 25: 
              combined_ids = combined_ids[:tokenizer.model_max_length - cfg.max_new_tokens - 25]
            
            summary_ids = tokenizer.encode('Summarize this: ', return_tensors = 'pt', add_special_tokens = False).squeeze()
            tldr_ids = tokenizer.encode('\nTL;DR: ', add_special_tokens = False, return_tensors = 'pt').squeeze()
            
            prompt_ids = torch.concat([summary_ids, combined_ids, tldr_ids])

            inputs = {
                'input_ids': prompt_ids.unsqueeze(0).to(device),
                'attention_mask': torch.ones_like(prompt_ids).unsqueeze(0).to(device)
            }

            summary_ids = model.generate(
                **inputs,
                max_new_tokens = cfg.max_new_tokens,
                num_beams = cfg.num_beams,
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                early_stopping = cfg.get('early_stopping', False),
                do_sample = cfg.do_sample, 
                top_p = cfg.get('top_p', None),
                top_k = cfg.get('top_k', None),
                temperature = cfg.get('temperature', None), 
                repetition_penalty = cfg.repetition_penalty, 
                length_penalty = cfg.length_penalty,
                no_repeat_ngram_size = cfg.no_repeat_ngram_size
            )

            final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            final_summaries[aid] = final_summary.split('TL;DR:')[1].strip()

    return final_summaries