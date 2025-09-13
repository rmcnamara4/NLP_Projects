import torch 
from tqdm import tqdm

def get_quantized_response(prompts, pipe, cfg): 
    """
    Generates responses using a quantized local language model pipeline.

    This function processes a list of prompts using a quantized model (e.g., GGUF or GPTQ via Hugging Face or Text Generation Inference),
    supporting batched generation and sampling configuration.

    Args:
        prompts (str or list of str): A single prompt or a list of prompts to generate responses for.
        pipe (transformers.Pipeline): A text generation pipeline (e.g., from `transformers.pipeline(...)`) with a quantized model.
        cfg (object): Configuration object containing generation parameters:
            - batch_size (int): Number of prompts per batch.
            - max_tokens (int): Maximum number of new tokens to generate.
            - temperature (float): Sampling temperature.
            - top_p (float): Nucleus sampling parameter.
            - seed (int): Random seed for reproducibility.
            - do_sample (bool): Whether to use sampling or greedy decoding.

    Returns:
        list of str: A list of generated responses, one for each input prompt.
    """
    if isinstance(prompts, str): 
        prompts = [prompts]

    responses = []

    for i in tqdm(range(0, len(prompts), cfg.batch_size)): 
        batch = prompts[i:i + cfg.batch_size]
        outputs = pipe(
            batch, 
            max_new_tokens = cfg.max_tokens, 
            temperature = cfg.temperature, 
            top_p = cfg.top_p, 
            seed = cfg.seed,
            do_sample = cfg.do_sample, 
            eos_token_id = pipe.tokenizer.eos_token_id,
            pad_token_id = pipe.tokenizer.eos_token_id
        )
        response = [out[0]['generated_text'].strip() for out in outputs]
        responses.extend(response)
    
    return responses