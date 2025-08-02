import torch 
from tqdm import tqdm

def get_quantized_response(prompts, pipe, cfg): 
    if isinstance(prompts, str): 
        prompts = [prompts]

    responses = []

    for i in tqdm(range(0, len(prompts), cfg.batch_size)): 
        batch = prompts[i:i + cfg.batch_size]
        outputs = pipe(
            batch, 
            max_tokens = cfg.max_tokens, 
            temperature = cfg.temperature, 
            top_p = cfg.top_p, 
            seed = cfg.seed,
            do_sample = cfg.do_sample
        )
        response = [out[0]['generated_text'].strip() for out in outputs]
        responses.extend(response)
    
    return responses