import openai 
from tqdm import tqdm 

def get_openai_response(prompts, client, cfg): 
    if isinstance(prompts, str): 
        prompts = [prompts]

    responses = []
    for prompt in tqdm(prompts, desc = f'Generating with {cfg.model}'): 
        messages = []
        if cfg.system_prompt: 
            messages.append({'role': 'system', 'content': cfg.system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        try: 
            reponse = client.chat.completions.create(
                model = cfg.model_name, 
                messages = messages, 
                max_tokens = cfg.max_tokens, 
                temperature = cfg.temperature, 
                top_p = cfg.top_p, 
                do_sample = cfg.do_sample,
                seed = cfg.seed
            )
            content = reponse['choices'][0]['message']['content'].strip()
        except Exception as e: 
            content = f'[ERROR] {e}'

        responses.append(content)

        if cfg.sleep_between_calls > 0: 
            import time 
            time.sleep(cfg.sleep_between_calls) 
        
        return responses 
        

