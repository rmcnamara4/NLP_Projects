import openai 
from tqdm import tqdm 

def get_openai_response(prompts, client, model_cfg, generation_cfg): 
    if isinstance(prompts, str): 
        prompts = [prompts]

    responses = []
    for prompt in tqdm(prompts, desc = f'Generating with {model_cfg.model_name}'): 
        messages = []
        if generation_cfg.system_prompt: 
            messages.append({'role': 'system', 'content': generation_cfg.system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        try: 
            response = client.chat.completions.create(
                model = model_cfg.model_name, 
                messages = messages, 
                max_tokens = generation_cfg.max_tokens, 
                temperature = generation_cfg.temperature, 
                top_p = generation_cfg.top_p
            )
            content = response.choices[0].message.content.strip()
            # content = reponse['choices'][0]['message']['content'].strip()
        except Exception as e: 
            content = f'[ERROR] {e}'

        responses.append(content)

        if generation_cfg.sleep_between_calls > 0: 
            import time 
            time.sleep(generation_cfg.sleep_between_calls) 
        
    return responses 
        

