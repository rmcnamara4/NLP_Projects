import openai 
from tqdm import tqdm 

def get_openai_response(prompts, client, model_cfg, generation_cfg): 
    """
    Generates responses from an OpenAI chat model given one or more prompts.

    This function sends each prompt to the OpenAI API using the specified model and generation configuration.
    It supports optional system prompts and handles rate-limiting with sleep intervals.

    Args:
        prompts (str or list of str): Single prompt or list of prompts to send to the model.
        client (openai.Client): The OpenAI client instance used to call the API.
        model_cfg (object): An object with a `model_name` attribute specifying which model to use (e.g., 'gpt-4').
        generation_cfg (object): An object containing generation parameters:
            - system_prompt (str): Optional system-level prompt.
            - max_tokens (int): Maximum number of tokens in the response.
            - temperature (float): Sampling temperature.
            - top_p (float): Nucleus sampling probability.
            - sleep_between_calls (float): Time to sleep (in seconds) between API calls to avoid rate limits.

    Returns:
        list of str: A list of generated responses (or error messages if an exception occurs).
    """
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
        

