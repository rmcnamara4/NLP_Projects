import hydra 
from omegaconf import DictConfig

import json

from src.evaluation.evaluate import calculate_accuracy
from src.evaluation.save import save_results
from src.models.openai import get_openai_response
from src.models.quantized import get_quantized_response
from src.utils.create_prompts import create_prompts
from src.utils.extract import extract_final_answer
from src.utils.load_data import load_data
from src.utils.load_model import load_quantized_model

import sys 
import os 
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
openai_api_key = os.getenv('OPENAI_API_KEY') 
hf_token = os.getenv('HUGGINGFACE_TOKEN') 

import openai 
from openai import OpenAI
openai.api_key = openai_api_key

from huggingface_hub import login 
login(hf_token)

from seed import set_seed

from jinja2 import Environment, FileSystemLoader


@hydra.main(config_path = '../configs', config_name = 'config', version_base = '1.3')
def main(cfg: DictConfig): 
    set_seed(cfg.seed) 

    model_type = cfg.model.model_type.lower()

    dataset = load_data(cfg.data)
    
    if model_type == 'huggingface': 
        pipe = load_quantized_model(cfg.model) 

    prompt_variables = [{'question': item['question']} for item in dataset]
    answers = [extract_final_answer(item['answer']) for item in dataset]

    env = Environment(loader = FileSystemLoader(cfg.prompt.prompt_dir))
    template = env.get_template(cfg.prompt.template_file)

    prompts = create_prompts(prompt_variables, template) 

    if model_type == 'huggingface': 
        responses = get_quantized_response(prompts, pipe, cfg.generation) 
    elif model_type == 'openai': 
        client = OpenAI(api_key = openai.api_key)
        responses = get_openai_response(prompts, client, cfg.model, cfg.generation) 

    predictions = [extract_final_answer(g) for g in responses]
    predictions = [float(p) if p else None for p in predictions]

    references = [float(a) for a in answers]

    accuracy, correct_vec = calculate_accuracy(predictions, references) 

    save_results(
        cfg.model.model_name,
        cfg.prompt.prompt_version, 
        [x['question'] for x in prompt_variables], 
        references, 
        responses, 
        predictions, 
        correct_vec, 
        os.path.join(cfg.paths.result_path, 'results.csv')
    )

    results = {
        'accuracy': accuracy
    }
    with open(os.path.join(cfg.paths.result_path, 'accuracy.json'), 'w') as f: 
        json.dump(results, f, indent = 4)


if __name__ == '__main__': 
    main()





