import torch 
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from omegaconf import OmegaConf

from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN') 

from huggingface_hub import login 
login(hf_token)

import hydra 
from omegaconf import DictConfig

@hydra.main(config_path = '../configs/save_model', config_name = 'to_huggingface.yaml')
def main(cfg: DictConfig): 
    model = PegasusForConditionalGeneration.from_pretrained(cfg._base_model) 
    tokenizer = PegasusTokenizer.from_pretrained(cfg._base_model) 

    state_dict = torch.load(cfg.model_path, map_location = 'cpu')
    model.load_state_dict(state_dict) 

    model.save_pretrained(cfg.save_directory) 
    tokenizer.save_pretrained(cfg.save_directory)

    if cfg.get('push_to_hub', False): 
        model.push_to_hub(cfg.repo_name) 
        tokenizer.push_to_hub(cfg.repo_name) 

if __name__ == '__main__': 
    main()