import torch 
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from omegaconf import OmegaConf

from dotenv import load_dotenv
load_dotenv()

import os
hf_token = os.getenv('HUGGINGFACE_TOKEN') 

from huggingface_hub import login 
login(hf_token)

import hydra 
from omegaconf import DictConfig, ListConfig

from src.models.summarizer import PegasusSummarizationModule

import logging
logging.getLogger().setLevel(logging.ERROR)

from dotmap import DotMap

@hydra.main(config_path='../configs/save_model', config_name='to_huggingface.yaml', version_base='1.3')
def main(cfg: DictConfig):
    # Load the Hydra-generated training config
    config_path = os.path.join('outputs', cfg.model_name, '.hydra', 'config.yaml')
    training_cfg = OmegaConf.load(config_path)

    # Convert each sub-config to dict then wrap in DotMap for dot-access
    model_cfg = DotMap(OmegaConf.to_container(training_cfg.model, resolve=True))
    lora_cfg = DotMap(OmegaConf.to_container(training_cfg.lora, resolve=True))
    optimizer_cfg = DotMap(OmegaConf.to_container(training_cfg.optimizer, resolve=True))
    scheduler_cfg = DotMap(OmegaConf.to_container(training_cfg.scheduler, resolve=True))

    tokenizer = PegasusTokenizer.from_pretrained(model_cfg.model_name)

    # Load your module and weights
    summarization_module = PegasusSummarizationModule(
        model_cfg, lora_cfg, optimizer_cfg, scheduler_cfg, tokenizer=tokenizer
    )

    model_path = os.path.join('models', cfg.model_name + '.pt')
    state_dict = torch.load(model_path, map_location='cpu')
    summarization_module.load_state_dict(state_dict)

    model = summarization_module.model

    # Set save directory
    save_directory = os.path.join(cfg.save_dir, cfg.model_name)

    # Save model & tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    # Push to Hugging Face Hub (optional)
    if cfg.get('push_to_hub', False):
        model.push_to_hub(cfg.repo_name)
        tokenizer.push_to_hub(cfg.repo_name)

if __name__ == '__main__':
    main()