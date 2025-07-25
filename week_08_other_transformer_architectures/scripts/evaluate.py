import hydra 
from omegaconf import DictConfig
import torch 

from src.evaluation.generation import * 
from src.evaluation.metrics import * 
from src.utils.save import * 
from src.data.dataset import PegasusDataModule
from src.models.summarizer import PegasusSummarizationModule

from torch.utils.data import DataLoader, Subset
from src.data.collators import TestCollator

import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from seed import set_seed

from transformers import AutoTokenizer

import pandas as pd

@hydra.main(config_path = '../configs', config_name = 'config', version_base = '1.3') 
def main(cfg: DictConfig):
    set_seed(cfg.seed) 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    data_module = PegasusDataModule(cfg.datamodule, tokenizer = tokenizer)
    data_module.setup(stage = 'test') 
    test_dataloader = data_module.test_dataloader()

    print('Data module instantiated!') 

    summarization_module = PegasusSummarizationModule(cfg.model, cfg.optimizer, cfg.scheduler, tokenizer = tokenizer)

    model_path = os.path.join(cfg.save_model.save_path, cfg.save_model.model_name + '.pt')  
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}')
    else: 
        state_dict = torch.load(model_path, map_location = device)
        summarization_module.load_state_dict(state_dict, strict = True) 
        model = summarization_module.model 
        model.to(device) 
        model.eval()

    print('Model loaded!') 

    save_path = os.path.join(cfg.paths.log_dir, 'test') 
    os.makedirs(save_path, exist_ok = True) 

    if os.path.exists(os.path.join(save_path, 'chunk_summaries.pt')): 
        chunk_summaries = torch.load(os.path.join(save_path, 'chunk_summaries.pt'))
        print('Chunk summaries loaded from disk!')
    else: 
        chunk_summaries = generate_summaries(
            cfg = cfg._generation_dict['chunk_generation'], 
            model = model, 
            dataloader = test_dataloader, 
            tokenizer = tokenizer, 
            device = device
        )
        torch.save(chunk_summaries, os.path.join(save_path, 'chunk_summaries.pt'))
        print('Chunk summaries generated!')

    final_summaries = resummarize_chunks(
        cfg = cfg._generation_dict['final_generation'], 
        all_preds = chunk_summaries, 
        model = model, 
        tokenizer = tokenizer, 
        device = device
    )

    print('Final summaries generated!')

    references = pd.DataFrame(data_module.test_dataset)[['article_id', 'reference']]
    references = references.drop_duplicates(subset = 'article_id')
    ref_dict = references.set_index('article_id')['reference'].to_dict()

    scores = compute_rouge_scores(
        predictions_dict = final_summaries, 
        references_dict = ref_dict
    )

    print('ROUGE scores computed!')

    save_results(scores, save_path) 
    save_predictions(final_summaries, ref_dict, save_path) 

    print('Results saved!')

if __name__ == '__main__':
    main()

