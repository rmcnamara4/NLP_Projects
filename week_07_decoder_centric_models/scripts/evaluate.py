import hydra 
from omegaconf import DictConfig
import torch 

from src.evaluation.generation import * 
from src.evaluation.metrics import * 
from src.utils.save import * 
from src.data.dataset import SummarizationDataModule

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
    tokenizer.pad_token = tokenizer.eos_token

    data_module = SummarizationDataModule(cfg.datamodule, tokenizer = tokenizer)
    data_module.setup(stage = 'test') 
    test_dataloader = data_module.test_dataloader()
    print('Data module instantiated!') 

    model_path = os.path.join(cfg.save_model.model_path, cfg.save_model.model_name) 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}')
    else: 
        model = torch.load(model_path) 
        model.to(device) 

    print('Model loaded!') 

    save_path = os.path.join(cfg.paths.log_dir, 'test')

    chunk_summaries = generate_summaries( 
        cfg = cfg.generation, 
        model = model, 
        dataloader = test_dataloader, 
        tokenizer = tokenizer, 
        device = device 
    )

    torch.save(os.path.join(save_path, 'chunk_summaries.pt'), chunk_summaries)
    print('Chunk summaries generated!') 

    final_summaries = resummarize_chunks( 
        cfg = cfg.generation, 
        all_preds = chunk_summaries,
        model = model, 
        tokenizer = tokenizer, 
        device = device 
    )

    print('Final summaries generated!') 

    references = pd.DataFrame(test_dataloader.dataset)[['article_id', 'reference']]
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



    



    



