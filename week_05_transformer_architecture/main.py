from src.data.dataset import TranslationDataset, collate_fn 
from src.data.preprocessing import numericalize, tokenize, build_vocab

from src.model.transformer import TransformerModel 

from src.utils.config import load_config
from src.utils.logging import setup_logging

import os

def main(): 
    config = load_config('./src/config.yaml')
    setup_logging(log_file = config['paths']['train_log'])

    src_lang = config['dataset']['src_lang']
    tgt_lang = config['dataset']['tgt_lang']
    
    

if __name__ == '__main__':
    main()