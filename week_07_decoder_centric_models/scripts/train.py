import sys 
import os 

import warnings
warnings.filterwarnings('ignore', message = 'pkg_resources is deprecated')

from src.data.dataset import SummarizationDataModule
from transformers import GPT2Tokenizer

def main(): 
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 

    data_module = SummarizationDataModule(
        tokenizer = tokenizer, 
        batch_size = 2, 
        chunk_len = 512,
        stride = 450, 
        min_len = 256, 
        max_len = 1024, 
        num_workers = 5, 
        prefetch_factor = 2, 
        split_sizes = (500, 200, 200), 
        padding_value = -100
    )

    data_module.prepare_data()
    data_module.setup(stage = 'fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(next(iter(train_loader)))

if __name__ == '__main__': 
    main()