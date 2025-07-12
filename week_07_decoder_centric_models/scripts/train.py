import hydra 
from omegaconf import DictConfig
from hydra.utils import instantiate

from pytorch_lightning import Trainer 

import warnings
warnings.filterwarnings('ignore', message = 'pkg_resources is deprecated')

from src.data.dataset import SummarizationDataModule
from src.models.summarizer import SummarizationModule
from transformers import AutoTokenizer

import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from seed import set_seed
import torch 

## DON'T FORGET TO SET SEED
@hydra.main(config_path = '../configs', config_name = 'config', version_base = '1.3')
def main(cfg: DictConfig): 
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token 

    data_module = SummarizationDataModule(cfg.datamodule, tokenizer = tokenizer)

    print('Data module instantiated!')

    data_module.prepare_data()
    data_module.setup(stage = 'fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = SummarizationModule(cfg.model, tokenizer = tokenizer)
    callbacks = [instantiate(cb) for cb in cfg.callbacks]

    print('Model instantiated!') 

    trainer = Trainer(
        **cfg.trainer, 
        callbacks = callbacks
    )

    trainer.fit(model, datamodule = data_module)

    if cfg.save_model.save_model: 
        save_path = os.path.join(cfg.save_model.save_path, f'{cfg.save_model.model_name}.pt')
        torch.save(model.state_dict(), save_path) 
        print(f'Saved model state_dict to: {save_path}')

if __name__ == '__main__': 
    main()