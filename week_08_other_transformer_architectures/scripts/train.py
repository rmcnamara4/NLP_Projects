import hydra 
from omegaconf import DictConfig
from hydra.utils import instantiate

from pytorch_lightning import Trainer 

import warnings 
warnings.filterwarnings('ignore', message = 'pkg_resources is deprecated')

from src.data.dataset import PegasusDataModule 
from src.models.summarizer import PegasusSummarizationModule
from transformers import PegasusTokenizer

import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from seed import set_seed 
import torch 

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import shutil
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@hydra.main(config_path = '../configs', config_name = 'config', version_base = '1.3') 
def main(cfg: DictConfig): 
    set_seed(cfg.seed) 

    logger = CSVLogger( 
        save_dir = cfg.paths.log_dir, 
        name = 'train', 
        version = ''
    )

    if not cfg.resume and os.path.exists(cfg.paths.log_dir): 
        shutil.rmtree(cfg.paths.log_dir)

    tokenizer = PegasusTokenizer.from_pretrained(cfg.model.model_name)

    data_module = PegasusDataModule(cfg.datamodule, tokenizer = tokenizer)

    model = PegasusSummarizationModule(
        model_cfg = cfg.model, 
        lora_cfg = cfg.lora, 
        optimizer_cfg = cfg.optimizer, 
        scheduler_cfg = cfg.scheduler, 
        tokenizer = tokenizer
    )

    data_module.model = model.model

    data_module.prepare_data()  # Ensure data is downloaded and processed
    data_module.setup(stage = 'fit')

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    print(next(iter(train_dataloader)))

if __name__ == '__main__': 
    main() 