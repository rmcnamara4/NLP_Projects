import hydra 
from omegaconf import DictConfig
from hydra.utils import instantiate

from pytorch_lightning import Trainer 

import warnings
warnings.filterwarnings('ignore', message = 'pkg_resources is deprecated')

from src.data.dataset import SummarizationDataModule
from src.models.summarizer import SummarizationModule
from transformers import AutoTokenizer

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import shutil
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

## DON'T FORGET TO SET SEED
@hydra.main(config_path = '../configs', config_name = 'config', version_base = '1.3')
def main(cfg: DictConfig): 
    logger = CSVLogger(
        save_dir = cfg.paths.log_dir, 
        name = 'train', 
        version = ''
    )

    if not cfg.resume and os.path.exists(cfg.paths.log_dir): 
        shutil.rmtree(cfg.paths.log_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token 

    data_module = SummarizationDataModule(cfg.datamodule, tokenizer = tokenizer)

    data_module.prepare_data()
    data_module.setup(stage = 'fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = SummarizationModule(cfg.model, cfg.optimizer, cfg.scheduler, tokenizer = tokenizer)
    callbacks = [instantiate(cb) for cb in cfg._callback_dict.values()]

    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, 'best_checkpoint.ckpt')
    resume_from_checkpoint = ckpt_path if cfg.resume and os.path.exists(ckpt_path) else None

    if resume_from_checkpoint:
        print(f'Resuming training from {resume_from_checkpoint}')
    else: 
        print('Starting training from scratch.') 

    trainer = Trainer(
        **cfg.trainer, 
        callbacks = callbacks, 
        logger = logger
    )

    trainer.fit(model, datamodule = data_module, ckpt_path = resume_from_checkpoint)

if __name__ == '__main__': 
    main()