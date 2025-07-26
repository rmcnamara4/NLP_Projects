import hydra 
from omegaconf import DictConfig
from hydra.utils import instantiate

from pytorch_lightning import Trainer 

import warnings 
warnings.filterwarnings('ignore', message = 'pkg_resources is deprecated')
warnings.filterwarnings('ignore', category = DeprecationWarning)

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
    
    print('Data module instantiated!')

    model = PegasusSummarizationModule(
        model_cfg = cfg.model, 
        lora_cfg = cfg.lora, 
        optimizer_cfg = cfg.optimizer, 
        scheduler_cfg = cfg.scheduler, 
        tokenizer = tokenizer
    )
    callbacks = [instantiate(cb) for cb in cfg._callback_dict.values()]
    data_module.model = model.model

    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, 'best_checkpoint.ckpt')
    resume_from_checkpoint = ckpt_path if cfg.resume and os.path.exists(ckpt_path) else None 

    if resume_from_checkpoint: 
        print(f'Resuming training from {resume_from_checkpoint}')
    else: 
        print('Starting training from scratch.')

    print('Model instantiated!')

    trainer = Trainer(
        **cfg.trainer, 
        callbacks = callbacks, 
        logger = logger
    )

    trainer.fit(model, datamodule = data_module, ckpt_path = resume_from_checkpoint)

    if cfg.save_model.save_model: 
        os.makedirs(cfg.save_model.save_path, exist_ok = True) 
        save_path = os.path.join(cfg.save_model.save_path, f'{cfg.save_model.model_name}.pt')
        torch.save(model.state_dict(), save_path) 
        print(f'Saved model state_dict to: {save_path}')


if __name__ == '__main__': 
    try:
        main()
    except Exception as e:
        print(f"❌ Training script failed with error: {e}")
    finally:
        print("⚠️ Triggering VM shutdown...")
        os.system("sudo shutdown -h now")