import hydra 
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call 

from pytorch_lightning import Trainer 

import warnings
warnings.filterwarnings('ignore', message = 'pkg_resources is deprecated')

from transformers import AutoTokenizer

## DON'T FORGET TO SET SEED
@hydra.main(config_path = 'configs', config_name = 'config', version_base = '1.3')
def main(cfg: DictConfig): 
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token 

    data_module = instantiate(cfg.datamodule) 

    data_module.prepare_data()
    data_module.setup(stage = 'fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = instantiate(cfg.model, tokenizer = tokenizer) 
    callbacks = [instantiate(cb) for cb in cfg.callbacks]

    trainer = Trainer(
        **cfg.trainer, 
        callbacks = callbacks
    )

    trainer.fit(model, datamodule = data_module)

if __name__ == '__main__': 
    main()