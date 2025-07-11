from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import hydra 
from transformers import GPT2Config, GPT2LMHeadModel
import pytorch_lightning as pl 
class SummarizationModule(pl.LightningModule): 
  def __init__(self, model_cfg, tokenizer): 
    super().__init__()
    self.save_hyperparameters(ignore = ['tokenizer'])

    config = GPT2Config.from_pretrained(
        model_cfg.model_name,
        attn_pdrop = model_cfg.attn_pdrop,
        resid_pdrop = model_cfg.resid_pdrop,
        embd_pdrop = model_cfg.embd_pdrop
    )

    self.model = GPT2LMHeadModel.from_pretrained(model_cfg.model_name, config = config)
    self.model.resize_token_embeddings(len(tokenizer))
    self.model.config.pad_token_id = tokenizer.pad_token_id

    self.gradual_unfreezing = model_cfg.get('gradual_unfreezing', False) 
    
    if self.gradual_unfreezing: 
        for param in self.model.transformer.parameters(): 
            param.requires_grad = False 
        self.current_stage = 0
    else: 
        for param in self.model.transformer.parameters(): 
            param.requires_grad = True 
        self.current_stage = len(self.model.transformer.h)

    for param in self.model.lm_head.parameters(): 
      param.requires_grad = True 

  def forward(self, input_ids, attention_mask, labels = None): 
    output = self.model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
    return output

  def training_step(self, batch, batch_idx):
      outputs = self(**batch)
      loss = outputs.loss
      num_tokens = (batch['labels'] != -100).sum()

      self.log(
        'train_loss_per_token', 
        loss.detach(), 
        prog_bar = True, 
        on_step = True, 
        on_epoch = False, 
        sync_dist = True,
        logger = True
      )

      self.log(
        'train_total_loss', 
        (loss.detach() * num_tokens).float(), 
        prog_bar = False, 
        on_step = False, 
        on_epoch = True, 
        reduce_fx = 'sum', 
        sync_dist = True, 
        logger = False
      )

      self.log(
        'train_total_tokens', 
        num_tokens.float(), 
        prog_bar = False, 
        on_step = False, 
        on_epoch = True, 
        reduce_fx = 'sum', 
        sync_dist = True, 
        logger = False
      )

      return loss.item()

  def validation_step(self, batch, batch_idx):
      outputs = self(**batch)
      loss = outputs.loss
      num_tokens = (batch["labels"] != -100).sum()

      self.log(
        'val_total_loss', 
        (loss.detach() * num_tokens).float(), 
        on_step = False, 
        on_epoch = True, 
        reduce_fx = 'sum', 
        sync_dist = True, 
        logger = False
      )

      self.log(
        'val_token_tokens', 
        num_tokens.float(), 
        on_step = False, 
        on_epoch = True, 
        reduce_fx = 'sum', 
        sync_dist = True, 
        logger = False
      )

      return loss.item()

  def configure_optimizers(self):
    optimizer: Optimizer = hydra.utils.instantiate(
      self.hparams.optimizer, params = self.parameters()
    )

    scheduler_cfg = self.hparams.get('scheduler') 
    if scheduler_cfg is not None: 
      scheduler: _LRScheduler = hydra.utils.instantiate(
        scheduler_cfg, optimizer = optimizer 
      )
  
    return {
      'optimizer': optimizer, 
      'lr_scheduler': {
        'scheduler': scheduler, 
        'interval': 'epoch'
      }
    }

  def on_train_epoch_start(self): 
    self.print_trainable_params()

  def on_train_epoch_end(self):
    total_loss = self.trainer.callback_metrics['train_total_loss']
    total_tokens = self.trainer.callback_metrics['train_total_tokens']

    if total_loss is not None and total_tokens is not None: 
      train_loss = total_loss / total_tokens
      self.log(
        'train_loss', 
        train_loss, 
        prog_bar = True, 
        logger = True, 
        sync_dist = True
      )

    if self.gradual_unfreezing and self.current_epoch > 0 and self.current_epoch % 2 == 0: 
      self.unfreeze_next()

  def on_validation_epoch_end(self):
    total_loss = self.trainer.callback_metrics['val_total_loss']
    total_tokens = self.trainer.callback_metrics['val_total_tokens']
    if total_loss is not None and total_tokens is not None: 
      val_loss = total_loss / total_tokens
      self.log(
        'val_loss', 
        val_loss, 
        prog_bar = True, 
        logger = True, 
        sync_dist = True
      )

  def unfreeze_next(self): 
    n_layers = len(self.model.transformer.h) 
    if self.current_stage < n_layers: 
      for param in self.model.transformer.h[n_layers - 1 - self.current_stage].parameters(): 
        param.requires_grad = True 
      self.current_stage += 1

  def print_trainable_params(self): 
    total = sum(p.numel() for p in self.model.parameters())
    trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)')
    self.log(
      'trainable_parameter_ratio', 
      100 * trainable / total, on_epoch = True, logger = True
    )
