from transformers import PegasusConfig, PegasusForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
import pytorch_lightning as pl
import torch 
import hydra 
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class PegasusSummarizationModule(pl.LightningModule):
  def __init__(self, model_cfg, lora_cfg, optimizer_cfg, scheduler_cfg, tokenizer):
      super().__init__()
      self.save_hyperparameters(ignore = ['tokenizer'])

      config = PegasusConfig.from_pretrained(
        model_cfg.model_name,
        dropout = model_cfg.dropout,  # global dropout
        attention_dropout = model_cfg.attention_dropout  # attention-specific
      )

      self.model = PegasusForConditionalGeneration.from_pretrained(model_cfg.model_name, config = config)
      self.model.config.pad_token_id = tokenizer.pad_token_id

      self.use_lora = lora_cfg.get('use_lora', False) 
      self.gradual_unfreezing = model_cfg.get('gradual_unfreeze', False)

      if lora_cfg.use_lora: 
          lora_config = LoraConfig( 
              r = lora_cfg.rank, 
              lora_alpha = lora_cfg.alpha, 
              target_modules = lora_cfg.target_modules, 
              lora_dropout = lora_cfg.dropout, 
              bias = lora_cfg.bias, 
              task_type = TaskType.SEQ_2_SEQ_LM
          )

          self.model = get_peft_model(self.model, lora_config) 

      else:
          for param in self.model.transformer.parameters(): 
              param.requires_grad = False

          if self.gradual_unfreezing: 
              self.current_stage = 0
          else: 
              self.current_stage = len(self.model.transformer.h)

          for param in self.model.lm_head.parameters():
              param.requires_grad = True


  def forward(self, input_ids, attention_mask, decoder_input_ids, labels = None):
      output = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, labels = labels)
      return output

  def training_step(self, batch, batch_idx):
      outputs = self(**batch)
      loss = outputs.loss
      num_tokens = (batch["labels"] != -100).sum()

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

      return loss

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
         'val_total_tokens', 
         num_tokens.float(),
         on_step = False,
         on_epoch = True,
         reduce_fx = 'sum',
         sync_dist = True,
         logger = False
      )

      return loss

  def configure_optimizers(self):
    optimizer: Optimizer = hydra.utils.instantiate(
       self.hparams.optimizer_cfg, 
       params = self.parameters()
    )

    scheduler_cfg = self.hparams.get('scheduler_cfg', None) 
    if scheduler_cfg is not None: 
        scheduler: _LRScheduler = hydra.utils.instantiate(
            scheduler_cfg, 
            optimizer = optimizer
        )
    else: 
        scheduler = None

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
    """
    Unfreezes the next transformer layer from top-down, limited by `max_unfrozen_layers`.
    """
    n_layers = len(self.model.transformer.h)

    # Calculate maximum allowed stages
    max_allowed = self.hparams.model_cfg.max_unfrozen_layers or n_layers

    if self.current_stage < min(max_allowed, n_layers):
        layer_idx = n_layers - 1 - self.current_stage  # Top-down
        for param in self.model.transformer.h[layer_idx].parameters():
            param.requires_grad = True
        self.current_stage += 1
        print(f'Unfroze layer {layer_idx}')
    else:
        print('Reached max number of unfrozen layers.')

  def print_trainable_params(self):
    total = sum(p.numel() for p in self.model.parameters())
    trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)')
