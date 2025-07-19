from transformers import PegasusConfig, PegasusForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
import pytorch_lightning as pl
import torch 
import hydra 
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class PegasusSummarizationModule(pl.LightningModule):
  """
  PyTorch Lightning module for abstractive summarization using the Pegasus model.

  This module supports both full fine-tuning and parameter-efficient fine-tuning with LoRA,
  along with optional gradual unfreezing of transformer layers during training.

  Args:
      model_cfg (DictConfig): Configuration for the Pegasus model (e.g., model name, dropout).
      lora_cfg (DictConfig): Configuration for LoRA (e.g., whether to use it, target modules, rank).
      optimizer_cfg (DictConfig): Optimizer configuration compatible with Hydra instantiation.
      scheduler_cfg (DictConfig): Learning rate scheduler configuration.
      tokenizer (PreTrainedTokenizer): Hugging Face tokenizer used for padding and decoding.
  """
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
      """
      Forward pass through the Pegasus model.

      Args:
          input_ids (torch.Tensor): Input token IDs.
          attention_mask (torch.Tensor): Attention mask indicating non-padding tokens.
          decoder_input_ids (torch.Tensor): Decoder input IDs for teacher forcing.
          labels (torch.Tensor, optional): Target token IDs for computing loss.

      Returns:
          Seq2SeqLMOutput: Output from the Pegasus model including loss and logits.
      """
      output = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, labels = labels)
      return output

  def training_step(self, batch, batch_idx):
      """
      Training step that computes and logs loss per token and total loss/token count.

      Args:
          batch (Dict[str, torch.Tensor]): Batch containing inputs and labels.
          batch_idx (int): Index of the current batch.

      Returns:
          torch.Tensor: Training loss for the batch.
      """
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

      return loss

  def validation_step(self, batch, batch_idx):
      """
      Validation step that computes and accumulates total validation loss and token count.

      Args:
          batch (Dict[str, torch.Tensor]): Batch containing inputs and labels.
          batch_idx (int): Index of the current batch.

      Returns:
          torch.Tensor: Validation loss for the batch.
      """
      outputs = self(**batch)
      loss = outputs.loss
      num_tokens = (batch['labels'] != -100).sum()

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
    """
    Configures and returns the optimizer and learning rate scheduler.

    Returns:
        Dict: Dictionary with optimizer and optional learning rate scheduler config.
    """
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
    """
    Logs trainable parameter count at the beginning of each training epoch.
    """
    self.print_trainable_params()

  def on_train_epoch_end(self):
    """
    Computes and logs average training loss at the end of the epoch.
    Also unfreezes the next transformer layer if gradual unfreezing is enabled.
    """
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
    """
    Computes and logs average validation loss at the end of the epoch.
    """
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
    Unfreezes the next highest transformer layer (top-down) if within allowed limit.

    This enables gradual unfreezing for stable training.
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
    """
    Prints the number of trainable vs total model parameters for transparency/debugging.
    """
    total = sum(p.numel() for p in self.model.parameters())
    trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)')
