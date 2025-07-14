import pytorch_lightning as pl
from datasets import load_dataset, DatasetDict 
from torch.utils.data import DataLoader
from src.data.collators import TrainCollator, TestCollator 
from src.data.preprocessing import train_preprocess, test_preprocess

class SummarizationDataModule(pl.LightningDataModule): 
  def __init__(self, cfg, tokenizer): 
    """
    PyTorch Lightning DataModule for preparing and loading the PubMed scientific papers dataset
    for summarization tasks.

    This module handles:
    - Loading and preprocessing the dataset using custom chunking and formatting logic.
    - Tokenizing input articles and abstracts with configurable chunking and length limits.
    - Creating train, validation, and test DataLoaders with appropriate collators.
    - Supporting efficient data loading with multi-worker prefetching.

    Args:
        cfg: Hydra configuration object containing preprocessing and dataloader parameters.
        tokenizer: Hugging Face tokenizer for encoding input and output text.
    """
    super().__init__()
    self.tokenizer = tokenizer
    self.train_batch_size = cfg.train_batch_size
    self.test_batch_size = cfg.test_batch_size
    self.chunk_len = cfg.chunk_len
    self.stride = cfg.stride
    self.min_len = cfg.min_len
    self.max_len = cfg.max_len
    self.num_workers = cfg.num_workers 
    self.prefetch_factor = cfg.prefetch_factor
    self.split_sizes = cfg.split_sizes
    self.padding_value = cfg.padding_value
    self.collate_fn = TrainCollator(tokenizer, self.padding_value)
    self.test_collate_fn = TestCollator(tokenizer)
    self.seed = cfg.seed

  def prepare_data(self): 
    """
    Downloads the PubMed subset of the 'scientific_papers' dataset from Hugging Face datasets.
    This is done only once and is safe to call in multi-GPU environments.
    """
    load_dataset('scientific_papers', 'pubmed')

  def setup(self, stage = None): 
    """
    Tokenizes and splits the dataset into train/validation/test sets depending on the stage.

    Args:
        stage (str, optional): Either 'fit', 'test', or None. Determines which splits to load
                                and preprocess.
    """
    if stage == 'fit' or stage is None: 
      train_data, val_data = load_dataset('scientific_papers', 'pubmed', split = ['train', 'validation'])
      dataset = DatasetDict({
          'train': train_data, 
          'validation': val_data
      })

      dataset['train'] = dataset['train'].shuffle(seed = self.seed).select(range(self.split_sizes[0]))
      dataset['validation'] = dataset['validation'].select(range(self.split_sizes[1]))

      tokenized_dataset = dataset.map(
          train_preprocess, 
          batched = True, 
          batch_size = self.train_batch_size, 
          remove_columns = dataset['train'].column_names, 
          load_from_cache_file = False,
          fn_kwargs = {
              'tokenizer': self.tokenizer, 
              'chunk_len': self.chunk_len, 
              'stride': self.stride, 
              'min_len': self.min_len, 
              'max_len': self.max_len
          }
      )

      self.train_dataset = tokenized_dataset['train']
      self.val_dataset = tokenized_dataset['validation']

    elif stage == 'test': 
      test_data = load_dataset('scientific_papers', 'pubmed', split = 'test')
      dataset = DatasetDict({
          'test': test_data
      })

      dataset['test'] = dataset['test'].select(range(self.split_sizes[2]))

      tokenized_dataset = dataset.map(
          test_preprocess, 
          batched = True, 
          with_indices = True, 
          batch_size = self.test_batch_size, 
          remove_columns = dataset['test'].column_names, 
          load_from_cache_file = False,
          fn_kwargs = {
              'tokenizer': self.tokenizer, 
              'chunk_len': self.chunk_len, 
              'stride': self.stride, 
              'min_len': self.min_len, 
              'max_len': self.max_len
          }
      )

      self.test_dataset = tokenized_dataset['test']

  def train_dataloader(self): 
    """
    Returns the DataLoader for the training dataset with shuffling and multi-worker support.
    """
    return DataLoader(
        self.train_dataset, 
        shuffle = True, 
        batch_size = self.train_batch_size, 
        collate_fn = self.collate_fn, 
        num_workers = self.num_workers, 
        prefetch_factor = self.prefetch_factor
    )

  def val_dataloader(self): 
    """
    Returns the DataLoader for the validation dataset without shuffling.
    """
    return DataLoader( 
        self.val_dataset, 
        shuffle = False, 
        batch_size = self.train_batch_size, 
        collate_fn = self.collate_fn, 
        num_workers = self.num_workers, 
        prefetch_factor = self.prefetch_factor
    )

  def test_dataloader(self): 
    """
    Returns the DataLoader for the test dataset without shuffling and using the test collator.
    """
    return DataLoader(
        self.test_dataset, 
        shuffle = False, 
        batch_size = self.test_batch_size, 
        collate_fn = self.test_collate_fn, 
        num_workers = self.num_workers, 
        prefetch_factor = self.prefetch_factor
    )