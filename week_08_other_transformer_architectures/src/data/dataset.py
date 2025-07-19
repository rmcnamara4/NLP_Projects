import pytorch_lightning as pl
from datasets import load_dataset, DatasetDict

from src.data.collators import StripFieldsCollator, DataCollatorWithID
from transformers import DataCollatorForSeq2Seq 

from torch.data.utils import DataLoader

class PegasusDataModule(pl.LightningDataModule):
  def __init__(self, cfg, tokenizer, embedding_model = None):
    """
    A PyTorch Lightning DataModule for hierarchical summarization of the PubMed scientific papers dataset.

    This class handles loading, preprocessing, chunking, and batching of the dataset for training, validation, 
    and testing phases. It supports long document tokenization with overlapping chunks and flexible DataLoader 
    configuration, including custom collators for different phases.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer used for encoding input and output text.
        batch_size (int): Batch size used for training, validation, and testing.
        model (PreTrainedModel, optional): The model used by collators (for padding settings, etc.).
        chunk_len (int): Length of each input chunk in tokens.
        stride (int): Number of overlapping tokens between chunks.
        min_len (int): Minimum chunk length required to keep a chunk.
        max_len (int): Maximum number of tokens for input truncation.
        num_workers (int): Number of subprocesses to use for data loading.
        prefetch_factor (int): Number of batches to prefetch per worker.
        split_sizes (tuple): Number of examples to use for (train, validation, test) splits.
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
    self.seed = cfg.seed 
    self.embedding_model = embedding_model
    self.chunking_strategy = cfg.chunking_strategy
    self.num_keep = cfg.num_keep

  def prepare_data(self):
    """
    Download the PubMed split of the 'scientific_papers' dataset from HuggingFace.
    This method is only called once per machine.
    """
    load_dataset('scientific_papers', 'pubmed')

  def setup(self, stage = None):
    """
    Set up datasets and preprocessing for the appropriate stage ('fit' or 'test').

    Args:
        stage (str, optional): One of 'fit' (train/val) or 'test'. If None, defaults to setting up all splits.
    """
    if stage == 'fit' or stage is None:
      train_data, val_data = load_dataset('scientific_papers', 'pubmed', split = ['train', 'validation'])
      dataset = DatasetDict({
          'train': train_data,
          'validation': val_data
      })

      dataset['train'] = dataset['train'].shuffle(seed = 24).select(range(self.split_sizes[0]))
      dataset['validation'] = dataset['validation'].select(range(self.split_sizes[1]))

      tokenized_dataset = dataset.map(
          preprocess,
          batched = True,
          batch_size = 32,
          with_indices = True,
          remove_columns = dataset['train'].column_names ,
          fn_kwargs = {
              'tokenizer': self.tokenizer,
              'chunk_len': self.chunk_len,
              'stride': self.stride,
              'min_len': self.min_len,
              'max_len': self.max_len, 
              'num_keep': self.num_keep,
              'train': True, 
              'chunking_strategy': self.chunking_strategy,
              'embedding_model': self.embedding_model
          }
      )

      self.train_dataset = tokenized_dataset['train']
      self.val_dataset = tokenized_dataset['validation']

      self.train_collate_fn = StripFieldsCollator(DataCollatorForSeq2Seq(
          self.tokenizer, 
          self.model, 
          padding = 'longest', 
          return_tensors = 'pt', 
          max_length = self.tokenizer.model_max_length
      ), allowed_fields = ['input_ids', 'attention_mask', 'labels'])

    elif stage == 'test':
      test_data = load_dataset('scientific_papers', 'pubmed', split = 'test')
      dataset = DatasetDict({
          'test': test_data
      })

      dataset['test'] = dataset['test'].select(range(self.split_sizes[2]))

      tokenized_dataset = dataset.map(
          preprocess,
          batched = True,
          batch_size = 32,
          with_indices = True,
          remove_columns = dataset['test'].column_names ,
          fn_kwargs = {
              'tokenizer': self.tokenizer,
              'chunk_len': self.chunk_len,
              'stride': self.stride,
              'min_len': self.min_len,
              'max_len': self.max_len, 
              'num_keep': self.num_keep,
              'train': False, 
              'chunking_strategy': self.chunking_strategy,
              'embedding_model': self.embedding_model
          }
      )

      self.test_dataset = tokenized_dataset['test']

      self.test_collate_fn = StripFieldsCollator(DataCollatorWithID(
          self.tokenizer, 
          self.model, 
          padding = 'longest', 
          return_tensors = 'pt', 
          max_length = self.tokenizer.model_max_length
      ), allowed_fields = ['input_ids', 'attention_mask', 'article_id'])

  def train_dataloader(self):
    """
    Returns:
        DataLoader: DataLoader for the training set.
    """
    return DataLoader(
        self.train_dataset,
        shuffle = True,
        batch_size = self.batch_size,
        collate_fn = self.train_collate_fn,
        num_workers = self.num_workers,
        prefetch_factor = self.prefetch_factor
    )

  def val_dataloader(self):
    """
    Returns:
        DataLoader: DataLoader for the validation set.
    """
    return DataLoader(
        self.val_dataset,
        shuffle = False,
        batch_size = self.batch_size,
        collate_fn = self.train_collate_fn,
        num_workers = self.num_workers,
        prefetch_factor = self.prefetch_factor
    )

  def test_dataloader(self):
    """
    Returns:
        DataLoader: DataLoader for the test set.
    """
    return DataLoader(
        self.test_dataset,
        shuffle = False,
        batch_size = self.batch_size,
        collate_fn = self.test_collate_fn,
        num_workers = self.num_workers,
        prefetch_factor = self.prefetch_factor
    )