import pytest 
import torch 
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset 

from src.model.classifier import DistilBERTClassifier
from src.train.trainer import Trainer

import torch.nn as nn 
import torch.optim as optim 

class TinyTextDataset(Dataset): 
    def __init__(self, tokenizer, texts, labels, max_length = 32): 
        self.encoding = tokenizer(texts, padding = True, truncation = True, max_length = max_length, return_tensors = 'pt') 
        self.labels = torch.tensor(labels) 

    def __getitem__(self, idx): 
        return {
            'input_ids': self.encoding['input_ids'][idx], 
            'attention_mask': self.encoding['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def __len__(self): 
        return len(self.labels) 
    
@pytest.fixture(scope = 'session')
def tokenizer(): 
    return AutoTokenizer.from_pretrained('distilbert-base-uncased')

@pytest.fixture(scope = 'session')
def dummy_dataset(tokenizer): 
    texts = [
        'This is a good comment.', 
        'This is a toxic comment.', 
        'Neutral text.', 
        'Some very bad words in here.', 
        'Helpful and kind message.'
    ]
    labels = [0, 1, 0, 1, 0]
    return TinyTextDataset(tokenizer, texts, labels) 

@pytest.fixture(scope = 'session') 
def dummy_dataloader(dummy_dataset): 
    return DataLoader(dummy_dataset, batch_size = 2, shuffle = False, collate_fn = DataCollatorWithPadding)

@pytest.fixture(scope = 'session')
def model(): 
    return DistilBERTClassifier(num_classes = 2, classifier_dim = 64, freeze_bert = True)


@pytest.fixture(scope="session")
def optimizer(model):
    return optim.AdamW(model.parameters(), lr=1e-4)


@pytest.fixture(scope="session")
def criterion():
    return nn.CrossEntropyLoss()


@pytest.fixture
def dummy_trainer(model, optimizer, criterion, dummy_dataloader):
    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=dummy_dataloader,
        val_dataloader=dummy_dataloader,
        device="cpu"
    )