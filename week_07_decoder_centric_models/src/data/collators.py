from torch.nn.utils.rnn import pad_sequence
import torch 

class TrainCollator:
    """
    Collator class for dynamic padding during language model training.

    This collator is designed for sequence-to-sequence or causal language modeling tasks where input IDs,
    attention masks, and labels are of variable lengths. It pads each to the max length in the batch:
    - `input_ids` are padded with the tokenizer's `pad_token_id`
    - `attention_mask` is padded with 0
    - `labels` are padded with a configurable padding value (default -100) so they are ignored in the loss

    Args:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer used for tokenization and padding.
        padding_value (int, optional): Value to pad the labels with. Default is -100.
    """
    def __init__(self, tokenizer, padding_value = -100):
        self.tokenizer = tokenizer
        self.padding_value = padding_value

    def __call__(self, features):
        """
        Collate a batch of examples by padding `input_ids`, `attention_mask`, and `labels`.

        Args:
            features (List[Dict]): A list of examples, each a dictionary with keys:
                - 'input_ids': List[int]
                - 'attention_mask': List[int]
                - 'labels': List[int]

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - input_ids: Padded tensor of input IDs
                - attention_mask: Padded tensor of attention masks
                - labels: Padded tensor of labels (with padding_value)
        """
        input_ids = [torch.tensor(f['input_ids']) for f in features]
        attention_mask = [torch.tensor(f['attention_mask']) for f in features]
        labels = [torch.tensor(f['labels']) for f in features]

        input_ids = pad_sequence(input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first = True, padding_value = 0)
        labels = pad_sequence(labels, batch_first = True, padding_value = self.padding_value)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
class TestCollator:
    """
    Collator class for padding input sequences during inference or evaluation.

    This collator is used to batch and pad examples for summarization or language generation tasks.
    It pads `input_ids` and `attention_mask` using the tokenizer's `pad` method (left-padding enforced),
    while preserving any additional metadata fields like `article_id` and `reference`.

    Args:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer for padding input sequences.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """
        Collate a batch of inference/evaluation examples.

        Args:
            batch (List[Dict]): A list of dictionaries, each with:
                - 'input_ids': List[int]
                - 'attention_mask': List[int]
                - Optional: 'article_id', 'reference'

        Returns:
            Dict[str, Union[torch.Tensor, List[Any]]]: A dictionary with:
                - input_ids: Padded input ID tensor
                - attention_mask: Padded attention mask tensor
                - article_id: List of article IDs (if present)
                - references: List of reference summaries (if present)
        """
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        # Pad input sequences
        self.tokenizer.padding_side = 'left'
        padded = self.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            padding = True,
            return_tensors = 'pt'
        )

        # Keep non-padded fields as-is
        article_ids = [item.get('article_id') for item in batch]
        references = [item.get('reference') for item in batch]

        padded['article_id'] = article_ids
        padded['references'] = references

        return padded