from transformers import DataCollatorForSeq2Seq

class StripFieldsCollator:
    """
    A wrapper around a base collator that filters out unwanted fields from input examples 
    before passing them to the actual collator.

    This is especially useful when you want to keep metadata (e.g., article IDs or references) 
    in the dataset but exclude them during collation to avoid errors from the underlying 
    tokenizer's collator (like DataCollatorWithPadding or DataCollatorForSeq2Seq).

    Args:
        base_collator (Callable): The data collator used to batch and pad input examples.
        allowed_fields (tuple): The keys to retain from each input example before passing 
                                to the base collator. Defaults to ('input_ids', 'attention_mask', 'labels').
    """
    def __init__(self, base_collator, allowed_fields = ('input_ids', 'attention_mask', 'labels')):
        self.base_collator = base_collator
        self.allowed_fields = allowed_fields

    def __call__(self, features):
        clean_features = [{k: f[k] for k in self.allowed_fields if k in f} for f in features]
        return self.base_collator(clean_features)

class DataCollatorWithID(DataCollatorForSeq2Seq):
    """
    Filters each feature in the batch to retain only the allowed fields, 
    and applies the base collator to the cleaned batch.

    Args:
        features (List[Dict[str, Any]]): A list of feature dictionaries from the dataset.

    Returns:
        Dict[str, torch.Tensor]: A batch dictionary returned by the base collator, 
        typically containing tensors like input_ids, attention_mask, and labels.
    """
    def __call__(self, features):
        # Extract article_ids before collating
        article_ids = [f['article_id'] for f in features]

        # Use the original collator to process input/label tensors
        batch = super().__call__(features)

        # Add back article_ids
        batch['article_id'] = article_ids
        return batch