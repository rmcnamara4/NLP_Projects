from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

class CivilDataset: 
    """
    CivilDataset handles loading, preprocessing, binarizing, and tokenizing
    the Civil Comments dataset for toxicity classification tasks.

    It supports binary classification based on a selected toxicity-related column
    and splits the validation set for threshold tuning. The dataset is returned
    in Hugging Face `Dataset` format and prepared for PyTorch training.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Hugging Face tokenizer.
        max_length (int): Maximum token length for each input.
        binary_col (str): Column name to binarize as the classification target.
        thresh_val_size (float): Proportion of validation set to use for threshold tuning.
        random_state (int): Seed for reproducible train/val splitting.
    """
    def __init__(self, tokenizer, max_length, binary_col = 'toxicity', thresh_val_size = 0.5, random_state = 32): 
        self.binary_col = binary_col 
        self.thresh_val_size = thresh_val_size
        self.random_state = random_state
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load(self): 
        """
        Loads and preprocesses the Civil Comments dataset.

        This includes:
        - Removing unnecessary columns
        - Binarizing the selected target column
        - Splitting validation set into `val` and `threshold_val`

        Returns:
            dict[str, datasets.Dataset]: Dictionary containing the following splits:
                - 'train': Training data
                - 'val': Validation data for evaluation
                - 'threshold_val': Validation data for threshold selection
                - 'test': Test data
        """
        data = load_dataset('civil_comments', cache_dir = None) \
    
        features_to_remove = list(set(data['train'].features.keys()) - set(['text', 'label']))
        data = data.map(self._binarize, remove_columns = features_to_remove)
        data.set_format(type = 'pandas') 
        
        val = data['validation'][:].copy()
        val, threshold_val = train_test_split(val, test_size = self.thresh_val_size, random_state = self.random_state, stratify = val['label'])

        splits = {
            'train': Dataset.from_pandas(data['train'][:].reset_index(drop = True)), 
            'val': Dataset.from_pandas(val.reset_index(drop = True)),
            'threshold_val': Dataset.from_pandas(threshold_val.reset_index(drop = True)),
            'test': Dataset.from_pandas(data['test'][:].reset_index(drop = True))
        }

        return splits
    
    def tokenize(self, dataset): 
        """
        Applies tokenizer to the provided dataset.

        Uses the tokenizer defined in the constructor to tokenize
        text data and convert it into input IDs and attention masks.

        Args:
            dataset (datasets.Dataset): Dataset split to tokenize.

        Returns:
            datasets.Dataset: Tokenized dataset formatted for PyTorch
                with 'input_ids', 'attention_mask', and 'labels'.
        """
        tokenized_dataset = dataset.map(self._tokenize, batched = True, remove_columns = ['text'])
        tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
        tokenized_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
        return tokenized_dataset
    
    def _binarize(self, example): 
        """
        Converts the selected toxicity score to a binary label.

        Args:
            example (dict): A single example from the dataset.

        Returns:
            dict: The same example with a new 'label' key (0 or 1).
        """
        example['label'] = int(example[self.binary_col] >= 0.5)
        return example 
    
    def _tokenize(self, example): 
        """
        Tokenizes a single example using the instance's tokenizer.

        Args:
            example (dict): A single example with a 'text' field.

        Returns:
            dict: Dictionary with 'input_ids' and 'attention_mask'.
        """
        return self.tokenizer(
            example['text'],
            truncation = True,
            max_length =self.max_length,
            padding = False
        )