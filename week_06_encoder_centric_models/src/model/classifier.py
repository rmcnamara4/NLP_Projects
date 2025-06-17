import torch.nn as nn 
from typing import Union
from transformers import AutoModel

class DistilBERTClassifier(nn.Module): 
    """
    A DistilBERT-based sequence classifier with flexible freezing/unfreezing options.

    Args:
        num_classes (int): Number of output classes.
        classifier_dim (int): Dimension of the hidden layer in the classifier head.
        dropout (float): Dropout rate applied in the classifier head.
        use_cls (bool): Whether to use the [CLS] token for pooling. If False, uses mean pooling.
        freeze_bert (Union[bool, int]): 
            - True: freeze all BERT layers
            - False: unfreeze all BERT layers
            - int: freeze bottom N transformer layers
    """
    def __init__(self, num_classes, classifier_dim, dropout = 0.1, use_cls = True, freeze_bert: Union[bool, int] = True): 
        super(DistilBERTClassifier, self).__init__()
        self.use_cls = use_cls
        self.dropout = dropout
        self.classifier_dim = classifier_dim
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')

        if isinstance(freeze_bert, bool): 
            if freeze_bert: 
                self.freeze_all_bert_layers()
            else: 
                self.unfreeze_all_bert_layers()
        elif isinstance(freeze_bert, int): 
            if freeze_bert > 0: 
                self.freeze_bert_layers(freeze_bert)
            else: 
                self.unfreeze_all_bert_layers()
        else: 
            raise ValueError('freeze_bert must be True, False, or an integer.')

        hidden_size = self.bert.config.hidden_size 

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask): 
        """
        Runs a forward pass through BERT and the classifier head.

        Args:
            input_ids (torch.Tensor): Token IDs of shape [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits of shape [batch_size, num_classes]
        """
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.use_cls: 
            pooled_output = hidden_states[:, 0, :]
        else: 
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            masked_hidden = hidden_states * mask 
            summed = masked_hidden.sum(1)
            counts = attention_mask.sum(1).clamp(min = 1e-9).unsqueeze(1)
            pooled_output = summed / counts
        
        return self.classifier(pooled_output)
    
    def freeze_all_bert_layers(self): 
        """Freezes all layers and embeddings in the BERT encoder."""
        for param in self.bert.parameters(): 
            param.requires_grad = False 

    def unfreeze_all_bert_layers(self): 
        """Unfreezes all layers and embeddings in the BERT encoder."""
        for param in self.bert.parameters(): 
            param.requires_grad = True

    def freeze_bert_layers(self, num_layers_to_freeze): 
        """
        Freezes the embeddings and the first `num_layers_to_freeze` transformer layers.

        Args:
            num_layers_to_freeze (int): Number of bottom layers to freeze (0–6 for DistilBERT).
        """
        for param in self.bert.embeddings.parameters(): 
            param.requires_grad = False 

        for i, layer in enumerate(self.bert.transformer.layer): 
            for param in layer.parameters(): 
                param.requires_grad = i < num_layers_to_freeze

    def unfreeze_next_layer(self, current_unfrozen_idx): 
        """
        Unfreezes the next transformer layer from the bottom.

        Args:
            current_unfrozen_idx (int): Index of the next layer to unfreeze (0 to 5).
        """
        if current_unfrozen_idx < len(self.bert.transformer.layer): 
            for param in self.bert.transformer.layer[current_unfrozen_idx].parameters(): 
                param.requires_grad = True

    def get_trainable_layers(self): 
        """
        Returns a list of BERT components that are currently trainable.

        Returns:
            List[str]: Names of trainable components (e.g., ['embeddings', 'layer_4', 'layer_5'])
        """
        trainable = []
        if any(p.requires_grad for p in self.bert.embeddings.parameters()): 
            trainable.append('embeddings')
        
        for i, layer in enumerate(self.bert.transformer.layer):
            if any(p.requires_grad for p in layer.parameters()): 
                trainable.append(f'layer_{i}')

        return trainable