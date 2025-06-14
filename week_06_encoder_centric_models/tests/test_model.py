from src.model.classifier import DistilBERTClassifier

def test_forward_pass(model, tokenizer): 
    text = ['This is an example test sentence.']
    enc = tokenizer(text, return_tensors = 'pt', padding = True, truncation = True)
    logits = model(enc['input_ids'], enc['attention_mask'])
    
    assert logits.shape == (1, 2), f"Expected shape (1, 2), but got {logits.shape}"

def test_freeze_all_layers(): 
    model = DistilBERTClassifier(num_classes = 2, classifier_dim = 64, freeze_bert = True)
    assert all(not param.requires_grad for param in model.bert.parameters()), 'All BERT parameters should be frozen.'
    assert all(param.requires_grad for param in model.classifier.parameters()), 'Classifier parameters should be trainable.'