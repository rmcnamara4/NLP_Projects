from src.data_utils import * 
from src.train_utils import * 
from src.models import *
from src.model_utils import *

import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from metric_utils import * 

import pandas as pd
import swifter 
from collections import Counter

import torch 
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # load data
    X_train = pd.read_csv('../data/amazon_sentiment/X_train_nltk.csv')
    X_val = pd.read_csv('../data/amazon_sentiment/X_val_nltk.csv')
    X_test = pd.read_csv('../data/amazon_sentiment/X_test_nltk.csv')

    y_train = pd.read_csv('../data/amazon_sentiment/y_train.csv').squeeze()
    y_val = pd.read_csv('../data/amazon_sentiment/y_val.csv').squeeze()
    y_test = pd.read_csv('../data/amazon_sentiment/y_test.csv').squeeze()

    # drop NaN values 
    inds = X_train['review'].isna()
    X_train = X_train[~inds]
    y_train = y_train[~inds]

    inds = X_val['review'].isna()
    X_val = X_val[~inds]
    y_val = y_val[~inds]

    inds = X_test['review'].isna()
    X_test = X_test[~inds]
    y_test = y_test[~inds]

    # remove short reviews
    inds = X_train['review'].swifter.apply(len).values > 10
    X_train = X_train[inds]
    y_train = y_train[inds]

    inds = X_val['review'].swifter.apply(len).values > 10
    X_val = X_val[inds]
    y_val = y_val[inds]

    inds = X_test['review'].swifter.apply(len).values > 10
    X_test = X_test[inds]
    y_test = y_test[inds]

    # tokenize reviews 
    X_train_tokens = X_train['review'].swifter.apply(lambda x: x.split()).tolist()
    X_val_tokens = X_val['review'].swifter.apply(lambda x: x.split()).tolist()
    X_test_tokens = X_test['review'].swifter.apply(lambda x: x.split()).tolist()

    # construct vocabulary 
    counter = Counter()
    for text in X_train_tokens:
        counter.update(text)

    # filter vocabulary to have at least 5 occurrences
    vocab_filtered = {k:v for k, v in counter.items() if v >= 5}

    # add special tokens
    final_vocab = ['<PAD>', '<UNK>'] + list(vocab_filtered.keys())

    # create stoi and itos 
    stoi = {word:i for i, word in enumerate(final_vocab)}
    itos = {i:word for word, i in stoi.items()}

    # set up model parameters 
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = CustomDataset(X_train_tokens, y_train.values, stoi)
    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, collate_fn = lambda x: collate_fn(x, stoi))

    val_dataset = CustomDataset(X_val_tokens, y_val.values, stoi)
    val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = False, collate_fn = lambda x: collate_fn(x, stoi))

    test_dataset = CustomDataset(X_test_tokens, y_test.values, stoi)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False, collate_fn = lambda x: collate_fn(x, stoi))

    ##########################################################################################
    # Base LSTM Classifier 
    ##########################################################################################
    # initiailze model 
    # model = LSTMClassifier(stoi, 100, 128, 1).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # # train and evaluate model 
    # print('Training Base LSTM Classifier...')
    # best_model_state, train_losses, val_losses, train_aurocs, val_aurocs, train_auprcs, val_auprcs =  train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, device, epochs = 10, patience = 2, print_every = 600)
    # print('Training complete!')
    # print()

    # # save 
    # torch.save(best_model_state, './results/base_model/lstm_classifier.pth')

    # torch.save(train_losses, './results/base_model/train_losses.pth')
    # torch.save(train_aurocs, './results/base_model/train_aurocs.pth')
    # torch.save(train_auprcs, './results/base_model/train_auprcs.pth')

    # torch.save(val_losses, './results/base_model/val_losses.pth')
    # torch.save(val_aurocs, './results/base_model/val_aurocs.pth')
    # torch.save(val_auprcs, './results/base_model/val_auprcs.pth')

    # # evaluate on test set
    # print('Evaluating Base LSTM Classifier on Test Set...')
    # print()
    # plot_loss(train_losses, val_losses, './results/base_model/loss_over_epochs.png')

    # model.eval()
    # with torch.no_grad(): 
    #     test_pred_proba, test_labels, test_avg_loss, _, _ = evaluate_one_epoch(model, criterion, test_loader, device)

    # test_pred = (test_pred_proba >= 0).astype(int)

    # base_metrics = calculate_metrics(test_labels, test_pred_proba, test_pred, set = 'test')
    # plot_roc_curve(test_labels, test_pred_proba, './results/base_model/roc_curve.png', set = 'Test')
    # plot_pr_curve(test_labels, test_pred_proba, './results/base_model/pr_curve.png', set = 'Test')
    # create_confusion_matrix(test_labels, test_pred, './results/base_model/confusion_matrix.png', set = 'Test')

    ##########################################################################################
    # Attention LSTM Classifier 
    ##########################################################################################
    model = LSTMClassifierWithAttention(stoi, 100, 128, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # train and evaluate model 
    print('Training Attention LSTM Classifier...')
    best_model_state, train_losses, val_losses, train_aurocs, val_aurocs, train_auprcs, val_auprcs =  train_and_evaluate_model(model, train_loader, val_loader, criterion, optimizer, device, epochs = 10, patience = 2, print_every = 600)
    print('Training complete!')
    print()

    # save 
    torch.save(best_model_state, '../results/attention_model/lstm_classifier.pth')

    torch.save(train_losses, './results/attention_model/train_losses.pth')
    torch.save(train_aurocs, './results/attention_model/train_aurocs.pth')
    torch.save(train_auprcs, './results/attention_model/train_auprcs.pth')

    torch.save(val_losses, './results/attention_model/val_losses.pth')
    torch.save(val_aurocs, './results/attention_model/val_aurocs.pth')
    torch.save(val_auprcs, './results/attention_model/val_auprcs.pth')

    # evaluate on test set
    print('Evaluating Attention LSTM Classifier on Test Set...')
    print()
    plot_loss(train_losses, val_losses, './results/attention_model/loss_over_epochs.png')

    model.eval()
    with torch.no_grad(): 
        test_pred_proba, test_labels, test_avg_loss, _, _ = evaluate_one_epoch(model, criterion, test_loader, device)

    test_pred = (test_pred_proba >= 0).astype(int)

    attention_metrics = calculate_metrics(test_labels, test_pred_proba, test_pred, set = 'test')
    plot_roc_curve(test_labels, test_pred_proba, './results/attention_model/roc_curve.png', set = 'Test')
    plot_pr_curve(test_labels, test_pred_proba, './results/attention_model/pr_curve.png', set = 'Test')
    create_confusion_matrix(test_labels, test_pred, './results/attention_model/confusion_matrix.png', set = 'Test')

    # save final metrics 
    final_metrics = pd.DataFrame([base_metrics, attention_metrics]).T.reset_index()
    final_metrics.columns = ['Metric', 'Base Model', 'Attention Model']
    final_metrics['Metric'] = final_metrics['Metric'].str.replace('test_', '').str.title()

    final_metrics.to_csv('./results/final_metrics.csv', index = False, header = True)









