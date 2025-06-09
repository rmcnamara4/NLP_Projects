from src.data.dataset import TranslationDataset, collate_fn 
from src.data.preprocessing import numericalize, tokenize, build_vocab

from src.model.transformer import TransformerModel 

from src.utils.config import load_config
from src.utils.logging import setup_logging

from src.data.dataset import load_tokenized_data

from src.training.train import train
from src.training.evaluation import evaluate_bleu

import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from nltk.translate.bleu_score import SmoothingFunction

import os
import shutil
import logging

def main(): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    config = load_config('./src/config.yaml')
    setup_logging(log_file = config['paths']['train_log'], filemode = config['logging']['filemode'])

    src_lang = config['dataset']['src_lang']
    tgt_lang = config['dataset']['tgt_lang']

    (train_src_tokens, train_tgt_tokens), (val_src_tokens, val_tgt_tokens), (test_src_tokens, test_tgt_tokens) = load_tokenized_data(src_lang, tgt_lang, config['dataset']['name'])

    source_vocab = build_vocab(train_src_tokens)
    source_vocab.set_default_index(source_vocab['<UNK>'])

    target_vocab = build_vocab(train_tgt_tokens)
    target_vocab.set_default_index(target_vocab['<UNK>'])

    train_dataset = TranslationDataset(train_src_tokens, train_tgt_tokens, source_vocab, target_vocab)
    val_dataset = TranslationDataset(val_src_tokens, val_tgt_tokens, source_vocab, target_vocab)
    test_dataset = TranslationDataset(test_src_tokens, test_tgt_tokens, source_vocab, target_vocab)

    max_len = config['model']['max_len']
    batch_size = config['training']['batch_size']

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda x: collate_fn(x, source_vocab, target_vocab, train = True, max_len = max_len), num_workers = 0, pin_memory = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, collate_fn = lambda x: collate_fn(x, source_vocab, target_vocab, train = True, max_len = max_len), num_workers = 0, pin_memory = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = lambda x: collate_fn(x, source_vocab, target_vocab, train = False, max_len = max_len), num_workers = 0, pin_memory = True)

    model = TransformerModel(
        source_vocab, 
        target_vocab, 
        d_model = config['model']['d_model'], 
        d_k = config['model']['d_k'],
        d_v = config['model']['d_v'],
        n_heads = config['model']['n_heads'],
        ffn_hidden = config['model']['ffn_hidden'],
        n_layers = config['model']['n_layers'],
        encoder_dropout = config['model']['encoder_dropout'],
        decoder_dropout = config['model']['decoder_dropout'],
        max_len = max_len,
        return_attn = config['model']['return_attn'], 
        device = device
    ).to(device) 

    criterion = nn.CrossEntropyLoss(ignore_index = target_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr = config['training']['learning_rate'])

    # train_losses, val_losses = train(
    #     model, 
    #     train_dataloader, 
    #     val_dataloader, 
    #     optimizer, 
    #     criterion, 
    #     device, 
    #     epochs = config['training']['epochs'],
    #     print_every = config['training']['print_every'],
    #     patience = config['training']['patience'],
    #     checkpoint_path = os.path.join(config['paths']['model_dir'], config['paths']['checkpoint_file']),
    #     resume = config['training']['resume']
    # )

    # torch.save(model.state_dict(), os.path.join(config['paths']['model_dir'], config['paths']['model_file']))
    # torch.save(train_losses, os.path.join(config['paths']['model_dir'], config['paths']['train_loss_file']))
    # torch.save(val_losses, os.path.join(config['paths']['model_dir'], config['paths']['val_loss_file']))

    # logging.info('Training complete and model saved.') 

    state_dict = torch.load(os.path.join(config['paths']['model_dir'], config['paths']['model_file']))
    model.load_state_dict(state_dict)
    logging.info('Starting evaluation...')

    smoother = SmoothingFunction().method4
    bleu_score = evaluate_bleu(
        model, 
        test_dataloader, 
        smoother, 
        beam_width = config['evaluation']['beam_width'], 
        max_len = config['model']['max_len'], 
        alpha = config['evaluation']['alpha'], 
        device = device
    )

    os.makedirs(config['paths']['model_dir'], exist_ok = True) 
    shutil.copy('./src/config.yaml', config['paths']['model_dir'])

    logging.info('Evaluation complete.')

if __name__ == '__main__':
    main()