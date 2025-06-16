from transformers import AutoTokenizer, DataCollatorWithPadding
import torch 
from torch.utils.data import DataLoader

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.data.dataset import CivilDataset
from seed import set_seed
from config import load_config
from setup_logging import setup_logging
from src.model.classifier import DistilBERTClassifier
from src.train.trainer import Trainer
from src.utils.optimizer import get_optimizer
from src.utils.scheduler import get_scheduler
from src.utils.save_model_history import save_model_history
from src.utils.class_weights import get_class_weights
from src.utils.save_plots import save_plots
from src.utils.threshold import find_best_threshold, save_threshold
from metrics import plot_loss

import logging
import pandas as pd  
import shutil

def main(): 
    ########################################################################################################
    # Setup
    ########################################################################################################
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # load config
    config = load_config('./src/config.yaml') 
    set_seed(config['seed']) # set seed

    # setup logging
    setup_logging(log_file = config['paths']['train_log'], filemode = config['logging']['filemode'])

    ########################################################################################################
    # Load data
    ########################################################################################################
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
    logging.info(f"Tokenizer loaded: {config['model']['model_name']}")

    # create dataset
    dataset = CivilDataset(
        tokenizer = tokenizer, 
        max_length = config['dataset']['max_length'],
        binary_col = config['dataset']['binary_col'],
        thresh_val_size = config['dataset']['thresh_val_size'],
        random_state = config['dataset']['random_state']
    )
    splits = dataset.load() # split dataset into train, val, test, and threshold_val 
    tokenized_splits = dataset.tokenize(splits) # tokenizes each split 
    logging.info("Dataset loaded and tokenized.")

    ########################################################################################################
    # Create dataloaders
    ########################################################################################################
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer) # initialize data collator 
    batch_size = config['training']['batch_size']
    pin_memory = config['dataloader']['pin_memory']
    num_workers = config['dataloader']['num_workers']
    prefetch_factor = config['dataloader']['prefetch_factor']

    # create dataloaders
    train_dataloader = DataLoader(tokenized_splits['train'], batch_size = batch_size, shuffle = True, collate_fn = data_collator, pin_memory = pin_memory, num_workers = num_workers, prefetch_factor = prefetch_factor)
    val_dataloader = DataLoader(tokenized_splits['val'], batch_size = batch_size, shuffle = False, collate_fn = data_collator, pin_memory = pin_memory, num_workers = num_workers, prefetch_factor = prefetch_factor)
    test_dataloader = DataLoader(tokenized_splits['test'], batch_size = batch_size, shuffle = False, collate_fn = data_collator, pin_memory = pin_memory, num_workers = num_workers, prefetch_factor = prefetch_factor)
    threshold_val_dataloader = DataLoader(tokenized_splits['threshold_val'], batch_size = batch_size, shuffle = False, collate_fn = data_collator, pin_memory = pin_memory, num_workers = num_workers, prefetch_factor = prefetch_factor)
    logging.info("Data loaders created.")

    ########################################################################################################
    # Model setup
    ########################################################################################################
    # initialize model with parameters from config 
    model = DistilBERTClassifier( 
        num_classes = config['model']['num_classes'],
        classifier_dim = config['model']['classifier_dim'],
        dropout = config['model']['dropout'],
        use_cls = config['model']['use_cls'],
        freeze_bert = config['model']['freeze_bert']
    ).to(device) 
    logging.info(f"Model initialized.")

    # initialize optimizer 
    optimizer = get_optimizer(
        model = model, 
        name = config['optimizer']['name'],
        lr = config['optimizer']['lr']
    )
    logging.info(f"Optimizer initialized: {config['optimizer']['name']} with lr = {config['optimizer']['lr']}")

    # initialize scheduler if use_scheduler is True else set to None
    if config['scheduler']['use_scheduler']: 
        scheduler = get_scheduler(
            optimizer = optimizer, 
            mode = config['scheduler']['mode'],
            factor = config['scheduler']['factor'],
            patience = config['scheduler']['patience']
        )
    else: 
        scheduler = None
    logging.info(f"Scheduler initialized. Mode = {config['scheduler']['mode']}, factor = {config['scheduler']['factor']} and patience = {config['scheduler']['patience']}")

    # intiialize loss function using class weights if use_class_weights is True else set to None
    class_weights = get_class_weights(labels = tokenized_splits['train']['labels'], strategy = config['training']['class_weights_strategy']) if config['training']['use_class_weights'] else 'None'
    criterion = torch.nn.CrossEntropyLoss(weight = class_weights.to(device))
    logging.info("Loss function initialized.")

    ########################################################################################################
    # Training
    ########################################################################################################
    # intialize trainer
    trainer = Trainer(
        model = model, 
        optimizer = optimizer, 
        criterion = criterion, 
        train_dataloader = train_dataloader, 
        val_dataloader = val_dataloader, 
        device = device, 
        scheduler = scheduler
    )

    # train model and validate on validation set 
    train_losses, train_auprcs, val_losses, val_auprcs = trainer.train(
        epochs = config['training']['epochs'], 
        patience = config['training']['patience'], 
        print_every = config['training']['print_every'], 
        checkpoint_path = os.path.join(config['paths']['model_dir'], config['paths']['checkpoint_file']), 
        resume = config['training']['resume']
    )
    logging.info('Training complete!') 

    # save the model state dict and the training history 
    save_model_history(
        model = model, 
        train_losses = train_losses, 
        val_losses = val_losses, 
        train_auprcs = train_auprcs, 
        val_auprcs = val_auprcs, 
        config = config
    )

    # plot the training and validation loss over epochs 
    plot_save_path = config['paths']['model_dir']
    plot_loss(
        train_losses = train_losses, 
        val_losses = val_losses, 
        path = os.path.join(plot_save_path, 'loss_over_epochs.png')
    )
    logging.info('Loss plot saved!')

    ########################################################################################################
    # Threshold tuning 
    ########################################################################################################
    # evaluate model on threshold tuning set 
    logging.info('Evaluating on threshold tuning set...') 
    _, thresh_pred_probas, thresh_labels = trainer.evaluate_one_epoch(
        dataloader = threshold_val_dataloader
    )

    # find the best threshold using the threshold tuning predictions
    logging.info('Finding the best threshold...') 
    best_threshold, best_score = find_best_threshold(
        thresh_pred_probas, 
        thresh_labels, 
        step = config['threshold']['step'],
        beta = config['threshold']['beta']
    )
    save_threshold(best_threshold, os.path.join(config['paths']['model_dir'], config['paths']['results_dir'], 'best_threshold.json')) # save threshold 
    logging.info(f'Best threshold: {best_threshold:.4f} | Best score: {best_score:.4f}')

    ########################################################################################################
    # Evaluate on the test set 
    ########################################################################################################
    # evaluate model on test set 
    logging.info('Evaluating on test set...') 
    _, test_pred_probas, test_labels = trainer.evaluate_one_epoch(
        dataloader = test_dataloader
    )

    # convert predicted probabilities to binary predictions using the best threshold
    test_pred = (test_pred_probas >= best_threshold).astype(int)

    # save test predictions and labels 
    logging.info('Saving test predictions and evaluation plots...')
    test_results = pd.DataFrame({
        'pred_probas': test_pred_probas, 
        'labels': test_labels, 
        'preds': test_pred
    })
    test_results.to_csv(os.path.join(config['paths']['model_dir'], config['paths']['results_dir'], 'test_predictions.csv'), index = False)

    # save test set ROC Curve, PR Curve, Confusion Matrix, and Metrics 
    save_plots(
        y_true = test_labels, 
        y_pred_proba = test_pred_probas, 
        y_pred = test_pred,
        path = os.path.join(config['paths']['model_dir'], config['paths']['results_dir']), 
        set = 'Test'
    )
    logging.info('Test predictions and evaluation plots saved!')
    logging.info('Training and evaluation complete!')

    shutil.copy('./src/config.yaml', config['paths']['model_dir'])

if __name__ == '__main__':
    main()

