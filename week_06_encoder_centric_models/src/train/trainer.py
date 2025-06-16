import torch 
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score
import logging
import os

class Trainer: 
    def __init__(self, model, optimizer, criterion, train_dataloader, val_dataloader, device = 'cpu', scheduler = None): 
        self.model = model 
        self.optimizer = optimizer 
        self.criterion = criterion 
        self.train_dataloader = train_dataloader 
        self.val_dataloader = val_dataloader
        self.device = device 
        self.scheduler = scheduler 

        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled = self.use_amp)

    def train_one_epoch(self, print_every = 100): 
        self.model.train()

        all_pred_probas = []
        all_labels = []

        running_loss = 0.0 
        total_samples = 0

        for i, batch in enumerate(self.train_dataloader): 
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled = self.use_amp): 
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Predictions
            prob = F.softmax(logits, dim = 1)[:, 1].detach().cpu().numpy()
            labs = labels.detach().cpu().numpy()

            all_pred_probas.extend(prob.tolist())
            all_labels.extend(labs.tolist()) 

            batch_size = input_ids.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            if i % print_every == 0: 
                logging.info(f'Batch {i + 1} Loss: {loss:.4f}')
            
            if self.use_amp: 
                torch.cuda.empty_cache()

        return running_loss / total_samples, all_pred_probas, all_labels
    
    def evaluate_one_epoch(self, dataloader): 
        self.model.eval()

        all_pred_probas = []
        all_labels = []

        running_loss = 0.0 
        total_samples = 0
        
        with torch.no_grad(): 
            for i, batch in enumerate(dataloader): 
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask) 

                loss = self.criterion(logits, labels) 

                prob = F.softmax(logits, dim = 1)[:, 1].cpu().numpy()
                labs = labels.cpu().numpy()

                all_pred_probas.extend(prob.tolist())
                all_labels.extend(labs.tolist())

                batch_size = input_ids.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

        return running_loss / total_samples, all_pred_probas, all_labels
    
    def train(self, epochs, patience = 3, print_every = 100, checkpoint_path = None, resume = False): 
        train_losses = []
        val_losses = []

        train_auprcs = []
        val_auprcs = []

        best_val_auprc = float('-inf') 
        no_improve_epochs = 0

        start_epoch = 0

        if resume and checkpoint_path and os.path.exists(checkpoint_path): 
            start_epoch, best_val_auprc, train_losses, val_losses = self.load_checkpoint(checkpoint_path)
            logging.info(f'Resuming training from epoch {start_epoch + 1} with best_val_auprc = {best_val_auprc:.4f}')

        best_model_state = self.model.state_dict()
        
        for epoch in range(start_epoch, epochs): 
            logging.info(f'Epoch {epoch + 1} / {epochs}')
            logging.info('-' * 30) 

            train_loss, train_pred_probas, train_labels = self.train_one_epoch(print_every) 
            val_loss, val_pred_probas, val_labels = self.evaluate_one_epoch(self.val_dataloader) 

            train_auprc = average_precision_score(train_labels, train_pred_probas)
            val_auprc = average_precision_score(val_labels, val_pred_probas)

            if self.scheduler: 
                self.scheduler.step(val_loss) 

            current_lr = self.optimizer.param_groups[0]['lr']
            
            logging.info('=' * 80)
            logging.info(f'Train Loss: {train_loss:.4f} | Train AUPRC: {train_auprc:.4f} | Val Loss: {val_loss:.4f} | Val AUPRC: {val_auprc:.4f} | LR: {current_lr:.2e}')
            logging.info('=' * 80)

            train_losses.append(train_loss) 
            val_losses.append(val_loss) 

            train_auprcs.append(train_auprc) 
            val_auprcs.append(val_auprc) 

            if val_auprc > best_val_auprc: 
                best_val_auprc = val_auprc 
                no_improve_epochs = 0 
                best_model_state = self.model.state_dict()

                if checkpoint_path: 
                    self.save_checkpoint(epoch, best_val_auprc, checkpoint_path, train_losses = train_losses, val_losses = val_losses)

            else: 
                no_improve_epochs += 1
                logging.info(f'No improvement in validation AUPRC for {no_improve_epochs} epochs.')

                if no_improve_epochs >= patience: 
                    logging.info(f'Early stopping triggered after {epoch + 1} epochs.')
                    break 

            logging.info('\n')

        self.model.load_state_dict(best_model_state) 

        return train_losses, train_auprcs, val_losses, val_auprcs
    
    def save_checkpoint(self, epoch, best_val_auprc, path, train_losses = None, val_losses = None):
        os.makedirs(os.path.dirname(path), exist_ok = True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auprc': best_val_auprc,
            'train_losses': train_losses, 
            'val_losses': val_losses
        }
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location = self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        best_val_auprc = checkpoint.get('best_val_auprc', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        logging.info(f"Loaded checkpoint from {path}")
        return epoch, best_val_auprc, train_losses, val_losses
