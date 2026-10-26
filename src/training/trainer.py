import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class EEGTrainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, optimizer, scheduler, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        self.device = next(model.parameters()).device
        
        self.use_amp = config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def apply_augmentations(self, eeg_data):
        """Simple augmentations as mentioned in paper"""
        augmented = []
        
        for region_data in eeg_data:
            batch_size = region_data.shape[0]
            region_augmented = []
            
            for b in range(batch_size):
                sample = region_data[b:b+1].clone()
                
                # Random temporal shift (paper mentions temporal dependencies)
                shift = torch.randint(-3, 4, (1,)).item()
                sample = torch.roll(sample, shifts=shift, dims=-1)
                
                # Small noise addition
                noise = torch.randn_like(sample) * 0.02
                sample = sample + noise
                
                region_augmented.append(sample)
            
            augmented.append(torch.cat(region_augmented, dim=0))
        
        return augmented

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_clm_loss = 0
        total_diversity_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            eeg_data = [region.to(self.device) for region in batch['eeg']]
            labels = batch['labels'].to(self.device)
            text_ids = batch['decoder_input_ids'].to(self.device)
            
            augmented_eeg = self.apply_augmentations(eeg_data)
            
            self.optimizer.zero_grad()
            outputs = self.model(
                eeg_data=augmented_eeg, 
                input_ids=text_ids, 
                labels=labels
            )
            
            total_loss_batch = outputs['total_loss']
            clm_loss = outputs['clm_loss']
            diversity_loss = outputs['diversity_loss']
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping as mentioned in paper
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += total_loss_batch.item()
            total_clm_loss += clm_loss.item()
            total_diversity_loss += diversity_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Total Loss: {total_loss_batch.item():.4f}, "
                    f"CLM: {clm_loss.item():.4f}, "
                    f"Diversity: {diversity_loss.item():.4f}"
                )
        
        avg_loss = total_loss / len(self.train_loader)
        avg_clm = total_clm_loss / len(self.train_loader)
        avg_diversity = total_diversity_loss / len(self.train_loader)
        
        logger.info(
            f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, "
            f"CLM: {avg_clm:.4f}, Diversity: {avg_diversity:.4f}"
        )
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                eeg_data = [region.to(self.device) for region in batch['eeg']]
                labels = batch['labels'].to(self.device)
                text_ids = batch['decoder_input_ids'].to(self.device)
                
                outputs = self.model(
                    eeg_data=eeg_data,
                    input_ids=text_ids,
                    labels=labels
                )
                
                total_loss += outputs['total_loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Epoch {epoch} - Val Loss: {avg_loss:.4f}")
        return avg_loss