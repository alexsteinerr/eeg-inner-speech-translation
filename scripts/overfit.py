#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import random
from torch.utils.data import Subset
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import logging
from datetime import datetime
import torch.nn.functional as F
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.model import EEGModel
from config.training_config import CONFIG
from src.data.dataset import EEGDataset
from src.data.utils import get_electrode_regions

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_CJK_RGX = r"[\u2e80-\u2eff\u2f00-\u2fdf\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
_RE_CJK_INNER_SPACE = re.compile(rf"(?<={_CJK_RGX})\s+(?={_CJK_RGX})")

def _merge_cjk_spaces(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = _RE_CJK_INNER_SPACE.sub("", s)
    return s

def check_early_collapse(model, dataloader, device):
    """Check for collapse in first few batches"""
    logger.info("=== EARLY COLLAPSE DETECTION ===")
    # CRITICAL FIX: Must be in train mode for Anti-Collapse noise to work
    model.train() 
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
                
            eeg_data = [region.to(device) for region in batch['eeg']]
            
            # Forward pass
            outputs = model(eeg_data=eeg_data, input_ids=None, labels=None)
            features = outputs['eeg_features']
            diversity_loss = outputs['diversity_loss']
            
            # Calculate similarity
            normalized = F.normalize(features, dim=1)
            similarity = torch.mm(normalized, normalized.t())
            mask = torch.eye(similarity.size(0), dtype=torch.bool, device=device)
            mean_similarity = similarity[~mask].mean().item() if similarity.size(0) > 1 else 0.0
            
            logger.info(f"Batch {batch_idx}: Similarity={mean_similarity:.4f}, Diversity Loss={diversity_loss.item():.6f}")
            
            # Adjusted logic: Only stop if similarity is high AND diversity loss is zero
            # If diversity loss > 0, it means the model is fighting back (injecting noise)
            if mean_similarity > 0.95 and diversity_loss.item() < 0.001:
                logger.error("EARLY COLLAPSE DETECTED! Model is stuck and not fighting back.")
                return True
            elif mean_similarity > 0.95:
                logger.info("High similarity detected, but Anti-Collapse mechanism is active (Loss > 0). Continuing.")
                
    return False

def create_properly_initialized_model(region_channel_counts, device):
    logger.info("Creating EEG model for overfitting test...")
    model = EEGModel(
        region_channel_counts=region_channel_counts,
        hidden_dim=CONFIG['hidden_dim'],
        pretrained_model=CONFIG['pretrained_model'],
        decoder_prefix_len=CONFIG['decoder_prefix_len'],
        use_lora=CONFIG.get('use_lora', True),
        lora_rank=CONFIG.get('lora_rank', 32), 
        lora_alpha=CONFIG.get('lora_alpha', 64),
        lora_dropout=CONFIG.get('lora_dropout', 0.1),
        load_in_8bit=CONFIG.get('load_in_8bit', False),
        load_in_4bit=CONFIG.get('load_in_4bit', False),
        diversity_lambda=CONFIG.get('diversity_lambda', 1.0) 
    ).to(device)
    
    logger.info("=== MODEL VERIFICATION ===")
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable:,}")
    return model

def create_simple_optimizer(model):
    named_params = list(model.named_parameters())
    
    encoder_params = [p for n, p in named_params if "encoder" in n and p.requires_grad]
    decoder_params = [p for n, p in named_params if "decoder" in n and p.requires_grad and "encoder" not in n]
    
    optimizer_groups = [
        {'params': encoder_params, 'lr': 1e-3, 'weight_decay': 0.01},
        {'params': decoder_params, 'lr': 1e-5, 'weight_decay': 0.01}
    ]
    
    return AdamW(optimizer_groups, eps=1e-8, betas=(0.9, 0.999))

def monitor_feature_evolution(model, dataloader, device, epoch):
    logger.info(f"=== FEATURE EVOLUTION EPOCH {epoch} ===")
    model.eval() # Monitor in eval mode to see 'true' underlying features
    with torch.no_grad():
        batch = next(iter(dataloader))
        eeg_data = [region.to(device) for region in batch['eeg']]
        
        outputs = model(eeg_data=eeg_data, input_ids=None, labels=None)
        eeg_features = outputs['eeg_features']
        diversity_loss = outputs['diversity_loss']
        
        if eeg_features.shape[0] > 1:
            normalized = F.normalize(eeg_features, dim=1)
            similarity = torch.mm(normalized, normalized.t())
            mask = torch.eye(similarity.size(0), dtype=torch.bool, device=device)
            mean_similarity = similarity[~mask].mean().item()
        else:
            mean_similarity = 0.0
        
        logger.info(f"Pairwise Similarity: {mean_similarity:.4f}, Diversity Loss: {diversity_loss.item():.4f}")

def simple_train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_clm_loss = 0
    total_diversity_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        eeg_data = [region.to(device) for region in batch['eeg']]
        labels = batch['labels'].to(device)
        text_ids = batch['decoder_input_ids'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(eeg_data=eeg_data, input_ids=text_ids, labels=labels)
        total_loss_batch = outputs['total_loss']
        clm_loss = outputs['clm_loss']
        diversity_loss = outputs['diversity_loss']
        
        if batch_idx % 5 == 0:
            logger.info(f"Batch {batch_idx} - Total: {total_loss_batch.item():.4f}, CLM: {clm_loss.item():.4f}, Diversity: {diversity_loss.item():.4f}")
        
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_clm_loss += clm_loss.item()
        total_diversity_loss += diversity_loss.item()
    
    return total_loss / len(train_loader)

def print_detailed_evaluation_samples(model, val_loader, device, num_samples=5):
    logger.info("\n" + "="*80)
    logger.info("DETAILED EVALUATION")
    logger.info("="*80)
    model.eval()
    samples_shown = 0
    tokenizer = model.decoder.tokenizer
    
    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            if samples_shown >= num_samples: break
            eeg_data = [region.to(device) for region in batch['eeg']]
            targets = batch['labels'].to(device)
            
            for i in range(min(targets.size(0), num_samples - samples_shown)):
                target_ids = targets[i].cpu().numpy()
                target_ids = target_ids[target_ids >= 0]
                target_text = tokenizer.decode(target_ids.tolist(), skip_special_tokens=True)
                
                single_eeg = [region[i:i+1] for region in eeg_data]
                greedy_ids = model.generate(
                    eeg_data=single_eeg,
                    max_new_tokens=32,
                    do_sample=False
                )
                greedy_text = tokenizer.decode(greedy_ids[0].cpu().numpy().tolist(), skip_special_tokens=True)
                
                logger.info(f"\n--- Sample {samples_shown + 1} ---")
                logger.info(f"TARGET: {_merge_cjk_spaces(target_text)}")
                logger.info(f"PRED:   {_merge_cjk_spaces(greedy_text)}")
                samples_shown += 1
    logger.info("="*80)

def main():
    torch.autograd.set_detect_anomaly(True)
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    region_indices = get_electrode_regions()
    region_channel_counts = {region: len(indices) for region, indices in region_indices.items()}
    model = create_properly_initialized_model(region_channel_counts, device)
    
    dataset = EEGDataset(
        data_dir=CONFIG['data_dir'],
        csv_path=CONFIG['montage_file'],
        tokenizer=model.decoder.tokenizer,
        max_length=CONFIG['max_length'],
        max_samples=CONFIG['max_samples']
    )
    
    n_total = min(5, len(dataset))
    train_dataset = Subset(dataset, list(range(n_total)))
    val_dataset = Subset(dataset, list(range(n_total)))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(2, n_total), shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=min(2, n_total), shuffle=False)
    
    # Check for early collapse
    if check_early_collapse(model, train_loader, device):
        return
    
    optimizer = create_simple_optimizer(model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100*len(train_loader))
    
    save_dir = os.path.join(CONFIG['save_dir'], f'overfit_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(save_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(100):
        logger.info(f"OVERFIT EPOCH {epoch+1}/100")
        train_loss = simple_train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        logger.info(f"Training loss: {train_loss:.4f}")
        print_detailed_evaluation_samples(model, val_loader, device, num_samples=2)
        
        if (epoch + 1) % 2 == 0:
            monitor_feature_evolution(model, train_loader, device, epoch + 1)
            
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_overfit_model.pt'))
            
        if train_loss < 0.5:
            logger.info("Outstanding performance! Stopping.")
            break

if __name__ == "__main__":
    main()