#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import random
import logging
import re
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm

try:
    from nltk.translate.bleu_score import corpus_bleu
    from rouge import Rouge
except ImportError:
    print("Please install nltk and rouge: pip install nltk rouge")
    sys.exit(1)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.model import EEGModel
from config.training_config import CONFIG
from src.data.dataset import EEGDataset
from src.data.utils import get_electrode_regions

file_handler = logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[stream_handler, file_handler]
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

def create_model(region_channel_counts, device):
    logger.info("Initializing EEG Model...")
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
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Created. Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    return model

def get_optimizer_and_scheduler(model, num_training_steps):
    named_params = list(model.named_parameters())
    encoder_params = [p for n, p in named_params if "encoder" in n and p.requires_grad]
    decoder_params = [p for n, p in named_params if "decoder" in n and p.requires_grad and "encoder" not in n]
    
    enc_lr = CONFIG.get('encoder_lr', 1e-3)
    dec_lr = CONFIG.get('decoder_lr', 1e-5)
    weight_decay = CONFIG.get('weight_decay', 0.01)
    
    optimizer_groups = [
        {'params': encoder_params, 'lr': enc_lr, 'weight_decay': weight_decay},
        {'params': decoder_params, 'lr': dec_lr, 'weight_decay': weight_decay}
    ]
    
    optimizer = AdamW(optimizer_groups, eps=1e-8, betas=(0.9, 0.999))
    warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, scheduler

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss, total_clm_loss, total_div_loss = 0, 0, 0
    steps = len(dataloader)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Train", unit="batch")
    
    for batch in pbar:
        eeg_data = [region.to(device) for region in batch['eeg']]
        labels = batch['labels'].to(device)
        text_ids = batch['decoder_input_ids'].to(device)
        
        optimizer.zero_grad()
        outputs = model(eeg_data=eeg_data, input_ids=text_ids, labels=labels)
        loss = outputs['total_loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_clm_loss += outputs['clm_loss'].item()
        total_div_loss += outputs['diversity_loss'].item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'clm': f"{outputs['clm_loss'].item():.4f}",
            'div': f"{outputs['diversity_loss'].item():.4f}"
        })
            
    return total_loss / steps, total_clm_loss / steps, total_div_loss / steps

def validate(model, dataloader, device):
    model.eval()
    total_loss, total_clm_loss = 0, 0
    pbar = tqdm(dataloader, desc="Validating", unit="batch", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            eeg_data = [region.to(device) for region in batch['eeg']]
            labels = batch['labels'].to(device)
            text_ids = batch['decoder_input_ids'].to(device)
            
            outputs = model(eeg_data=eeg_data, input_ids=text_ids, labels=labels)
            total_loss += outputs['total_loss'].item()
            total_clm_loss += outputs['clm_loss'].item()
            pbar.set_postfix({'val_loss': f"{outputs['total_loss'].item():.4f}"})
            
    return total_loss / len(dataloader), total_clm_loss / len(dataloader)

def generate_samples(model, dataloader, device, num_samples=3):
    model.eval()
    logger.info("\n--- Qualitative Evaluation ---")
    tokenizer = model.decoder.tokenizer
    
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples: break
            
            eeg_data = [region.to(device) for region in batch['eeg']]
            labels = batch['labels'].to(device)
            
            for i in range(len(labels)):
                if count >= num_samples: break
                single_eeg = [r[i:i+1] for r in eeg_data]
                
                gen_ids = model.generate(
                    eeg_data=single_eeg, max_new_tokens=64,
                    do_sample=True, temperature=0.7, top_p=0.9
                )
                
                target_ids = labels[i].cpu().numpy()
                target_ids = target_ids[target_ids != -100]
                target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
                pred_text = tokenizer.decode(gen_ids[0].cpu().numpy(), skip_special_tokens=True)
                
                logger.info(f"Sample {count+1}:")
                logger.info(f"  Target: {_merge_cjk_spaces(target_text)}")
                logger.info(f"  Pred:   {_merge_cjk_spaces(pred_text)}")
                count += 1
    logger.info("------------------------------\n")

def compute_scores(predictions, references):
    rouge = Rouge()
    tokenized_preds = []
    tokenized_refs = []
    
    for p, r in zip(predictions, references):
        p_clean = _merge_cjk_spaces(p)
        r_clean = _merge_cjk_spaces(r)
        
        p_chars = [c for c in p_clean]
        r_chars = [c for c in r_clean]
        
        tokenized_preds.append(p_chars)
        tokenized_refs.append([r_chars])
    
    bleu1 = corpus_bleu(tokenized_refs, tokenized_preds, weights=(1.0, 0, 0, 0))
    
    rouge_preds = [" ".join(p) for p in tokenized_preds]
    rouge_refs = [" ".join(r[0]) for r in tokenized_refs]
    
    try:
        rouge_scores = rouge.get_scores(rouge_preds, rouge_refs, avg=True)
        rouge_l_f1 = rouge_scores['rouge-l']['f']
    except ValueError:
        rouge_l_f1 = 0.0
        
    return bleu1, rouge_l_f1

def test_model(model, dataloader, device):
    logger.info("Starting evaluation on TEST set (Remaining 15%)...")
    model.eval()
    tokenizer = model.decoder.tokenizer
    
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing Generation"):
            eeg_data = [region.to(device) for region in batch['eeg']]
            labels = batch['labels'].to(device)
            
            gen_ids = model.generate(
                eeg_data=eeg_data,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1
            )
            
            decoded_preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            
            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_preds.extend(decoded_preds)
            all_refs.extend(decoded_refs)
            
    bleu1, rouge_l = compute_scores(all_preds, all_refs)
    
    logger.info("="*40)
    logger.info(f"TEST RESULTS (Samples: {len(all_preds)})")
    logger.info(f"Chinese OBCV BLEU-1: {bleu1*100:.2f}")
    logger.info(f"ROUGE-L F1:          {rouge_l*100:.2f}")
    logger.info("="*40)

def main():
    torch.autograd.set_detect_anomaly(True)
    seed = CONFIG.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    region_indices = get_electrode_regions()
    region_channel_counts = {region: len(indices) for region, indices in region_indices.items()}
    
    logger.info("Initializing Dataset...")
    temp_model = create_model(region_channel_counts, "cpu") 
    tokenizer = temp_model.decoder.tokenizer
    del temp_model
    
    full_dataset = EEGDataset(
        data_dir=CONFIG['data_dir'],
        csv_path=CONFIG['montage_file'],
        tokenizer=tokenizer,
        max_length=CONFIG['max_length'],
        max_samples=CONFIG.get('max_samples', None)
    )
    
    total_len = len(full_dataset)
    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    logger.info(f"Dataset Split -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    if sys.platform == 'win32':
        num_workers = 0
    else:
        num_workers = CONFIG.get('num_workers', 4)

    batch_size = CONFIG.get('batch_size', 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = create_model(region_channel_counts, device)
    
    num_epochs = CONFIG.get('num_epochs', 30)
    total_steps = len(train_loader) * num_epochs
    optimizer, scheduler = get_optimizer_and_scheduler(model, total_steps)
    
    save_dir = os.path.join(CONFIG['save_dir'], f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    patience = CONFIG.get('patience', 5)
    patience_counter = 0
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    
    logger.info("Starting Training...")
    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch+1}/{num_epochs} ===")
        
        avg_train_loss, train_clm, train_div = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        logger.info(f"Epoch {epoch+1} Train - Loss: {avg_train_loss:.4f} (CLM: {train_clm:.4f}, Div: {train_div:.4f})")
        
        avg_val_loss, val_clm = validate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} Val   - Loss: {avg_val_loss:.4f} (CLM: {val_clm:.4f})")
        
        if (epoch + 1) % 2 == 0:
            generate_samples(model, val_loader, device)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training Complete.")

    logger.info("\nLoading best model for Final Testing...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        logger.warning("Best model not found, testing with current weights.")

    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()