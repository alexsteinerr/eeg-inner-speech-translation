import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CONFIG = {
    'data_dir': './data/eeg_data/',
    'montage_file': 'data/montage.csv',
    'save_dir': './checkpoints/',
    'pretrained_model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'hidden_dim': 1024,
    'max_length': 12,
    'epochs': 200,
    'batch_size': 2,
    'accumulation_steps': 4,
    'brain_encoder_lr': 1e-3,
    'llm_lr': 1e-5,
    'weight_decay': 0.01,
    'grad_clip_norm': 1.0,
    'warmup_ratio': 0.2,
    'warmup_steps': 200,
    'patience': 15,
    'dropout_rate': 0.4,
    'decoder_prefix_len': 16,
    'use_lora': True,
    'lora_rank': 32,
    'lora_alpha': 64,
    'lora_dropout': 0.1,
    'load_in_8bit': False,
    'load_in_4bit': False,
    'max_samples': None,
    'train_split': 0.9,
    'val_split': 0.1,
    'seed': 42,
    'mixed_precision': True,
    'unfreeze_layers': 6,
    'diversity_lambda': 5.0,
    'anti_collapse_threshold': 0.8,   
    'anti_collapse_eta': 1.0,    
    'anti_collapse_xi': 0.1,     
    'generation': {
        'eval': {
            'max_new_tokens': 15,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 20,
            'repetition_penalty': 1.3,
            'num_beams': 1,
        }
    }
}

def get_optimizer_groups(model):
    encoder_params = []
    llm_params = []
    lora_params = []
    projection_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'encoder' in name:
            encoder_params.append(param)
        elif 'lora_' in name:
            lora_params.append(param)
        elif 'eeg_to_prefix' in name:
            projection_params.append(param)
        elif 'decoder.llm' in name:
            llm_params.append(param)
    
    param_groups = []
    
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': CONFIG['brain_encoder_lr'],
            'weight_decay': CONFIG['weight_decay'],
        })
    
    if projection_params:
        param_groups.append({
            'params': projection_params,
            'lr': CONFIG['brain_encoder_lr'] * 0.5,
            'weight_decay': CONFIG['weight_decay'],
        })
    
    if llm_params:
        param_groups.append({
            'params': llm_params,
            'lr': CONFIG['llm_lr'],
            'weight_decay': CONFIG['weight_decay'] * 0.1,
        })
    
    if lora_params:
        param_groups.append({
            'params': lora_params,
            'lr': CONFIG['llm_lr'] * 2,
            'weight_decay': CONFIG['weight_decay'] * 0.01,
        })

    return param_groups