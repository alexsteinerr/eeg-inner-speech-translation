import torch
import torch.nn as nn
import logging

from .encoder import Encoder
from .decoder import Decoder

logger = logging.getLogger(__name__)

class EEGModel(nn.Module):
    def __init__(
        self, 
        region_channel_counts: dict, 
        hidden_dim: int = 1024,
        pretrained_model: str = 'Qwen/Qwen2.5-1.5B-Instruct',
        decoder_prefix_len: int = 16,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        diversity_lambda: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.region_channel_counts = region_channel_counts
        self.pretrained_model = pretrained_model
        self.decoder_prefix_len = decoder_prefix_len
        self.diversity_lambda = diversity_lambda
        
        logger.info(f"Initializing EEG-to-Text Model with {pretrained_model}")
        logger.info(f"Diversity lambda: {diversity_lambda}")
        
        self.encoder = Encoder(
            region_channel_counts=region_channel_counts,
            hidden_dim=hidden_dim
        )
        
        self.decoder = Decoder(
            hidden_dim=hidden_dim,
            pretrained_model=pretrained_model,
            decoder_prefix_len=decoder_prefix_len,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        
        self._initialize_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        self.encoder.apply(init_weights)

    def forward(self, eeg_data, input_ids=None, labels=None, return_encoder_features=False, **kwargs):
        # Encoder returns both features and diversity loss (paper Eq. 10-12)
        eeg_features, diversity_loss = self.encoder(eeg_data)
        
        # Handle potential NaN/Inf values
        if torch.isnan(eeg_features).any() or torch.isinf(eeg_features).any():
            logger.warning("NaN/Inf detected in encoder features, applying correction")
            eeg_features = torch.randn_like(eeg_features) * 0.01
            diversity_loss = torch.tensor(0.0, device=eeg_features.device)
        
        # Ensure proper shape
        if eeg_features.dim() == 1:
            eeg_features = eeg_features.unsqueeze(0)
        
        # Get decoder outputs with causal language modeling loss (paper Eq. 3)
        outputs = self.decoder(
            eeg_feat=eeg_features,
            input_ids=input_ids,
            labels=labels,
            **kwargs
        )
        
        # Combine CLM loss with diversity regularization (paper Eq. 12)
        clm_loss = outputs['loss']
        total_loss = clm_loss + self.diversity_lambda * diversity_loss
        
        # Return all components for monitoring and training
        outputs.update({
            'total_loss': total_loss,
            'clm_loss': clm_loss,
            'diversity_loss': diversity_loss,
            'eeg_features': eeg_features
        })
        
        if return_encoder_features:
            outputs['encoder_features'] = eeg_features
            
        return outputs

    def generate(self, eeg_data, **kwargs):
        with torch.no_grad():
            # For generation, we only need features, not diversity loss
            eeg_features, _ = self.encoder(eeg_data)
            
            if torch.isnan(eeg_features).any():
                logger.warning("NaN detected in generation features, using zeros")
                eeg_features = torch.zeros_like(eeg_features)
            
            return self.decoder.generate(eeg_features, **kwargs)
    
    def get_model_info(self):
        return {
            'eeg_encoder': {
                'regions': self.encoder.get_region_info(),
                'output_dim': self.hidden_dim
            },
            'llm_decoder': {
                'model': self.pretrained_model,
                'prefix_length': self.decoder_prefix_len,
                'vocab_size': self.decoder.vocab_size,
                'model_dim': self.decoder.model_dim,
                'lora_enabled': self.decoder.use_lora
            },
            'parameters': {
                'total': sum(p.numel() for p in self.parameters()),
                'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)
            },
            'loss_config': {
                'diversity_lambda': self.diversity_lambda
            }
        }