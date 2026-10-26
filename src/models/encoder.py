import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ChannelAttention(nn.Module):
    """Channel attention module for better feature selection"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch, channels, _ = x.size()
        y = self.global_avgpool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1)
        return x * y

class EnhancedSelfAttentionPooling(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softmax(dim=-1)
        )
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        attention_weights = self.attention(x)
        attended = torch.sum(x * attention_weights, dim=1)
        return attended

class EnhancedRegionEncoder(nn.Module):
    def __init__(self, n_channels, output_dim):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_channels, 128, kernel_size=15, padding=7),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.2),
                ChannelAttention(128)
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=11, padding=5),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.2),
                ChannelAttention(256)
            ),
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=7, padding=3),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.2),
                ChannelAttention(512)
            )
        ])
        
        self.residuals = nn.ModuleList([
            nn.Conv1d(n_channels, 128, kernel_size=1) if n_channels != 128 else nn.Identity(),
            nn.Conv1d(128, 256, kernel_size=1) if 128 != 256 else nn.Identity(),
            nn.Conv1d(256, 512, kernel_size=1) if 256 != 512 else nn.Identity()
        ])
        
        self.self_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True, dropout=0.1)
        self.layer_norm1 = nn.LayerNorm(512)
        self.pooling = EnhancedSelfAttentionPooling(512)
        
        self.projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Orthogonal initialization to prevent initial mode collapse"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        
        for conv, res in zip(self.conv_layers, self.residuals):
            identity = res(x)
            x = conv(x)
            if identity.shape[-1] != x.shape[-1]:
                identity = F.adaptive_avg_pool1d(identity, x.shape[-1])
            x = x + identity
            x = F.gelu(x)
        
        x_attn = x.transpose(1, 2)
        attn_out, _ = self.self_attention(x_attn, x_attn, x_attn)
        x_attn = self.layer_norm1(x_attn + attn_out)
        x = x_attn.transpose(1, 2)
        
        pooled = self.pooling(x)
        return self.projection(pooled)

class Encoder(nn.Module):
    def __init__(self, region_channel_counts, hidden_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.region_names = ['frontal', 'temporal', 'central', 'parietal']
        
        self.region_encoders = nn.ModuleDict()
        for region in self.region_names:
            n_channels = region_channel_counts.get(region, 10)
            self.region_encoders[region] = EnhancedRegionEncoder(n_channels, hidden_dim // 4)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize fusion weights orthogonally as well
        self._init_fusion()
        
        logger.info(f"Enhanced Encoder initialized. Adaptive Noise & Orthogonal Init enabled.")

    def _init_fusion(self):
        for m in self.fusion_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def get_region_info(self):
        return {'regions': self.region_names, 'output_dim': self.hidden_dim}

    def forward(self, eeg_data):
        batch_size = eeg_data[0].shape[0]
        features = []
        for i, region in enumerate(self.region_names):
            features.append(self.region_encoders[region](eeg_data[i]))
        
        combined = torch.cat(features, dim=1)
        fused = self.fusion_mlp(combined) + combined
        fused = self.final_norm(fused)
        
        diversity_loss = torch.tensor(0.0, device=fused.device)
        
        # Adaptive Gaussian Perturbation (Batch Anti-Collapse)
        if self.training and batch_size > 1:
            fused_norm = F.normalize(fused, p=2, dim=1)
            sim_matrix = torch.mm(fused_norm, fused_norm.t())
            mask = torch.eye(batch_size, device=fused.device).bool()
            mean_sim = sim_matrix[~mask].mean()
            
            threshold = 0.4 # Threshold should auto-adapt per batch sentence embedding cosine similarity
            if mean_sim > threshold:
                # Dynamic noise proportional to collapse severity
                noise_scale = (mean_sim - threshold) * 1.0
                noise = torch.randn_like(fused) * noise_scale
                fused = fused + noise
                
                # Strong penalty
                off_diag = sim_matrix[~mask]
                diversity_loss = (off_diag.clamp(min=threshold) ** 2).mean() * 10.0

        return fused, diversity_loss