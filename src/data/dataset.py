import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import logging
from functools import lru_cache
import gc
import re

logger = logging.getLogger(__name__)

_CJK_RGX = r"[\u2e80-\u2eff\u2f00-\u2fdf\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
_RE_ZWSP = re.compile(r"[\u200b\u200c\u200d\u2060]")
_RE_FULLWIDTH_SPACE = re.compile(r"\u3000")
_RE_CJK_INNER_SPACE = re.compile(rf"(?<={_CJK_RGX})\s+(?={_CJK_RGX})")
_RE_WS = re.compile(r"\s+")

def normalize_text_preserve_cjk(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = _RE_ZWSP.sub("", text)
    text = _RE_FULLWIDTH_SPACE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_CJK_INNER_SPACE.sub("", text)
    parts = []
    i = 0
    while i < len(text):
        ch = text[i]
        if re.match(_CJK_RGX, ch):
            j = i + 1
            while j < len(text) and re.match(_CJK_RGX, text[j]):
                j += 1
            parts.append(text[i:j])
            i = j
        else:
            j = i + 1
            while j < len(text) and not re.match(_CJK_RGX, text[j]):
                j += 1
            seg = _RE_WS.sub(" ", text[i:j]).strip()
            if seg:
                if parts and not parts[-1].endswith(" ") and not seg.startswith(" "):
                    parts.append(" ")
                parts.append(seg)
                parts.append(" ")
            i = j
    out = "".join(parts).strip()
    return out if out else "默认文本"

class EEGDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        csv_path: str, 
        tokenizer, 
        max_length: int = 64, 
        eps: float = 1e-6, 
        max_samples: int = None, 
        use_cache: bool = True,
        cache_size: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eps = eps
        self.max_samples = max_samples
        self.use_cache = use_cache
        
        try:
            df = pd.read_csv(csv_path)
            self.ch_names = df['label'].to_numpy()
            logger.info(f"Loaded {len(self.ch_names)} electrode channels")
        except Exception as e:
            logger.error(f"Failed to load montage: {e}")
            raise
            
        self.region_indices = self._build_region_indices()
        self.region_channel_counts = {region: len(indices) for region, indices in self.region_indices.items()}
        
        self._setup_tokenizer()
        self.data_files = self._get_validated_data_files(data_dir)
        self.sample_index = self._build_sample_index()
        
        if len(self.sample_index) == 0:
            raise ValueError("No valid samples found")
            
        self._initialize_scalers()
        
        if self.use_cache:
            self._load_single_sample = lru_cache(maxsize=cache_size)(self._load_single_sample_uncached)
        else:
            self._load_single_sample = self._load_single_sample_uncached
            
        logger.info(f"EEG Dataset initialized with {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.sample_index)
    
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.sample_index):
            raise IndexError(f"Index {idx} out of range")
            
        sample_info = self.sample_index[idx]
        sample = self._load_single_sample(sample_info['file'], sample_info['index'])
        
        if sample is None:
            return self._create_fallback_sample()
            
        eeg_regions = self._normalize_eeg_sample(sample['input_features'])
        if eeg_regions is None:
            return self._create_fallback_sample()
            
        text = sample.get('text', '')
        if isinstance(text, list):
            text = "".join(map(str, text))
        if not isinstance(text, str):
            text = str(text)
            
        text = text.strip()
        if not text:
            text = "默认文本"
            
        tokenization = self._safe_tokenize(text)
        return {'eeg': eeg_regions, **tokenization}
    
    def _build_sample_index(self):
        sample_index = []
        total_samples = 0
        
        for file_idx, file_path in enumerate(self.data_files):
            try:
                with open(file_path, 'rb') as f:
                    loaded = pickle.load(f)
                    
                if isinstance(loaded, list):
                    for i in range(len(loaded)):
                        if self._validate_sample(loaded[i]):
                            sample_index.append({'file': file_path, 'index': i})
                            total_samples += 1
                            if self.max_samples and total_samples >= self.max_samples:
                                break
                else:
                    if self._validate_sample(loaded):
                        sample_index.append({'file': file_path, 'index': 0})
                        total_samples += 1
                        
                if self.max_samples and total_samples >= self.max_samples:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
                
        if self.max_samples and len(sample_index) > self.max_samples:
            sample_index = sample_index[:self.max_samples]
            
        return sample_index

    def _initialize_scalers(self):
        logger.info("Initializing EEG scalers")
        sample_size = min(200, max(50, len(self.sample_index) // 5))
        
        if len(self.sample_index) == 0:
            self.scalers = {}
            return
            
        sample_indices = np.random.choice(len(self.sample_index), size=min(sample_size, len(self.sample_index)), replace=False)
        region_data = {region: [] for region in self.region_indices}
        
        for idx in sample_indices:
            try:
                sample_info = self.sample_index[idx]
                sample = self._load_single_sample_uncached(sample_info['file'], sample_info['index'])
                if not self._validate_sample(sample):
                    continue
                    
                eeg = self._process_raw_eeg(sample['input_features'])
                if eeg is None:
                    continue
                    
                for region_name, indices in self.region_indices.items():
                    try:
                        region_eeg = eeg[indices].astype(np.float32)
                        region_data[region_name].append(region_eeg)
                    except (IndexError, ValueError):
                        continue
            except Exception:
                continue
                
        self.scalers = {}
        for region_name, data_list in region_data.items():
            if data_list:
                try:
                    all_region_data = np.concatenate(data_list, axis=1).T
                    self.scalers[region_name] = StandardScaler()
                    self.scalers[region_name].fit(all_region_data)
                except Exception:
                    self.scalers[region_name] = None
            else:
                self.scalers[region_name] = None
                
        gc.collect()

    def _normalize_eeg_sample(self, eeg_data):
        eeg = self._process_raw_eeg(eeg_data)
        if eeg is None:
            return None
            
        normalized_regions = []
        for region_name in ['frontal', 'temporal', 'central', 'parietal']:
            indices = self.region_indices[region_name]
            try:
                region_data = eeg[indices].astype(np.float32)
                if region_name in self.scalers and self.scalers[region_name] is not None:
                    region_data_norm = self.scalers[region_name].transform(region_data.T).T
                else:
                    mean_vals = np.mean(region_data, axis=1, keepdims=True)
                    std_vals = np.std(region_data, axis=1, keepdims=True) + self.eps
                    region_data_norm = (region_data - mean_vals) / std_vals
                    
                region_data_norm = np.clip(region_data_norm, -5.0, 5.0)
                normalized_regions.append(region_data_norm)
            except Exception:
                return None
                
        return normalized_regions
        
    def _create_fallback_sample(self):
        fallback_texts = ["数据加载问题样本", "信号处理异常样本", "临时数据替代样本", "系统错误恢复样本"]
        fallback_text = np.random.choice(fallback_texts)
        fallback_eeg = []
        
        for region_name in ['frontal', 'temporal', 'central', 'parietal']:
            n_channels = len(self.region_indices[region_name])
            pattern_type = np.random.choice(['random', 'sinusoidal', 'sparse'])
            
            if pattern_type == 'random':
                fallback_region = np.random.randn(n_channels, 125).astype(np.float32) * 0.5
            elif pattern_type == 'sinusoidal':
                t = np.linspace(0, 4*np.pi, 125)
                fallback_region = np.array([np.sin(t + i*0.5) for i in range(n_channels)]) * 0.3
            else:
                fallback_region = np.zeros((n_channels, 125))
                for ch in range(n_channels):
                    spikes = np.random.randint(0, 125, 5)
                    fallback_region[ch, spikes] = np.random.randn(5) * 0.8
                    
            fallback_eeg.append(fallback_region.astype(np.float32))
            
        tokenization = self._safe_tokenize(fallback_text)
        return {'eeg': fallback_eeg, **tokenization}
            
    def _build_region_indices(self):
        from .utils import get_electrode_regions
        electrode_regions = get_electrode_regions()
        return {
            'frontal': [i for i, ch in enumerate(self.ch_names) if ch in electrode_regions['frontal']],
            'temporal': [i for i, ch in enumerate(self.ch_names) if ch in electrode_regions['temporal']],
            'central': [i for i, ch in enumerate(self.ch_names) if ch in electrode_regions['central']],
            'parietal': [i for i, ch in enumerate(self.ch_names) if ch in electrode_regions['parietal']],
        }

    def _setup_tokenizer(self):
        if self.tokenizer.eos_token is None:
            if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token is not None:
                self.tokenizer.eos_token = self.tokenizer.sep_token
                self.tokenizer.eos_token_id = self.tokenizer.sep_token_id
            else:
                self.tokenizer.eos_token = '[EOS]'
                self.tokenizer.eos_token_id = len(self.tokenizer) - 1
                
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _get_validated_data_files(self, data_dir: str):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        all_files = os.listdir(data_dir)
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        if not pkl_files:
            raise ValueError(f"No .pkl files found in {data_dir}")
            
        file_paths = [os.path.join(data_dir, fname) for fname in pkl_files]
        file_paths.sort()
        return file_paths

    def _validate_sample(self, sample):
        if 'input_features' not in sample or 'text' not in sample:
            return False
            
        eeg_data = sample['input_features']
        if not isinstance(eeg_data, (list, np.ndarray)):
            return False
            
        try:
            eeg_array = np.array(eeg_data).squeeze(0)
            n_channels, n_time = eeg_array.shape
            if n_time < 100 or n_time > 2000:
                return False
        except Exception:
            return False
            
        text = sample.get('text', '')
        if isinstance(text, list):
            text = "".join(map(str, text))
        return isinstance(text, str)

    def _load_single_sample_uncached(self, file_path: str, sample_idx: int):
        try:
            with open(file_path, 'rb') as f:
                loaded = pickle.load(f)
                
            if isinstance(loaded, list):
                if 0 <= sample_idx < len(loaded):
                    return loaded[sample_idx]
                else:
                    return None
            else:
                return loaded if sample_idx == 0 else None
        except Exception:
            return None

    def _process_raw_eeg(self, eeg_data):
        try:
            eeg = np.array(eeg_data, dtype=np.float32)
            if eeg.ndim == 1:
                eeg = eeg.reshape(1, -1)
            elif eeg.ndim > 2:
                eeg = eeg.squeeze()
                if eeg.ndim > 2:
                    eeg = eeg.reshape(eeg.shape[0], -1)
                    
            if not np.isfinite(eeg).all():
                eeg = np.nan_to_num(eeg, nan=0.0, posinf=10.0, neginf=-10.0)
                
            return eeg
        except Exception:
            return None

    def _safe_tokenize(self, text: str):
        text = normalize_text_preserve_cjk(text)
        try:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            decoder_input_ids = input_ids.clone()
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'decoder_input_ids': decoder_input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
            }
        except Exception:
            return self._create_fallback_tokenization()

    def _create_fallback_tokenization(self):
        fallback_text = "默认样本"
        try:
            encoding = self.tokenizer(
                fallback_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            decoder_input_ids = input_ids.clone()
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'decoder_input_ids': decoder_input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
            }
        except Exception:
            seq_len = self.max_length
            return {
                'decoder_input_ids': torch.zeros(seq_len, dtype=torch.long),
                'labels': torch.full((seq_len,), -100, dtype=torch.long),
                'attention_mask': torch.zeros(seq_len, dtype=torch.long),
            }