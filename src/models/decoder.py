# decoder.py - FIXED VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import logging

logger = logging.getLogger(__name__)

try:
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available - LoRA will be disabled")

try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

class EnhancedProjection(nn.Module):
    """Enhanced projection network with residual connections and layer normalization"""
    def __init__(self, hidden_dim, model_dim, prefix_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.prefix_len = prefix_len
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, model_dim * 4),
            nn.LayerNorm(model_dim * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(model_dim * 4, model_dim * 2),
            nn.LayerNorm(model_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(model_dim, prefix_len * model_dim)
        )
        
        # Residual connection path
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, model_dim),
            nn.LayerNorm(model_dim)
        ) if hidden_dim != model_dim else nn.Identity()
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0) 
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        residual = self.residual(x)
        main_output = self.projection(x)
        
        # Add residual to the main output (after reshaping if needed)
        batch_size = x.size(0)
        main_reshaped = main_output.view(batch_size, self.prefix_len, self.model_dim)
        residual_expanded = residual.unsqueeze(1).expand(-1, self.prefix_len, -1)
        
        output = main_reshaped + residual_expanded
        return output

class Decoder(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        pretrained_model: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',  # Using smaller model for stability
        decoder_prefix_len: int = 16,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pretrained_model = pretrained_model
        self.decoder_prefix_len = decoder_prefix_len
        self.use_lora = use_lora and PEFT_AVAILABLE
        self.device = device
        
        logger.info(f"Initializing Enhanced Decoder with {pretrained_model}")
        
        # Load tokenizer with better settings
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model,
                trust_remote_code=True,
                padding_side='right'
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Setup quantization
        quantization_config = None
        if load_in_4bit and BITSANDBYTES_AVAILABLE:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization")
        elif load_in_8bit and BITSANDBYTES_AVAILABLE:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("Using 8-bit quantization")
        
        # Load base model
        try:
            logger.info("Loading base model...")
            self.llm = AutoModelForCausalLM.from_pretrained(
                pretrained_model,
                quantization_config=quantization_config,
                device_map="auto" if quantization_config else None,
                torch_dtype=torch.float16 if quantization_config else torch.float32,
                trust_remote_code=True,
                use_cache=True
            )
            
            if not quantization_config:
                self.llm = self.llm.to(device)
                
            logger.info("Base model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Model dimensions
        self.model_dim = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size
        logger.info(f"Model dimensions: hidden_size={self.model_dim}, vocab_size={self.vocab_size}")
        
        # Enhanced EEG to prefix projection
        self.eeg_to_prefix = EnhancedProjection(
            hidden_dim=hidden_dim,
            model_dim=self.model_dim,
            prefix_len=decoder_prefix_len
        )
        
        # FREEZE THE BASE MODEL COMPLETELY first
        self._freeze_base_model()
        
        # Setup LoRA with configuration
        if self.use_lora:
            success = self._setup_enhanced_lora(lora_rank, lora_alpha, lora_dropout, quantization_config is not None)
            if not success:
                logger.warning("LoRA setup failed, falling back to partial unfreezing")
                self._unfreeze_enhanced_layers()
        else:
            self._unfreeze_enhanced_layers()
        
        self.eeg_to_prefix = self.eeg_to_prefix.to(device)
        
        # Final parameter count and verification
        self._log_enhanced_parameter_count()

    def _freeze_base_model(self):
        """Freeze all parameters of the base model"""
        for param in self.llm.parameters():
            param.requires_grad = False
        logger.info("Base model frozen")

    def _setup_enhanced_lora(self, rank, alpha, dropout, is_quantized):
        """Enhanced LoRA setup with better target modules"""
        try:
            logger.info("Setting up Enhanced LoRA...")
            
            # Prepare model for k-bit training if quantized
            if is_quantized:
                logger.info("Preparing model for k-bit training...")
                self.llm = prepare_model_for_kbit_training(self.llm)
            
            # Enhanced target modules for Qwen2.5
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
            
            logger.info(f"Using Enhanced LoRA target modules: {target_modules}")
            
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA
            self.llm = get_peft_model(self.llm, lora_config)
            
            # Enhanced verification
            lora_param_count = self._count_enhanced_lora_parameters()
            
            logger.info(f"LoRA parameters: {lora_param_count:,}")
            
            if lora_param_count > 1000: 
                logger.info("✓ Enhanced LoRA successfully applied")
                return True
            else:
                logger.error("✗ LoRA applied but insufficient parameters found")
                return False
                
        except Exception as e:
            logger.error(f"Enhanced LoRA setup failed: {e}")
            return False

    def _count_enhanced_lora_parameters(self):
        """Count only LoRA parameters with better detection"""
        lora_params = 0
        for name, param in self.llm.named_parameters():
            if param.requires_grad and ('lora' in name.lower() or 'lora_' in name):
                lora_params += param.numel()
        return lora_params

    def _unfreeze_enhanced_layers(self):
        """Enhanced partial unfreezing"""
        logger.info("Using enhanced partial unfreezing")
        
        layers_unfrozen = 0
        
        # Unfreeze embeddings
        if hasattr(self.llm.model, 'embed_tokens'):
            for param in self.llm.model.embed_tokens.parameters():
                param.requires_grad = True
            layers_unfrozen += 1
        
        # Unfreeze last 4 layers 
        if hasattr(self.llm.model, 'layers'):
            layers = self.llm.model.layers
            total_layers = len(layers)
            layers_to_unfreeze = min(4, total_layers)
            
            for i in range(total_layers - layers_to_unfreeze, total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
            layers_unfrozen += layers_to_unfreeze
            logger.info(f"Unfrozen last {layers_to_unfreeze} transformer layers")
        
        # Unfreeze LM head and final layers
        if hasattr(self.llm, 'lm_head'):
            for param in self.llm.lm_head.parameters():
                param.requires_grad = True
            layers_unfrozen += 1
        
        logger.info(f"Total layers unfrozen: {layers_unfrozen}")

    def _log_enhanced_parameter_count(self):
        """Enhanced parameter counting with better categorization"""
        total_params = 0
        trainable_params = 0
        
        categories = {
            'eeg_projection': 0,
            'lora': 0,
            'transformer': 0,
            'embeddings': 0,
            'lm_head': 0
        }
        
        # Count EEG projection
        for p in self.eeg_to_prefix.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
                categories['eeg_projection'] += p.numel()
        
        # Count LLM parameters
        for name, p in self.llm.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
                
                # Categorize
                if 'lora' in name.lower():
                    categories['lora'] += p.numel()
                elif 'embed' in name:
                    categories['embeddings'] += p.numel()
                elif 'lm_head' in name:
                    categories['lm_head'] += p.numel()
                elif any(layer_name in name for layer_name in ['layers', 'blocks']):
                    categories['transformer'] += p.numel()
        
        logger.info("=" * 60)
        logger.info("ENHANCED PARAMETER BREAKDOWN:")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Training ratio: {trainable_params/total_params*100:.4f}%")
        logger.info("\nTrainable by category:")
        for category, count in categories.items():
            if count > 0:
                logger.info(f"  {category}: {count:,} ({count/trainable_params*100:.1f}%)")
        logger.info("=" * 60)
        
    def create_eeg_conditioned_embeddings(self, eeg_feat):
        """Create EEG-conditioned embeddings with enhanced projection"""
        prefix_embeds = self.eeg_to_prefix(eeg_feat)
        return prefix_embeds
        
    def forward(self, eeg_feat, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch_size = eeg_feat.shape[0]
        device = eeg_feat.device
        
        eeg_prefix = self.create_eeg_conditioned_embeddings(eeg_feat)
        
        if input_ids is not None:
            text_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([eeg_prefix, text_embeds], dim=1)
            
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=device)
            prefix_mask = torch.ones(batch_size, self.decoder_prefix_len, device=device, dtype=attention_mask.dtype)
            full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, self.decoder_prefix_len), -100, device=device, dtype=labels.dtype
                )
                full_labels = torch.cat([prefix_labels, labels], dim=1)
            else:
                full_labels = None
                
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels,
                return_dict=True
            )
            
        else:
            inputs_embeds = eeg_prefix
            full_attention_mask = torch.ones(batch_size, self.decoder_prefix_len, device=device)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            full_labels = None

        return {
            'logits': outputs.logits,
            'loss': outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else torch.tensor(0.0, device=device),
            'hidden_states': getattr(outputs, 'hidden_states', None)
        }
    
    def generate(self, eeg_feat, generation_config=None, **kwargs):
        """Enhanced generation with better default parameters"""
        
        with torch.no_grad():
            eeg_prefix = self.create_eeg_conditioned_embeddings(eeg_feat)
            
            if generation_config is None:
                generation_config = GenerationConfig(
                    max_new_tokens=kwargs.get('max_new_tokens', 20),  
                    do_sample=kwargs.get('do_sample', True),
                    temperature=kwargs.get('temperature', 0.8),  # Adjusted for better diversity
                    top_p=kwargs.get('top_p', 0.9),
                    top_k=kwargs.get('top_k', 40),
                    repetition_penalty=kwargs.get('repetition_penalty', 1.5),  # Increased to reduce repetition
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            generated_ids = self.llm.generate(
                inputs_embeds=eeg_prefix,
                generation_config=generation_config,
                **kwargs
            )
            return generated_ids