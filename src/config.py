"""
Configuration file for MiniLLM and KVCache
Centralized configuration for easy model scaling
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration class"""
    # Model architecture
    vocab_size: int = 1000
    dim_model: int = 512
    num_heads: int = 8
    dim_feedforward: int = 1024
    num_layers: int = 1
    
    # Context and sequence settings
    max_seq_len: int = 1024  # Maximum sequence length for the model
    max_kv_cache_len: int = 2048  # Maximum KV cache length (can be larger than max_seq_len)
    
    # Training/inference settings
    dropout: float = 0.1
    dtype: torch.dtype = torch.float16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Batch settings
    max_batch_size: int = 32
    
    @property
    def head_dim(self) -> int:
        """Head dimension calculated from model dimension and number of heads"""
        assert self.dim_model % self.num_heads == 0, f"dim_model ({self.dim_model}) must be divisible by num_heads ({self.num_heads})"
        return self.dim_model // self.num_heads
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.dim_model % self.num_heads == 0, "dim_model must be divisible by num_heads"
        assert self.max_kv_cache_len >= self.max_seq_len, "KV cache length should be >= max_seq_len"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.num_heads > 0, "num_heads must be positive"


# Predefined configurations for different scales
class Configs:
    """Predefined configurations for different model scales"""
    
    @staticmethod
    def small():
        """Small model configuration - suitable for testing"""
        return ModelConfig(
            vocab_size=1000,
            dim_model=512,
            num_heads=8,
            dim_feedforward=1024,
            max_seq_len=1024,
            max_kv_cache_len=2048,
            max_batch_size=16
        )
    
    @staticmethod
    def medium():
        """Medium model configuration"""
        return ModelConfig(
            vocab_size=5000,
            dim_model=1024,
            num_heads=16,
            dim_feedforward=4096,
            max_seq_len=2048,
            max_kv_cache_len=4096,
            max_batch_size=8
        )
    
    @staticmethod
    def large():
        """Large model configuration - for high-end hardware"""
        return ModelConfig(
            vocab_size=10000,
            dim_model=2048,
            num_heads=32,
            dim_feedforward=8192,
            max_seq_len=4096,
            max_kv_cache_len=8192,
            max_batch_size=4
        )
    
    @staticmethod
    def xlarge():
        """Extra large configuration - for future hardware upgrades"""
        return ModelConfig(
            vocab_size=50000,
            dim_model=4096,
            num_heads=64,
            dim_feedforward=16384,
            max_seq_len=8192,
            max_kv_cache_len=16384,
            max_batch_size=2
        )


# Default configuration
DEFAULT_CONFIG = Configs.small()
