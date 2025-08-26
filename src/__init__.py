"""
MiniLLM - A lightweight LLM implementation with Flash Attention and KV Cache
"""

from .models.mini_llm import MiniLLM
from .cache.kv_cache import KVCache
from .attention.flash_attention import flash_attention_with_kv_cache
from .config import ModelConfig, Configs, DEFAULT_CONFIG

__version__ = "1.0.0"
__author__ = "EricWuji"

__all__ = [
    "MiniLLM",
    "KVCache", 
    "flash_attention_with_kv_cache",
    "ModelConfig",
    "Configs",
    "DEFAULT_CONFIG"
]
