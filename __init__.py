"""
MiniLLM - 方便导入的包装器
这个文件让用户可以从项目根目录直接导入主要组件
"""
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 重新导出主要组件以保持向后兼容性
from src.models.mini_llm import MiniLLM, create_model
from src.cache.kv_cache import KVCache  
from src.config import ModelConfig, Configs, DEFAULT_CONFIG
from src.attention.flash_attention import flash_attention_with_kv_cache, multi_head_flash_attention

__version__ = "1.0.0"
__all__ = [
    "MiniLLM", 
    "create_model",
    "KVCache",
    "ModelConfig", 
    "Configs", 
    "DEFAULT_CONFIG",
    "flash_attention_with_kv_cache",
    "multi_head_flash_attention"
]
