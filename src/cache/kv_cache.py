import torch
from typing import Tuple, Dict, List, Optional
from ..config import ModelConfig

class KVCache:
    def __init__(self, config: ModelConfig, batch_size: Optional[int] = None):
        """
        Initialize KV Cache with model configuration
        
        Args:
            config: ModelConfig object containing all model parameters
            batch_size: Optional batch size override. If None, uses config.max_batch_size
        """
        config.validate()  # Validate configuration
        
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.max_batch_size = batch_size if batch_size is not None else config.max_batch_size
        self.max_seq_len = config.max_kv_cache_len  # Use KV cache length, not model seq len
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # Initialize cache tensors
        self.cache_k = torch.zeros(
            (self.num_layers, self.max_batch_size, self.num_heads, self.max_seq_len, self.head_dim),
            dtype=self.dtype, device=self.device
        )

        self.cache_v = torch.zeros(
            (self.num_layers, self.max_batch_size, self.num_heads, self.max_seq_len, self.head_dim),
            dtype=self.dtype, device=self.device
        )

        self.seq_len = torch.zeros(self.max_batch_size, dtype=torch.int32, device=self.device)
    
    @classmethod
    def from_legacy_params(cls, max_batch_size: int, max_seq_len: int, num_layers: int, 
                          num_heads: int, head_dim: int, dtype=torch.float16, device="cuda"):
        """
        Create KVCache from legacy parameters for backward compatibility
        """
        from ..config import ModelConfig
        config = ModelConfig(
            dim_model=num_heads * head_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            max_kv_cache_len=max_seq_len,
            max_batch_size=max_batch_size,
            dtype=dtype,
            device=device
        )
        return cls(config, batch_size=max_batch_size)
    
    def update(self, layer_idx: int,
               k: torch.Tensor,
               v: torch.Tensor,
               cache_position: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches using transformers-style implementation.
        
        Args:
            layer_idx: Layer index
            k: [batch_size, num_heads, seq_len, head_dim] (new tokens)
            v: [batch_size, num_heads, seq_len, head_dim] (new tokens)
            cache_position: [seq_len] positions in cache to update. If None, auto-calculated.
        
        Returns:
            k_out: [batch_size, num_heads, total_seq_len, head_dim]
            v_out: [batch_size, num_heads, total_seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = k.shape
        assert layer_idx < self.num_layers
        
        # Auto-calculate cache_position if not provided
        if cache_position is None:
            current_seq_lens = self.seq_len[:batch_size]
            # Assume all sequences in batch have same length (take max for safety)
            start_pos = current_seq_lens.max().item()
            cache_position = torch.arange(start_pos, start_pos + seq_len, 
                                        dtype=torch.long, device=k.device)
        
        # Update the cache using transformers-style index_copy_
        try:
            # Use index_copy_ for efficient updating (preferred method)
            self.cache_k[layer_idx, :batch_size].index_copy_(2, cache_position, k)
            self.cache_v[layer_idx, :batch_size].index_copy_(2, cache_position, v)
        except (NotImplementedError, RuntimeError):
            # Fallback for devices like MPS where index_copy_ might not be supported
            self.cache_k[layer_idx, :batch_size, :, cache_position] = k
            self.cache_v[layer_idx, :batch_size, :, cache_position] = v
        
        # Update sequence lengths
        if cache_position is not None:
            new_seq_len = cache_position[-1].item() + 1
            self.seq_len[:batch_size] = torch.full((batch_size,), new_seq_len, 
                                                 dtype=torch.int32, device=self.device)
        else:
            self.seq_len[:batch_size] += seq_len
        
        # Return the used portion of cache
        max_seq_len = self.seq_len[:batch_size].max().item()
        k_out = self.cache_k[layer_idx, :batch_size, :, :max_seq_len, :]
        v_out = self.cache_v[layer_idx, :batch_size, :, :max_seq_len, :]
        
        return k_out, v_out
    
    def get_slice(self, layer_idx: int,
                  batch_idx: int, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.cache_k[layer_idx, batch_idx, :, start: end, :]
        v = self.cache_v[layer_idx, batch_idx, :, start: end, :]
        return k, v
    
    def reset(self, batch_indices: Optional[List[int]] = None):
        """
        重置self.seq_len, 若batch_indices = None 重置所有的
        """
        if batch_indices is None:
            self.seq_len.zero_()
        else:
            self.seq_len[batch_indices] = 0
    
    def get_seq_len(self, batch_idx: int) -> int:
        """
        得到batch_idx的长度
        """
        return self.seq_len[batch_idx].item()