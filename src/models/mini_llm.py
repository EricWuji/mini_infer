import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from cache.kv_cache import KVCache
from config import ModelConfig, DEFAULT_CONFIG
from attention.flash_integration import flash_attention_with_kv_cache

class MiniLLM(nn.Module):
    def __init__(self, config: ModelConfig = None):
        """
        Initialize MiniLLM with configuration
        
        Args:
            config: ModelConfig object. If None, uses DEFAULT_CONFIG
        """
        super().__init__()
        
        if config is None:
            config = DEFAULT_CONFIG
        
        config.validate()  # Validate configuration
        self.config = config
        
        # Model parameters from config
        self.dim_model = config.dim_model
        self.device = config.device
        self.dtype = config.dtype
        self.max_seq_len = config.max_seq_len
        self.num_head = config.num_heads
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size

        # Initialize layers
        self.embed = nn.Embedding(config.vocab_size, config.dim_model).to(device=config.device, dtype=config.dtype)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.dim_model, device=config.device, dtype=config.dtype))
        self.norm = nn.LayerNorm(config.dim_model).to(device=config.device, dtype=config.dtype)
        self.dropout = nn.Dropout(config.dropout)

        decode_layer = nn.TransformerDecoderLayer(
            d_model=config.dim_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            device=config.device,
            dtype=config.dtype
        )
        self.decode = nn.TransformerDecoder(decoder_layer=decode_layer, num_layers=config.num_layers)

        self.lm_head = nn.Linear(config.dim_model, config.vocab_size, bias=False).to(device=config.device, dtype=config.dtype)
    
    def forward(self, input_ids: torch.Tensor,
                kv_cache: Optional[KVCache] = None,
                use_flash_attention: bool = True,
                flash_block_size: int = 128) -> torch.Tensor:
        """
        Forward pass with integrated Flash Attention and KV Cache
        
        Args:
            input_ids: [batch_size, seq_len]
            kv_cache: KV Cache对象
            use_flash_attention: 是否使用Flash Attention
            flash_block_size: Flash Attention的块大小
        
        Returns:
            logits: [batch_size, seq_len, V]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 计算位置编码的起始位置
        if kv_cache is not None:
            start_pos = kv_cache.get_seq_len(0)  # 假设所有batch的长度相同
        else:
            start_pos = 0

        # 嵌入和位置编码
        x = self.embed(input_ids) * self.dim_model ** 0.5
        
        # 确保位置编码不超出范围
        if start_pos + seq_len > self.max_seq_len:
            # 如果超出范围，使用循环位置编码或者截断
            pos_indices = torch.arange(start_pos, start_pos + seq_len, device=device) % self.max_seq_len
            pos = self.pos_embed[:, pos_indices, :]
        else:
            pos = self.pos_embed[:, start_pos:start_pos + seq_len, :]
        
        x = self.dropout(x + pos)

        # 处理多层Transformer
        for layer_idx in range(self.config.num_layers):
            # 获取对应层
            layer = self.decode.layers[layer_idx]
            
            # 残差连接的输入
            residual = x
            
            if use_flash_attention:
                # 使用Triton Flash Attention 2 + KV Cache
                if kv_cache is not None:
                    # 计算cache position
                    cache_position = torch.arange(start_pos, start_pos + seq_len, device=device)
                else:
                    cache_position = None
                
                # 应用Flash Attention with KV Cache
                attn_output = flash_attention_with_kv_cache(
                    x=x,
                    attn_layer=layer.self_attn,
                    kv_cache=kv_cache,
                    layer_idx=layer_idx,
                    cache_position=cache_position,
                    causal=True  # 始终使用因果掩码
                )
            else:
                # 使用原始的手动attention实现（保留兼容性）
                attn = layer.self_attn
                
                # 计算q, k, v
                if attn._qkv_same_embed_dim:
                    x_flat = x.reshape(-1, self.dim_model)
                    qkv = F.linear(x_flat, attn.in_proj_weight, attn.in_proj_bias)
                    qkv = qkv.reshape(batch_size, seq_len, 3, self.dim_model)
                    q, k, v = qkv.chunk(3, dim=2)
                    q = q.squeeze(2).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
                    k = k.squeeze(2).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
                    v = v.squeeze(2).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
                else:
                    x_flat = x.reshape(-1, self.dim_model)
                    q = F.linear(x_flat, attn.q_proj_weight, attn.in_proj_bias[:self.dim_model] if attn.in_proj_bias is not None else None)
                    k = F.linear(x_flat, attn.k_proj_weight, attn.in_proj_bias[self.dim_model:2*self.dim_model] if attn.in_proj_bias is not None else None)
                    v = F.linear(x_flat, attn.v_proj_weight, attn.in_proj_bias[2*self.dim_model:] if attn.in_proj_bias is not None else None)
                    
                    q = q.reshape(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
                    k = k.reshape(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
                    v = v.reshape(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)

                # 更新KV Cache
                if kv_cache is not None:
                    cache_position = torch.arange(start_pos, start_pos + seq_len, device=device)
                    k, v = kv_cache.update(layer_idx=layer_idx, k=k, v=v, cache_position=cache_position)
                    kv_seq_len = k.size(2)
                    
                    # 创建因果掩码
                    causal_mask = torch.zeros((seq_len, kv_seq_len), device=device)
                    for i in range(seq_len):
                        query_pos = start_pos + i
                        causal_mask[i, :query_pos + 1] = 1
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                else:
                    kv_seq_len = seq_len
                    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)

                # 标准注意力计算
                attn_weight = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attn_weight = attn_weight.masked_fill(causal_mask == 0, float('-inf'))
                attn_weight = F.softmax(attn_weight, dim=-1)
                attn_output = torch.matmul(attn_weight, v)
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                attn_output = attn.out_proj(attn_output)
            
            # 残差连接和层归一化
            x = residual + attn_output
            x = layer.norm1(x)
            
            # FFN
            residual = x
            ffn_output = layer.linear2(F.gelu(layer.linear1(x)))
            x = residual + ffn_output
            x = layer.norm2(x)
        
        # 最终输出
        logits = self.lm_head(x)
        return logits
    
def create_model(config: ModelConfig = None) -> MiniLLM:
    """
    Create a MiniLLM model with specified or default configuration
    
    Args:
        config: ModelConfig object. If None, uses DEFAULT_CONFIG
    
    Returns:
        MiniLLM model instance
    """
    if config is None:
        config = DEFAULT_CONFIG
    return MiniLLM(config)