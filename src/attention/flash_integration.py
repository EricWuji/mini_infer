"""
Flash Attention集成模块

将Triton实现的Flash Attention 2集成到MiniLLM的前向传播中
支持KV Cache加速
"""

import torch
import torch.nn.functional as F
from typing import Optional

from .flash_attn2 import flash_attention_2
from cache.kv_cache import KVCache


def flash_attention_with_kv_cache(
    x: torch.Tensor,
    attn_layer,
    kv_cache: Optional[KVCache] = None,
    layer_idx: int = 0,
    cache_position: Optional[torch.Tensor] = None,
    causal: bool = True
) -> torch.Tensor:
    """
    使用Triton Flash Attention 2 + KV Cache的集成函数
    
    Args:
        x: 输入张量 [batch_size, seq_len, dim_model]
        attn_layer: PyTorch的MultiheadAttention层
        kv_cache: KV Cache对象
        layer_idx: 当前层索引
        cache_position: 缓存位置，如果为None则自动计算
        causal: 是否使用因果掩码
        
    Returns:
        attention输出 [batch_size, seq_len, dim_model]
    """
    batch_size, seq_len, dim_model = x.shape
    device = x.device
    
    # 获取注意力参数
    num_heads = attn_layer.num_heads
    head_dim = attn_layer.head_dim if hasattr(attn_layer, 'head_dim') else dim_model // num_heads
    
    # 计算Q, K, V
    if attn_layer._qkv_same_embed_dim:
        # 使用合并的权重矩阵
        x_flat = x.reshape(-1, dim_model)
        qkv = F.linear(x_flat, attn_layer.in_proj_weight, attn_layer.in_proj_bias)
        qkv = qkv.reshape(batch_size, seq_len, 3, dim_model)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.squeeze(2).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.squeeze(2).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.squeeze(2).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    else:
        # 使用分离的权重矩阵
        x_flat = x.reshape(-1, dim_model)
        q = F.linear(x_flat, attn_layer.q_proj_weight, 
                    attn_layer.in_proj_bias[:dim_model] if attn_layer.in_proj_bias is not None else None)
        k = F.linear(x_flat, attn_layer.k_proj_weight,
                    attn_layer.in_proj_bias[dim_model:2*dim_model] if attn_layer.in_proj_bias is not None else None)
        v = F.linear(x_flat, attn_layer.v_proj_weight,
                    attn_layer.in_proj_bias[2*dim_model:] if attn_layer.in_proj_bias is not None else None)
        
        q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # 处理KV Cache
    if kv_cache is not None:
        # 更新KV Cache
        k_cached, v_cached = kv_cache.update(
            layer_idx=layer_idx,
            k=k,
            v=v,
            cache_position=cache_position
        )
        
        # 对于KV cache场景，当前query长度和cached kv长度不匹配
        # Flash Attention 2的causal masking可能无法正确处理这种情况
        # 因此在这种场景下我们回退到标准attention
        
        cached_seq_len = k_cached.size(2)  # [batch, heads, seq_len, head_dim]
        query_seq_len = q.size(2)
        
        if query_seq_len != cached_seq_len:
            # 使用标准attention计算，正确处理causal mask
            scale = 1.0 / (head_dim ** 0.5)
            scores = torch.matmul(q, k_cached.transpose(-2, -1)) * scale
            
            # 创建正确的causal mask
            # query tokens的位置是 [cached_seq_len - query_seq_len : cached_seq_len]
            causal_mask = torch.tril(torch.ones(query_seq_len, cached_seq_len, 
                                               device=q.device, dtype=torch.bool))
            
            # 只有当query position >= key position时才能attend
            query_positions = torch.arange(cached_seq_len - query_seq_len, cached_seq_len, 
                                         device=q.device)[:, None]
            key_positions = torch.arange(cached_seq_len, device=q.device)[None, :]
            causal_mask = query_positions >= key_positions
            
            # Apply causal mask
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            
            # Softmax and apply to values
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v_cached)
        else:
            # 相同长度时可以使用Flash Attention
            attn_output, _, _ = flash_attention_2(
                q=q,
                k=k_cached, 
                v=v_cached,
                causal=causal
            )
    else:
        # 没有KV Cache时，直接使用Flash Attention
        attn_output, _, _ = flash_attention_2(
            q=q,
            k=k,
            v=v,
            causal=causal
        )
    
    # 转换回原始形状 [batch_size, seq_len, dim_model]
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim_model)
    
    # 应用输出投影
    attn_output = attn_layer.out_proj(attn_output)
    
    return attn_output


def create_flash_attention_layer(
    dim_model: int,
    num_heads: int,
    dropout: float = 0.0,
    device: str = "cuda",
    dtype=torch.float16
):
    """
    创建一个优化的Flash Attention层
    
    Args:
        dim_model: 模型维度
        num_heads: 注意力头数
        dropout: dropout比率
        device: 设备
        dtype: 数据类型
        
    Returns:
        MultiheadAttention层
    """
    return torch.nn.MultiheadAttention(
        embed_dim=dim_model,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
        device=device,
        dtype=dtype
    )


class FlashAttentionModule(torch.nn.Module):
    """
    封装Flash Attention的模块，可以直接替代标准的MultiheadAttention
    """
    
    def __init__(self, dim_model: int, num_heads: int, dropout: float = 0.0, 
                 device: str = "cuda", dtype=torch.float16):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.dropout = dropout
        
        # 创建线性层用于Q, K, V投影
        self.q_proj = torch.nn.Linear(dim_model, dim_model, bias=True, device=device, dtype=dtype)
        self.k_proj = torch.nn.Linear(dim_model, dim_model, bias=True, device=device, dtype=dtype)
        self.v_proj = torch.nn.Linear(dim_model, dim_model, bias=True, device=device, dtype=dtype)
        self.out_proj = torch.nn.Linear(dim_model, dim_model, bias=True, device=device, dtype=dtype)
        
        # 为了与PyTorch的MultiheadAttention保持兼容
        self._qkv_same_embed_dim = False
        self.q_proj_weight = self.q_proj.weight
        self.k_proj_weight = self.k_proj.weight
        self.v_proj_weight = self.v_proj.weight
        
    def forward(self, 
                x: torch.Tensor,
                kv_cache: Optional[KVCache] = None,
                layer_idx: int = 0,
                cache_position: Optional[torch.Tensor] = None,
                causal: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, dim_model]
            kv_cache: KV Cache对象
            layer_idx: 层索引
            cache_position: 缓存位置
            causal: 是否使用因果掩码
            
        Returns:
            输出张量 [batch_size, seq_len, dim_model]
        """
        return flash_attention_with_kv_cache(
            x=x,
            attn_layer=self,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            cache_position=cache_position,
            causal=causal
        )
