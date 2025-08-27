import torch
import torch.nn.functional as F
from typing import Tuple, Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from cache.kv_cache import KVCache


def flash_attention_with_kv_cache(
    q: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
    k: torch.Tensor,  # [batch_size, num_heads, kv_seq_len, head_dim] 
    v: torch.Tensor,  # [batch_size, num_heads, kv_seq_len, head_dim]
    causal_mask: Optional[torch.Tensor] = None,  # [seq_len, kv_seq_len] or None
    block_size: int = 128
) -> torch.Tensor:
    """
    Flash Attention 2实现，支持KV Cache和多头注意力
    
    Args:
        q: [batch_size, num_heads, seq_len, head_dim] 查询矩阵
        k: [batch_size, num_heads, kv_seq_len, head_dim] 键矩阵（包含历史）
        v: [batch_size, num_heads, kv_seq_len, head_dim] 值矩阵（包含历史）
        causal_mask: [seq_len, kv_seq_len] 因果掩码，None表示无掩码
        block_size: 块大小
        
    Returns:
        output: [batch_size, num_heads, seq_len, head_dim] 注意力输出
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    
    # 保存原始数据类型
    original_dtype = q.dtype
    
    scale = head_dim ** -0.5
    
    # 初始化输出张量
    output = torch.zeros_like(q, dtype=torch.float32)
    
    # 转换为float32进行计算以提高数值稳定性
    q = q.float()
    k = k.float() 
    v = v.float()
    
    # 为每个batch和head处理
    for b in range(batch_size):
        for h in range(num_heads):
            q_bh = q[b, h] * scale  # [seq_len, head_dim]
            k_bh = k[b, h]  # [kv_seq_len, head_dim]  
            v_bh = v[b, h]  # [kv_seq_len, head_dim]
            
            # 初始化输出、归一化统计量
            o_bh = torch.zeros(seq_len, head_dim, device=q.device, dtype=torch.float32)
            l_bh = torch.zeros(seq_len, device=q.device, dtype=torch.float32)  # 行和
            m_bh = torch.full((seq_len,), -torch.inf, device=q.device, dtype=torch.float32)  # 行最大值
            
            # 查询块的外循环
            for i in range(0, seq_len, block_size):
                q_i = q_bh[i:i + block_size]  # [q_block_size, head_dim]
                q_block_size = q_i.size(0)
                
                o_i = torch.zeros(q_block_size, head_dim, device=q.device, dtype=torch.float32)
                l_i = torch.zeros(q_block_size, device=q.device, dtype=torch.float32)
                m_i = torch.full((q_block_size,), -torch.inf, device=q.device, dtype=torch.float32)
                
                # 键值块的内循环
                for j in range(0, kv_seq_len, block_size):
                    k_j = k_bh[j:j + block_size]  # [k_block_size, head_dim]
                    v_j = v_bh[j:j + block_size]  # [k_block_size, head_dim]
                    k_block_size = k_j.size(0)
                    
                    # 计算注意力分数
                    s_ij = q_i @ k_j.T  # [q_block_size, k_block_size]
                    
                    # 应用因果掩码（如果提供）
                    if causal_mask is not None:
                        # 获取对应的掩码块
                        mask_block = causal_mask[i:i + q_block_size, j:j + k_block_size]
                        s_ij = s_ij.masked_fill(mask_block == 0, float('-inf'))
                    
                    # 在线softmax更新
                    m_ij = s_ij.max(dim=-1)[0]  # [q_block_size]
                    
                    # 更新行最大值
                    m_i_new = torch.maximum(m_i, m_ij)
                    
                    # 计算重缩放因子
                    alpha = torch.exp(m_i - m_i_new)  # [q_block_size]
                    
                    # 重新缩放之前的输出和统计量
                    o_i = o_i * alpha.unsqueeze(-1)
                    l_i = l_i * alpha
                    
                    # 计算当前块的注意力权重
                    p_ij = torch.exp(s_ij - m_i_new.unsqueeze(-1))  # [q_block_size, k_block_size]
                    
                    # 累加当前块的贡献
                    o_i = o_i + (p_ij @ v_j)
                    l_i = l_i + p_ij.sum(dim=-1)
                    
                    # 更新最大值
                    m_i = m_i_new
                
                # 最终归一化并存储结果
                o_bh[i:i + q_block_size] = o_i / l_i.unsqueeze(-1)
                l_bh[i:i + q_block_size] = l_i
                m_bh[i:i + q_block_size] = m_i
            
            # 存储到输出张量
            output[b, h] = o_bh
    
    return output.to(original_dtype)


def multi_head_flash_attention(
    x: torch.Tensor,  # [batch_size, seq_len, dim_model]
    attn_layer: torch.nn.Module,  # MultiheadAttention layer
    kv_cache: Optional['KVCache'] = None,
    layer_idx: int = 0,
    cache_position: Optional[torch.Tensor] = None,
    block_size: int = 128
) -> torch.Tensor:
    """
    多头Flash Attention的完整实现，集成KV Cache
    
    Args:
        x: [batch_size, seq_len, dim_model] 输入张量
        attn_layer: PyTorch的MultiheadAttention层
        kv_cache: KV Cache对象
        layer_idx: 当前层索引
        cache_position: 缓存位置
        block_size: Flash Attention的块大小
        
    Returns:
        output: [batch_size, seq_len, dim_model] 注意力输出
    """
    batch_size, seq_len, dim_model = x.shape
    num_heads = attn_layer.num_heads
    head_dim = dim_model // num_heads
    
    # 计算Q, K, V
    if attn_layer._qkv_same_embed_dim:
        # 使用统一的权重矩阵
        x_flat = x.reshape(-1, dim_model)
        qkv = F.linear(x_flat, attn_layer.in_proj_weight, attn_layer.in_proj_bias)
        qkv = qkv.reshape(batch_size, seq_len, 3, dim_model)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.squeeze(2)
        k = k.squeeze(2) 
        v = v.squeeze(2)
    else:
        # 分别计算Q, K, V
        x_flat = x.reshape(-1, dim_model)
        q = F.linear(x_flat, attn_layer.q_proj_weight, 
                    attn_layer.in_proj_bias[:dim_model] if attn_layer.in_proj_bias is not None else None)
        k = F.linear(x_flat, attn_layer.k_proj_weight,
                    attn_layer.in_proj_bias[dim_model:2*dim_model] if attn_layer.in_proj_bias is not None else None)
        v = F.linear(x_flat, attn_layer.v_proj_weight,
                    attn_layer.in_proj_bias[2*dim_model:] if attn_layer.in_proj_bias is not None else None)
        
        q = q.reshape(batch_size, seq_len, dim_model)
        k = k.reshape(batch_size, seq_len, dim_model)
        v = v.reshape(batch_size, seq_len, dim_model)
    
    # 重塑为多头格式 [batch_size, num_heads, seq_len, head_dim]
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # 更新KV Cache（如果提供）
    if kv_cache is not None:
        k, v = kv_cache.update(layer_idx=layer_idx, k=k, v=v, cache_position=cache_position)
        kv_seq_len = k.size(2)
        
        # 创建因果掩码
        if cache_position is not None:
            start_pos = cache_position[0].item()
        else:
            start_pos = kv_seq_len - seq_len
            
        causal_mask = torch.zeros((seq_len, kv_seq_len), device=x.device)
        for i in range(seq_len):
            query_pos = start_pos + i
            causal_mask[i, :query_pos + 1] = 1
    else:
        kv_seq_len = seq_len
        # 标准因果掩码
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))
    
    # 应用Flash Attention
    attn_output = flash_attention_with_kv_cache(
        q=q, k=k, v=v,
        causal_mask=causal_mask,
        block_size=block_size
    )
    
    # 重塑回原始形状并确保数据类型一致
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim_model)
    
    # 确保数据类型与模型权重一致
    original_dtype = next(attn_layer.out_proj.parameters()).dtype
    attn_output = attn_output.to(original_dtype)
    
    # 应用输出投影
    attn_output = attn_layer.out_proj(attn_output)
    
    return attn_output
