import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from KVCache import KVCache
from config import ModelConfig, DEFAULT_CONFIG

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
                kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            kv_cache
        
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

        x = self.embed(input_ids) * self.dim_model ** 0.5
        # 确保位置编码不超出范围
        if start_pos + seq_len > self.max_seq_len:
            # 如果超出范围，使用循环位置编码或者截断
            pos_indices = torch.arange(start_pos, start_pos + seq_len, device=device) % self.max_seq_len
            pos = self.pos_embed[:, pos_indices, :]
        else:
            pos = self.pos_embed[:, start_pos:start_pos + seq_len, :]
        x = self.dropout(x + pos)

        # 手动实现attention with KV cache
        attn = self.decode.layers[0].self_attn
        
        # 使用权重矩阵手动计算q, k, v
        if attn._qkv_same_embed_dim:
            # 当q, k, v维度相同时，使用in_proj_weight
            x_flat = x.reshape(-1, self.dim_model)  # [batch_size * seq_len, dim_model]
            qkv = F.linear(x_flat, attn.in_proj_weight, attn.in_proj_bias)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.dim_model)
            q, k, v = qkv.chunk(3, dim=2)
            q = q.squeeze(2).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
            k = k.squeeze(2).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
            v = v.squeeze(2).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        else:
            # 分别计算q, k, v
            x_flat = x.reshape(-1, self.dim_model)
            q = F.linear(x_flat, attn.q_proj_weight, attn.in_proj_bias[:self.dim_model] if attn.in_proj_bias is not None else None)
            k = F.linear(x_flat, attn.k_proj_weight, attn.in_proj_bias[self.dim_model:2*self.dim_model] if attn.in_proj_bias is not None else None)
            v = F.linear(x_flat, attn.v_proj_weight, attn.in_proj_bias[2*self.dim_model:] if attn.in_proj_bias is not None else None)
            
            q = q.reshape(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
            v = v.reshape(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            layer_idx = 0
            # Calculate cache position for the new tokens
            cache_position = torch.arange(start_pos, start_pos + seq_len, device=device)
            k, v = kv_cache.update(layer_idx=layer_idx, k=k, v=v, cache_position=cache_position)
            # 需要调整causal_mask的大小以匹配kv_cache中的长度
            kv_seq_len = k.size(2)  # kv cache中的序列长度
            # 创建正确的因果掩码：当前query位置对之前所有位置可见
            causal_mask = torch.zeros((seq_len, kv_seq_len), device=device)
            for i in range(seq_len):
                query_pos = start_pos + i
                causal_mask[i, :query_pos + 1] = 1
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            kv_seq_len = seq_len
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)

        attn_weight = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weight = attn_weight.masked_fill(causal_mask == 0, float('-inf'))

        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_output = torch.matmul(attn_weight, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 使用out_proj层
        attn_output = attn.out_proj(attn_output)
        x = x + attn_output
        x = self.norm(x)

        # FFN部分
        ffn_output = self.decode.layers[0].linear2(
            F.gelu(self.decode.layers[0].linear1(x))
        )
        x = x + ffn_output
        x = self.norm(x)
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