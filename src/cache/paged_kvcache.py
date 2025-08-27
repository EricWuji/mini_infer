import torch
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from .kv_cache import KVCache
from ..config import ModelConfig

@dataclass
class Block:
    """
    代表一个显卡memory中的block
    存储K和V两个张量，每个张量形状为 [num_heads, block_size, head_dim]
    """
    block_id: int
    k_data: torch.Tensor  # [num_heads, block_size, head_dim] 
    v_data: torch.Tensor  # [num_heads, block_size, head_dim]

class BlockAllocator:
    """
    Block分配器 - 管理GPU内存中的blocks
    """
    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 num_heads: int,
                 head_dim: int,
                 dtype=torch.float16,
                 device = "cuda"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # 预分配所有blocks
        self.blocks = []
        for i in range(num_blocks):
            k_data = torch.zeros((num_heads, block_size, head_dim), dtype=dtype, device=device)
            v_data = torch.zeros((num_heads, block_size, head_dim), dtype=dtype, device=device)
            block = Block(block_id=i, k_data=k_data, v_data=v_data)
            self.blocks.append(block)
            
        self.free_list = list(range(num_blocks))  # 可用的block id列表

    def allocate(self) -> Optional[Block]:
        """分配一个block"""
        if not self.free_list:
            return None
        block_id = self.free_list.pop()
        block = self.blocks[block_id]
        # 清零block数据
        block.k_data.zero_()
        block.v_data.zero_()
        return block

    def free(self, block_id: int):
        """释放一个block"""
        if block_id not in self.free_list and 0 <= block_id < self.num_blocks:
            self.free_list.append(block_id)

    def available_blocks(self) -> int:
        """返回可用blocks数量"""
        return len(self.free_list)
    
class PagedKVCache:
    """
    Paged KV Cache - 模拟vLLM中的paged attention机制
    与传统KVCache兼容，支持更高效的内存管理
    """  
    def __init__(self, 
                 config: ModelConfig = None,
                 block_size: int = 16,
                 num_blocks: int = 1024,
                 **kwargs):
        """
        初始化PagedKVCache
        
        Args:
            config: ModelConfig配置对象，如果为None则使用kwargs
            block_size: 每个block的token数量
            num_blocks: 总的block数量
            **kwargs: 兼容性参数(num_layers, num_heads, head_dim等)
        """
        # 处理配置
        if config is not None:
            config.validate()
            self.num_layers = config.num_layers
            self.num_heads = config.num_heads
            self.head_dim = config.head_dim
            self.dtype = config.dtype
            self.device = config.device
        else:
            # 兼容性支持 - 从kwargs获取参数
            self.num_layers = kwargs.get('num_layers', 1)
            self.num_heads = kwargs.get('num_heads', 8)
            self.head_dim = kwargs.get('head_dim', 64)
            self.dtype = kwargs.get('dtype', torch.float16)
            self.device = kwargs.get('device', "cuda")
            
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.config = config

        # 创建block分配器
        self.allocator = BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device
        )

        # 序列管理
        self.block_tables: Dict[int, List[int]] = {}  # seq_id -> block_id列表
        self.seq_lens: Dict[int, int] = {}  # seq_id -> 当前长度
        self.layer_blocks: Dict[int, Dict[int, Block]] = {}  # layer_idx -> {block_id: Block}
        
        # 初始化每层的block字典
        for layer_idx in range(self.num_layers):
            self.layer_blocks[layer_idx] = {}

    @classmethod
    def from_legacy_params(cls, num_layers: int, num_heads: int, head_dim: int,
                          block_size: int = 16, num_blocks: int = 1024,
                          dtype=torch.float16, device="cuda"):
        """从传统参数创建PagedKVCache，保持向后兼容"""
        return cls(
            config=None,
            block_size=block_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device
        )

    def _get_block_offset(self, token_pos: int) -> Tuple[int, int]:
        """
        获取token在block中的位置
        Args:
            token_pos: token的绝对位置
        Returns:
            (block_index, offset_in_block)
        """
        block_idx = token_pos // self.block_size
        offset = token_pos % self.block_size
        return block_idx, offset
    
    def allocate_sequence(self, seq_id: int, initial_len: int = 0) -> bool:
        """
        为新序列分配初始blocks
        
        Args:
            seq_id: 序列ID
            initial_len: 初始长度（prompt长度）
        Returns:
            是否分配成功
        """
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already exists")

        # 计算需要的block数量
        num_blocks_needed = max(1, (initial_len + self.block_size - 1) // self.block_size)
        
        # 检查是否有足够的blocks
        if self.allocator.available_blocks() < num_blocks_needed:
            return False

        # 分配blocks
        allocated_blocks = []
        for _ in range(num_blocks_needed):
            block = self.allocator.allocate()
            if block is None:
                # 释放已分配的blocks并返回失败
                for blk_id in allocated_blocks:
                    self.allocator.free(blk_id)
                return False
            allocated_blocks.append(block.block_id)
            
            # 为每一层注册这个block
            for layer_idx in range(self.num_layers):
                self.layer_blocks[layer_idx][block.block_id] = block

        # 记录sequence信息
        self.block_tables[seq_id] = allocated_blocks
        self.seq_lens[seq_id] = initial_len
        return True

    def append_tokens(self, seq_id: int, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> bool:
        """
        向序列追加新的K,V tokens
        
        Args:
            seq_id: 序列ID
            layer_idx: 层索引
            k: 新的K张量 [num_heads, seq_len, head_dim]
            v: 新的V张量 [num_heads, seq_len, head_dim]
        Returns:
            是否追加成功
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} does not exist")

        current_seq_len = self.seq_lens[seq_id]
        new_tokens = k.size(1)
        
        # 检查是否需要分配新的blocks
        total_len = current_seq_len + new_tokens
        blocks_needed = (total_len + self.block_size - 1) // self.block_size
        current_blocks = len(self.block_tables[seq_id])
        
        if blocks_needed > current_blocks:
            # 需要分配新blocks
            additional_blocks = blocks_needed - current_blocks
            if self.allocator.available_blocks() < additional_blocks:
                return False
                
            # 分配新blocks
            for _ in range(additional_blocks):
                block = self.allocator.allocate()
                if block is None:
                    return False
                self.block_tables[seq_id].append(block.block_id)
                # 为每层注册新block
                for l_idx in range(self.num_layers):
                    self.layer_blocks[l_idx][block.block_id] = block

        # 写入K,V数据
        block_ids = self.block_tables[seq_id]
        for token_idx in range(new_tokens):
            global_pos = current_seq_len + token_idx
            block_idx, offset = self._get_block_offset(global_pos)
            
            if block_idx >= len(block_ids):
                return False
                
            block_id = block_ids[block_idx]
            block = self.layer_blocks[layer_idx][block_id]
            
            # 写入当前token的K,V
            block.k_data[:, offset, :] = k[:, token_idx, :]
            block.v_data[:, offset, :] = v[:, token_idx, :]

        # 更新序列长度
        self.seq_lens[seq_id] = total_len
        return True

    def get_kv(self, seq_id: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取序列的完整K,V
        
        Args:
            seq_id: 序列ID  
            layer_idx: 层索引
        Returns:
            (k_tensor, v_tensor): K和V张量，形状为[num_heads, seq_len, head_dim]
        """
        if seq_id not in self.block_tables:
            raise ValueError(f"Sequence {seq_id} does not exist")

        block_ids = self.block_tables[seq_id]
        seq_len = self.seq_lens[seq_id]
        
        if seq_len == 0:
            # 返回空张量
            empty_shape = (self.num_heads, 0, self.head_dim)
            k_empty = torch.zeros(empty_shape, dtype=self.dtype, device=self.device)
            v_empty = torch.zeros(empty_shape, dtype=self.dtype, device=self.device)
            return k_empty, v_empty

        # 计算需要多少个完整blocks和最后一个block的有效长度
        full_blocks = seq_len // self.block_size
        last_block_len = seq_len % self.block_size
        
        k_chunks = []
        v_chunks = []
        
        # 收集完整blocks的数据
        for i in range(full_blocks):
            block_id = block_ids[i]
            block = self.layer_blocks[layer_idx][block_id]
            k_chunks.append(block.k_data)  # [num_heads, block_size, head_dim]
            v_chunks.append(block.v_data)
        
        # 处理最后一个不完整的block
        if last_block_len > 0 and full_blocks < len(block_ids):
            block_id = block_ids[full_blocks]
            block = self.layer_blocks[layer_idx][block_id]
            k_chunks.append(block.k_data[:, :last_block_len, :])
            v_chunks.append(block.v_data[:, :last_block_len, :])
        
        # 拼接所有chunks
        if k_chunks:
            k_full = torch.cat(k_chunks, dim=1)  # [num_heads, seq_len, head_dim]  
            v_full = torch.cat(v_chunks, dim=1)  # [num_heads, seq_len, head_dim]
        else:
            empty_shape = (self.num_heads, 0, self.head_dim)
            k_full = torch.zeros(empty_shape, dtype=self.dtype, device=self.device)
            v_full = torch.zeros(empty_shape, dtype=self.dtype, device=self.device)
        
        return k_full, v_full

    def update_kv_cache_compatible(self, seq_id: int, layer_idx: int, 
                                   k: torch.Tensor, v: torch.Tensor, 
                                   cache_position: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        与KVCache.update兼容的接口
        
        Args:
            seq_id: 序列ID
            layer_idx: 层索引
            k: [num_heads, seq_len, head_dim] 新的K
            v: [num_heads, seq_len, head_dim] 新的V  
            cache_position: 缓存位置（可选，用于兼容性）
        Returns:
            完整的K,V张量
        """
        # 如果序列不存在，先分配
        if seq_id not in self.block_tables:
            if not self.allocate_sequence(seq_id, 0):
                raise RuntimeError("Failed to allocate sequence")
        
        # 追加新tokens
        if not self.append_tokens(seq_id, layer_idx, k, v):
            raise RuntimeError("Failed to append tokens - out of memory")
        
        # 返回完整的K,V
        return self.get_kv(seq_id, layer_idx)

    def free_sequence(self, seq_id: int):
        """
        释放序列占用的所有blocks
        
        Args:
            seq_id: 序列ID
        """
        if seq_id not in self.block_tables:
            return
            
        block_ids = self.block_tables[seq_id]
        
        # 从每层移除blocks引用
        for layer_idx in range(self.num_layers):
            for block_id in block_ids:
                if block_id in self.layer_blocks[layer_idx]:
                    del self.layer_blocks[layer_idx][block_id]
        
        # 释放blocks
        for block_id in block_ids:
            self.allocator.free(block_id)
        
        # 清理序列记录
        del self.block_tables[seq_id]
        del self.seq_lens[seq_id]

    def get_seq_len(self, seq_id: int) -> int:
        """获取序列长度"""
        return self.seq_lens.get(seq_id, 0)

    def reset_sequence(self, seq_id: int):
        """重置序列（保持blocks分配但清零长度）"""
        if seq_id in self.seq_lens:
            self.seq_lens[seq_id] = 0
            # 可选：清零所有相关blocks的数据
            block_ids = self.block_tables[seq_id]
            for layer_idx in range(self.num_layers):
                for block_id in block_ids:
                    if block_id in self.layer_blocks[layer_idx]:
                        block = self.layer_blocks[layer_idx][block_id]
                        block.k_data.zero_()
                        block.v_data.zero_()

    def available_blocks(self) -> int:
        """获取可用blocks数量"""
        return self.allocator.available_blocks()

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用统计"""
        total_blocks = self.num_blocks
        used_blocks = total_blocks - self.allocator.available_blocks()
        
        return {
            'total_blocks': total_blocks,
            'used_blocks': used_blocks, 
            'free_blocks': self.allocator.available_blocks(),
            'block_size': self.block_size,
            'memory_per_block_mb': (
                2 * self.num_heads * self.block_size * self.head_dim *
                (2 if self.dtype == torch.float16 else 4)
            ) / (1024 * 1024),
            'active_sequences': len(self.block_tables)
        }


class PagedKVCacheAdapter(KVCache):
    """
    PagedKVCache的适配器，使其能够兼容原有的KVCache接口
    这允许在MiniLLM中无缝切换使用PagedKVCache
    """
    def __init__(self, config: ModelConfig, 
                 batch_size: Optional[int] = None,
                 block_size: int = 16,
                 blocks_per_seq: int = 64):
        """
        初始化PagedKVCache适配器
        
        Args:
            config: 模型配置
            batch_size: 批大小
            block_size: 每个block的token数
            blocks_per_seq: 每个序列最大block数量
        """
        # 初始化基类但不使用其cache tensors
        super().__init__(config, batch_size)
        
        # 计算总block数 = batch_size * blocks_per_seq
        total_blocks = self.max_batch_size * blocks_per_seq
        
        # 创建PagedKVCache实例
        self.paged_cache = PagedKVCache(
            config=config,
            block_size=block_size,
            num_blocks=total_blocks
        )
        
        # 为每个batch index预分配序列
        self.batch_seq_mapping = {}  # batch_idx -> seq_id
        for batch_idx in range(self.max_batch_size):
            seq_id = batch_idx  # 简单映射：batch_idx == seq_id
            self.batch_seq_mapping[batch_idx] = seq_id
            # 预分配序列
            self.paged_cache.allocate_sequence(seq_id, initial_len=0)

    def update(self, layer_idx: int,
               k: torch.Tensor,
               v: torch.Tensor,
               cache_position: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        兼容原KVCache.update接口的实现
        
        Args:
            layer_idx: 层索引
            k: [batch_size, num_heads, seq_len, head_dim]
            v: [batch_size, num_heads, seq_len, head_dim]
            cache_position: 缓存位置
            
        Returns:
            (k_out, v_out): 完整的K,V张量
        """
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        # 对每个batch分别处理
        k_outputs = []
        v_outputs = []
        
        for batch_idx in range(batch_size):
            seq_id = self.batch_seq_mapping[batch_idx]
            
            # 获取该batch的k,v [num_heads, seq_len, head_dim]
            k_batch = k[batch_idx]
            v_batch = v[batch_idx]
            
            # 使用paged cache更新
            k_full, v_full = self.paged_cache.update_kv_cache_compatible(
                seq_id, layer_idx, k_batch, v_batch, cache_position
            )
            
            k_outputs.append(k_full)
            v_outputs.append(v_full)
        
        # 拼接所有batch的结果
        if k_outputs:
            # 确保所有序列长度一致（padding到最大长度）
            max_seq_len = max(k_out.size(1) for k_out in k_outputs)
            
            padded_k_outputs = []
            padded_v_outputs = []
            
            for k_out, v_out in zip(k_outputs, v_outputs):
                current_len = k_out.size(1)
                if current_len < max_seq_len:
                    # 需要padding
                    pad_len = max_seq_len - current_len
                    k_pad = torch.zeros((num_heads, pad_len, head_dim), 
                                      dtype=k_out.dtype, device=k_out.device)
                    v_pad = torch.zeros((num_heads, pad_len, head_dim),
                                      dtype=v_out.dtype, device=v_out.device)
                    k_out = torch.cat([k_out, k_pad], dim=1)
                    v_out = torch.cat([v_out, v_pad], dim=1)
                
                padded_k_outputs.append(k_out)
                padded_v_outputs.append(v_out)
            
            # Stack成batch format [batch_size, num_heads, seq_len, head_dim]
            k_final = torch.stack(padded_k_outputs, dim=0)
            v_final = torch.stack(padded_v_outputs, dim=0)
        else:
            # 空输出
            k_final = torch.zeros((batch_size, num_heads, 0, head_dim), 
                                dtype=self.dtype, device=self.device)
            v_final = torch.zeros((batch_size, num_heads, 0, head_dim),
                                dtype=self.dtype, device=self.device)
        
        return k_final, v_final

    def get_slice(self, layer_idx: int, batch_idx: int, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定范围的K,V切片"""
        seq_id = self.batch_seq_mapping[batch_idx]
        k_full, v_full = self.paged_cache.get_kv(seq_id, layer_idx)
        
        # 截取指定范围
        k_slice = k_full[:, start:end, :]
        v_slice = v_full[:, start:end, :]
        
        return k_slice, v_slice

    def reset(self, batch_indices: Optional[List[int]] = None):
        """重置指定batch的缓存"""
        if batch_indices is None:
            batch_indices = list(range(self.max_batch_size))
        
        for batch_idx in batch_indices:
            seq_id = self.batch_seq_mapping[batch_idx]
            self.paged_cache.reset_sequence(seq_id)

    def get_seq_len(self, batch_idx: int) -> int:
        """获取指定batch的序列长度"""
        seq_id = self.batch_seq_mapping[batch_idx]
        return self.paged_cache.get_seq_len(seq_id)

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用统计"""
        return self.paged_cache.get_memory_usage()

    def available_blocks(self) -> int:
        """获取可用blocks数量"""
        return self.paged_cache.available_blocks()


def create_paged_kv_cache(config: ModelConfig, 
                         batch_size: Optional[int] = None,
                         block_size: int = 16,
                         blocks_per_seq: int = 64,
                         use_adapter: bool = True) -> KVCache:
    """
    创建PagedKVCache的工厂函数
    
    Args:
        config: 模型配置
        batch_size: 批大小
        block_size: 每个block的token数量
        blocks_per_seq: 每个序列的最大block数量
        use_adapter: 是否使用适配器（兼容KVCache接口）
        
    Returns:
        KVCache兼容的实例
    """
    if use_adapter:
        return PagedKVCacheAdapter(config, batch_size, block_size, blocks_per_seq)
    else:
        # 直接返回PagedKVCache（需要不同的接口）
        total_blocks = (batch_size or config.max_batch_size) * blocks_per_seq
        return PagedKVCache(config=config, block_size=block_size, num_blocks=total_blocks)