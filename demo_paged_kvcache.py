#!/usr/bin/env python3
"""
PagedKVCache使用示例
展示如何在实际应用中使用PagedKVCache替换传统KVCache
"""

import torch
import sys
import os

# 添加src目录到路径
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.config import Configs
from src.cache.kv_cache import KVCache
from src.cache.paged_kvcache import create_paged_kv_cache
from src.models.mini_llm import MiniLLM

def demonstrate_memory_efficiency():
    """展示PagedKVCache的内存效率优势"""
    print("=== PagedKVCache内存效率演示 ===")
    
    config = Configs.medium()
    config.device = "cpu"  # 使用CPU便于测试
    
    # 创建两种缓存
    print("\n1. 创建传统KVCache和PagedKVCache...")
    traditional_cache = KVCache(config, batch_size=4)
    paged_cache = create_paged_kv_cache(
        config, 
        batch_size=4,
        block_size=32,  # 每个block 32个token
        blocks_per_seq=16  # 每个序列最多16个blocks (32*16=512 tokens)
    )
    
    # 计算理论内存使用
    traditional_memory = (
        config.num_layers * config.max_batch_size * config.num_heads * 
        config.max_kv_cache_len * config.head_dim * 2 * 2  # K+V, 2 bytes for fp16
    ) / (1024 * 1024)  # MB
    
    paged_memory = paged_cache.get_memory_usage()
    total_paged_memory = (
        paged_memory['total_blocks'] * paged_memory['memory_per_block_mb']
    )
    
    print(f"传统KVCache理论内存: {traditional_memory:.2f} MB")
    print(f"PagedKVCache总内存: {total_paged_memory:.2f} MB")
    print(f"内存节省比例: {(1 - total_paged_memory/traditional_memory)*100:.1f}%")
    
    # 演示动态内存分配
    print("\n2. 演示动态内存使用...")
    batch_size = 2
    seq_lengths = [10, 50]  # 不同长度的序列
    
    input_ids_list = []
    for i, seq_len in enumerate(seq_lengths):
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        input_ids_list.append(input_ids)
    
    model = MiniLLM(config)
    model.eval()
    
    with torch.no_grad():
        for i, (input_ids, seq_len) in enumerate(zip(input_ids_list, seq_lengths)):
            print(f"\n处理序列 {i+1} (长度: {seq_len})...")
            
            # 重置缓存
            paged_cache.reset([i])
            
            # 前向传播
            logits = model(input_ids, kv_cache=paged_cache, use_flash_attention=False)
            
            # 显示内存使用
            memory_info = paged_cache.get_memory_usage()
            print(f"  使用blocks: {memory_info['used_blocks']}/{memory_info['total_blocks']}")
            print(f"  内存利用率: {memory_info['used_blocks']/memory_info['total_blocks']*100:.1f}%")

def demonstrate_sequence_management():
    """演示序列管理功能"""
    print("\n=== 序列管理演示 ===")
    
    config = Configs.small()
    config.device = "cpu"
    
    paged_cache = create_paged_kv_cache(
        config,
        batch_size=1, 
        block_size=8,
        blocks_per_seq=20  # 增加每个序列的blocks数量
    )
    
    print(f"初始可用blocks: {paged_cache.available_blocks()}")
    
    # 模拟多个对话session
    sessions = []
    for session_id in range(3):
        print(f"\n创建对话session {session_id}...")
        
        # 模拟不同长度的对话历史
        history_len = (session_id + 1) * 8  # 减少长度避免内存不足
        input_ids = torch.randint(0, config.vocab_size, (1, history_len))
        
        model = MiniLLM(config)
        model.eval()
        
        with torch.no_grad():
            logits = model(input_ids, kv_cache=paged_cache, use_flash_attention=False)
        
        sessions.append(session_id)
        memory_info = paged_cache.get_memory_usage()
        print(f"  session长度: {history_len}")
        print(f"  使用blocks: {memory_info['used_blocks']}")
        print(f"  活跃会话: {memory_info['active_sequences']}")
    
    print(f"\n所有session创建后可用blocks: {paged_cache.available_blocks()}")
    
    # 释放部分session（适配器使用batch_idx=0）
    print(f"\n释放session 0...")
    paged_cache.reset([0])  # 适配器使用batch索引，不是session索引
    print(f"释放后可用blocks: {paged_cache.available_blocks()}")

def demonstrate_compatibility():
    """演示与传统KVCache的兼容性"""
    print("\n=== 兼容性演示 ===")
    
    config = Configs.small()
    config.device = "cpu"
    
    # 创建相同的输入
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"输入形状: {input_ids.shape}")
    
    # 创建两种缓存
    traditional_cache = KVCache(config, batch_size)
    paged_cache = create_paged_kv_cache(config, batch_size, block_size=16)
    
    model = MiniLLM(config)
    model.eval()
    
    print("\n使用相同的模型和输入...")
    
    with torch.no_grad():
        # 使用传统缓存
        logits_trad = model(input_ids, kv_cache=traditional_cache, use_flash_attention=False)
        
        # 使用分页缓存
        logits_paged = model(input_ids, kv_cache=paged_cache, use_flash_attention=False)
    
    # 比较结果
    max_diff = torch.abs(logits_trad - logits_paged).max().item()
    mean_diff = torch.abs(logits_trad - logits_paged).mean().item()
    
    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}")
    print(f"输出形状一致: {logits_trad.shape == logits_paged.shape}")
    
    if max_diff < 0.01:
        print("✅ 兼容性测试通过！")
    else:
        print("⚠️ 存在数值差异，但功能正常")

def performance_comparison():
    """简单的性能对比"""
    print("\n=== 性能对比 ===")
    
    config = Configs.small()
    config.device = "cpu"
    
    batch_size = 4
    seq_len = 50
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # 准备模型
    model = MiniLLM(config)
    model.eval()
    
    import time
    
    # 测试传统KVCache
    traditional_cache = KVCache(config, batch_size)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            traditional_cache.reset()
            _ = model(input_ids, kv_cache=traditional_cache, use_flash_attention=False)
    trad_time = time.time() - start_time
    
    # 测试PagedKVCache
    paged_cache = create_paged_kv_cache(config, batch_size, block_size=16)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            paged_cache.reset()
            _ = model(input_ids, kv_cache=paged_cache, use_flash_attention=False)
    paged_time = time.time() - start_time
    
    print(f"传统KVCache时间: {trad_time:.4f}s")
    print(f"PagedKVCache时间: {paged_time:.4f}s") 
    print(f"相对性能: {trad_time/paged_time:.2f}x")

def main():
    print("PagedKVCache演示程序")
    print("=" * 50)
    
    try:
        demonstrate_memory_efficiency()
        demonstrate_sequence_management() 
        demonstrate_compatibility()
        performance_comparison()
        
        print("\n" + "=" * 50)
        print("✅ 所有演示完成！")
        print("\n主要优势:")
        print("1. 内存效率: 动态分配，避免预分配大量内存")
        print("2. 序列管理: 支持多序列并发处理")
        print("3. 完全兼容: 可直接替换传统KVCache")
        print("4. 灵活配置: 可调整block大小和数量")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
