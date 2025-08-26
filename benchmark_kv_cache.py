#!/usr/bin/env python3
"""
Compare the performance of the old and new KV Cache implementations
"""

import torch
import time
from config import Configs
from MiniLLM import create_model
from KVCache import KVCache

def benchmark_kv_cache_methods():
    """比较不同KV Cache实现方法的性能"""
    print("🚀 KV Cache Implementation Comparison")
    print("=" * 60)
    
    # Setup
    config = Configs.small()
    device = config.device
    model = create_model(config).to(device=device, dtype=config.dtype)
    model.eval()
    
    # Test parameters
    batch_size = 8
    seq_len = 512
    max_new_tokens = 20
    
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    print(f"📊 Test Configuration:")
    print(f"   - Input shape: {input_ids.shape}")
    print(f"   - New tokens: {max_new_tokens}")
    print(f"   - Device: {device}")
    print()
    
    # Create KV Cache
    kv_cache = KVCache(config, batch_size=batch_size)
    
    # Test the new implementation
    print("🔥 Testing New KV Cache Implementation (with cache_position)")
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        # Warmup
        _ = model(generated, kv_cache=kv_cache)
        
        # Reset cache for actual test
        kv_cache.reset()
        
        # Timing
        start_time = time.time()
        
        for i in range(max_new_tokens):
            if i == 0:
                # First forward pass with full input
                logits = model(generated, kv_cache=kv_cache)
            else:
                # Subsequent passes with single token
                logits = model(generated[:, -1:], kv_cache=kv_cache)
            
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        
        end_time = time.time()
        
    total_time = end_time - start_time
    tokens_per_second = (max_new_tokens * batch_size) / total_time
    
    print(f"⏱️  Results:")
    print(f"   - Total time: {total_time:.4f} seconds")
    print(f"   - Tokens/second: {tokens_per_second:.2f}")
    print(f"   - Time per token: {total_time/max_new_tokens:.4f} seconds")
    print(f"   - Final sequence length: {generated.shape[1]}")
    
    # Test cache efficiency
    print(f"\n🔍 Cache Efficiency Test:")
    cache_memory_usage = kv_cache.cache_k.numel() * 4 + kv_cache.cache_v.numel() * 4  # 4 bytes per float32
    print(f"   - Cache memory usage: {cache_memory_usage / 1024 / 1024:.2f} MB")
    print(f"   - Cache utilization: {kv_cache.seq_len.max().item()}/{kv_cache.max_seq_len} tokens")

if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_kv_cache_methods()
    else:
        print("⚠️ CUDA not available - using CPU")
        benchmark_kv_cache_methods()
