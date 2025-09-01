"""
Performance comparison script for MiniLLM optimizations

This script compares:
1. Original MiniLLM
2. MiniLLM with KV Cache
3. MiniLLM with Flash Attention
4. MiniLLM with Chunked Prefill + KV Cache + Flash Attention
"""

import torch
import time
import sys
import gc

sys.path.insert(0, '/home/wuyinqi/mini_infer')

from src.config import Configs
from src.models.mini_llm import MiniLLM
from src.models.chunked_prefill import ChunkedPrefillMiniLLM
from src.cache.kv_cache import KVCache


def benchmark_setup():
    """Setup benchmark configuration"""
    config = Configs.medium()
    config.num_layers = 2  # Reasonable depth for benchmarking
    config.max_seq_len = 2048
    config.chunk_size = 512
    
    return config


def run_benchmark(model, input_ids, kv_cache=None, use_flash_attention=False, 
                 use_chunked=False, num_trials=5, warmup=2):
    """Run benchmark for a specific configuration"""
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            if use_chunked and hasattr(model, 'chunked_prefill_forward'):
                _ = model.chunked_prefill_forward(
                    input_ids, 
                    kv_cache=kv_cache, 
                    use_flash_attention=use_flash_attention
                )
            else:
                _ = model.forward(
                    input_ids, 
                    kv_cache=kv_cache, 
                    use_flash_attention=use_flash_attention
                )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Actual benchmark
    times = []
    for _ in range(num_trials):
        # Reset cache for each trial if using cache
        if kv_cache is not None:
            kv_cache.reset()
        
        start_time = time.time()
        
        with torch.no_grad():
            if use_chunked and hasattr(model, 'chunked_prefill_forward'):
                output = model.chunked_prefill_forward(
                    input_ids, 
                    kv_cache=kv_cache, 
                    use_flash_attention=use_flash_attention
                )
            else:
                output = model.forward(
                    input_ids, 
                    kv_cache=kv_cache, 
                    use_flash_attention=use_flash_attention
                )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append(time.time() - start_time)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'output_shape': output.shape,
        'memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    }


def main():
    print("MiniLLM Performance Comparison")
    print("=" * 60)
    
    config = benchmark_setup()
    print(f"Configuration:")
    print(f"  - Model dimension: {config.dim_model}")
    print(f"  - Number of heads: {config.num_heads}")
    print(f"  - Number of layers: {config.num_layers}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Max sequence length: {config.max_seq_len}")
    print(f"  - Chunk size: {config.chunk_size}")
    print(f"  - Device: {config.device}")
    print()
    
    # Test configurations
    test_configs = [
        {
            'name': 'Baseline (Regular Attention)',
            'use_kv_cache': False,
            'use_flash_attention': False,
            'use_chunked': False
        },
        {
            'name': 'With KV Cache',
            'use_kv_cache': True,
            'use_flash_attention': False,
            'use_chunked': False
        },
        {
            'name': 'With Flash Attention',
            'use_kv_cache': False,
            'use_flash_attention': True,
            'use_chunked': False
        },
        {
            'name': 'With KV Cache + Flash Attention',
            'use_kv_cache': True,
            'use_flash_attention': True,
            'use_chunked': False
        },
        {
            'name': 'Chunked Prefill + KV Cache + Flash Attention',
            'use_kv_cache': True,
            'use_flash_attention': True,
            'use_chunked': True
        }
    ]
    
    # Test different sequence lengths
    sequence_lengths = [512, 1024, 1536, 2048]
    batch_size = 2
    
    results = {}
    
    for seq_len in sequence_lengths:
        print(f"Testing sequence length: {seq_len}")
        print("-" * 40)
        
        results[seq_len] = {}
        
        # Generate test input
        input_ids = torch.randint(
            0, config.vocab_size, 
            (batch_size, seq_len), 
            device=config.device
        )
        
        baseline_time = None
        
        for test_config in test_configs:
            config_name = test_config['name']
            
            try:
                # Create appropriate model
                if test_config['use_chunked']:
                    model = ChunkedPrefillMiniLLM(config, chunk_size=config.chunk_size)
                else:
                    model = MiniLLM(config)
                
                model.eval()
                
                # Create KV cache if needed
                kv_cache = None
                if test_config['use_kv_cache']:
                    kv_cache = KVCache(config, batch_size=batch_size)
                
                # Run benchmark
                result = run_benchmark(
                    model=model,
                    input_ids=input_ids,
                    kv_cache=kv_cache,
                    use_flash_attention=test_config['use_flash_attention'],
                    use_chunked=test_config['use_chunked'],
                    num_trials=5
                )
                
                results[seq_len][config_name] = result
                
                # Calculate speedup relative to baseline
                if baseline_time is None:
                    baseline_time = result['avg_time']
                    speedup = 1.0
                else:
                    speedup = baseline_time / result['avg_time']
                
                # Calculate throughput
                tokens_per_second = (batch_size * seq_len) / result['avg_time']
                
                print(f"  {config_name:40s}: {result['avg_time']:.4f}s "
                      f"(speedup: {speedup:.2f}x, "
                      f"throughput: {tokens_per_second:.0f} tokens/s, "
                      f"memory: {result['memory_mb']:.1f}MB)")
                
                # Clean up
                del model
                if kv_cache is not None:
                    del kv_cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  {config_name:40s}: FAILED - {str(e)}")
                results[seq_len][config_name] = {'error': str(e)}
        
        print()
    
    # Summary
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print("\nSpeedup Matrix (relative to baseline):")
    print("-" * 60)
    
    header = "Sequence Length"
    for config in test_configs[1:]:  # Skip baseline
        header += f" | {config['name'][:15]}"
    print(header)
    print("-" * len(header))
    
    for seq_len in sequence_lengths:
        if seq_len not in results:
            continue
        
        row = f"{seq_len:14d}"
        baseline_time = results[seq_len].get('Baseline (Regular Attention)', {}).get('avg_time')
        
        if baseline_time:
            for config in test_configs[1:]:  # Skip baseline
                config_name = config['name']
                if config_name in results[seq_len] and 'avg_time' in results[seq_len][config_name]:
                    current_time = results[seq_len][config_name]['avg_time']
                    speedup = baseline_time / current_time
                    row += f" | {speedup:13.2f}x"
                else:
                    row += f" | {'ERROR':>13s}"
        
        print(row)
    
    print("\nBest configuration for each sequence length:")
    print("-" * 50)
    
    for seq_len in sequence_lengths:
        if seq_len not in results:
            continue
            
        best_config = None
        best_speedup = 0
        baseline_time = results[seq_len].get('Baseline (Regular Attention)', {}).get('avg_time')
        
        if baseline_time:
            for config_name, result in results[seq_len].items():
                if 'avg_time' in result and config_name != 'Baseline (Regular Attention)':
                    speedup = baseline_time / result['avg_time']
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_config = config_name
        
        if best_config:
            print(f"  Sequence {seq_len:4d}: {best_config} ({best_speedup:.2f}x speedup)")
        else:
            print(f"  Sequence {seq_len:4d}: No valid results")
    
    print("\nRecommendations:")
    print("-" * 30)
    print("- For short sequences (<1024): Use Flash Attention with KV Cache")
    print("- For long sequences (>1024): Use Chunked Prefill with all optimizations")
    print("- Memory constrained environments: Use Chunked Prefill with smaller chunks")
    print("- Maximum throughput: Combine all optimizations (Chunked + KV Cache + Flash)")


if __name__ == "__main__":
    main()
