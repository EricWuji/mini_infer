#!/usr/bin/env python3
"""
Easy configuration selector for MiniLLM
Run this script to quickly test different model sizes
"""

import argparse
from config import Configs, ModelConfig
from test_kvcache import bench_mark

def main():
    parser = argparse.ArgumentParser(description="Test MiniLLM with different configurations")
    parser.add_argument(
        "--config", 
        choices=["small", "medium", "large", "xlarge", "custom"], 
        default="small",
        help="Choose model configuration size"
    )
    parser.add_argument("--max-seq-len", type=int, help="Custom max sequence length")
    parser.add_argument("--max-kv-cache-len", type=int, help="Custom max KV cache length")
    parser.add_argument("--dim-model", type=int, help="Custom model dimension")
    parser.add_argument("--num-heads", type=int, help="Custom number of attention heads")
    
    args = parser.parse_args()
    
    # Get configuration
    if args.config == "small":
        config = Configs.small()
    elif args.config == "medium":
        config = Configs.medium()
    elif args.config == "large":
        config = Configs.large()
    elif args.config == "xlarge":
        config = Configs.xlarge()
    elif args.config == "custom":
        config = Configs.small()  # Start with small as base
        if args.max_seq_len:
            config.max_seq_len = args.max_seq_len
        if args.max_kv_cache_len:
            config.max_kv_cache_len = args.max_kv_cache_len
        if args.dim_model:
            config.dim_model = args.dim_model
        if args.num_heads:
            config.num_heads = args.num_heads
    
    print(f"🚀 Running benchmark with {args.config.upper()} configuration:")
    print(f"   - Max seq len: {config.max_seq_len}")
    print(f"   - Max KV cache len: {config.max_kv_cache_len}")
    print(f"   - Model dim: {config.dim_model}")
    print(f"   - Num heads: {config.num_heads}")
    print(f"   - Head dim: {config.head_dim}")
    print()
    
    bench_mark(config)

if __name__ == "__main__":
    main()
