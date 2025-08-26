#!/usr/bin/env python3
"""
Example: How to easily upgrade your model for different hardware configurations
This demonstrates how simple it is to scale your model when you get new hardware.
"""

from config import Configs, ModelConfig
from MiniLLM import create_model
from KVCache import KVCache
import torch

def example_usage():
    print("🔧 MiniLLM Configuration System Demo")
    print("=" * 50)
    
    # Scenario 1: Development/Testing (Limited GPU memory)
    print("\n📱 Scenario 1: Development on RTX 3060 (8GB)")
    config_dev = Configs.small()
    print(f"   Context length: {config_dev.max_seq_len}")
    print(f"   Model parameters: ~{estimate_params(config_dev):,}")
    
    # Scenario 2: Production deployment (Better GPU)
    print("\n🖥️  Scenario 2: Production on RTX 4090 (24GB)")
    config_prod = Configs.medium()
    print(f"   Context length: {config_prod.max_seq_len}")
    print(f"   Model parameters: ~{estimate_params(config_prod):,}")
    
    # Scenario 3: Future upgrade (Next-gen GPU)
    print("\n🚀 Scenario 3: Future upgrade to H100 (80GB)")
    config_future = Configs.large()
    print(f"   Context length: {config_future.max_seq_len}")
    print(f"   Model parameters: ~{estimate_params(config_future):,}")
    
    # Scenario 4: Custom configuration for specific needs
    print("\n⚙️  Scenario 4: Custom configuration for research")
    config_custom = ModelConfig(
        vocab_size=32000,        # Larger vocabulary
        dim_model=1536,          # Custom model size
        num_heads=24,            # Custom head count
        max_seq_len=16384,       # 16K context
        max_kv_cache_len=32768,  # 32K cache
        max_batch_size=4         # Smaller batches for memory
    )
    print(f"   Context length: {config_custom.max_seq_len}")
    print(f"   Model parameters: ~{estimate_params(config_custom):,}")
    
    print("\n" + "=" * 50)
    print("💡 Key Benefits:")
    print("   ✅ One-line config changes")
    print("   ✅ Automatic parameter validation")
    print("   ✅ Memory usage estimation")
    print("   ✅ Backward compatibility")
    print("   ✅ Easy hardware scaling")

def estimate_params(config):
    """Rough estimation of model parameters"""
    vocab_embed = config.vocab_size * config.dim_model
    pos_embed = config.max_seq_len * config.dim_model
    
    # Per transformer layer
    attention = 4 * config.dim_model * config.dim_model  # Q, K, V, O projections
    ffn = 2 * config.dim_model * config.dim_feedforward   # Up and down projections
    layer_params = attention + ffn
    
    total_params = vocab_embed + pos_embed + (layer_params * config.num_layers)
    return total_params

def quick_test():
    """Quick test to show how easy it is to switch configurations"""
    print("\n🧪 Quick Performance Test")
    print("-" * 30)
    
    configs_to_test = [
        ("Small", Configs.small()),
        ("Medium", Configs.medium()),
    ]
    
    for name, config in configs_to_test:
        print(f"\n⏱️  Testing {name} config...")
        
        # Create model and input
        model = create_model(config)
        batch_size = min(2, config.max_batch_size)
        seq_len = min(512, config.max_seq_len)
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=config.device)
        
        # Test inference speed
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            _ = model(input_ids)
            end_time.record()
            torch.cuda.synchronize()
            
            elapsed = start_time.elapsed_time(end_time)
            print(f"   Inference time: {elapsed:.2f}ms")
            print(f"   Input shape: {input_ids.shape}")

if __name__ == "__main__":
    example_usage()
    if torch.cuda.is_available():
        quick_test()
    else:
        print("\n⚠️  CUDA not available - skipping performance test")
