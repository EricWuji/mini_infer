import torch
import time
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cache.kv_cache import KVCache
from models.mini_llm import create_model, MiniLLM
from config import Configs, ModelConfig

def generate(model: MiniLLM, 
             input_ids: torch.Tensor, 
             max_new_tokens = 20,
             use_kv_cache = True) -> torch.Tensor:
    model.eval()
    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    if use_kv_cache:
        # Create KV cache using model's configuration
        kv_cache = KVCache(model.config, batch_size=batch_size)
    else:
        kv_cache = None

    generated = input_ids.clone()

    with torch.no_grad():
        for i in range(max_new_tokens):
            if use_kv_cache and i > 0:
                # 对于KV Cache，第一次之后只传入最新的token
                logits = model(generated[:, -1:], kv_cache=kv_cache)
            elif use_kv_cache and i == 0:
                # 第一次调用时传入完整序列
                logits = model(generated, kv_cache=kv_cache)
            else:
                # 不使用KV Cache时，每次都传入完整序列
                logits = model(generated)
            
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)

    return generated

def bench_mark(config: ModelConfig = None):
    """
    Benchmark function with configurable model size
    
    Args:
        config: Model configuration. If None, uses default small config
    """
    if config is None:
        config = Configs.small()
    
    device = config.device
    print(f"Using device: {device}")
    print(f"Model config: {config.max_seq_len} max_seq_len, {config.max_kv_cache_len} kv_cache_len")
    
    model = create_model(config).to(device=device, dtype=config.dtype)
    model = model.half()

    # Generate input based on model configuration
    batch_size = min(16, config.max_batch_size)  # Use smaller batch for testing
    seq_len = min(2048, config.max_seq_len)  # Use reasonable seq length for testing
    input_ids = torch.randint(0, min(100, config.vocab_size), (batch_size, seq_len), device=device)
    print(f"Input shape: {input_ids.shape}")
    print("Warming up...")
    _ = generate(model, input_ids, max_new_tokens=1, use_kv_cache=False)
    _ = generate(model, input_ids, max_new_tokens=1, use_kv_cache=True)

    print("\n🚀 Testing with KV Cache...")
    torch.cuda.synchronize()
    start = time.time()
    out_with_cache = generate(model, input_ids, max_new_tokens=20, use_kv_cache=True)
    torch.cuda.synchronize()
    time_with_cache = time.time() - start
    print(f"Output shape with KV Cache: {out_with_cache.shape}, Time taken: {time_with_cache:.4f} seconds")

    print("\n🚀 Testing without KV Cache...")
    torch.cuda.synchronize()
    start = time.time()
    out_without_cache = generate(model, input_ids, max_new_tokens=20, use_kv_cache=False)
    torch.cuda.synchronize()
    time_without_cache = time.time() - start
    print(f"Output shape without KV Cache: {out_without_cache.shape}, Time taken: {time_without_cache:.4f} seconds")
    print("\n✅ Verification:", torch.equal(out_with_cache, out_without_cache))

    print(f"\n💡 Speedup: {time_without_cache / time_with_cache:.2f}x faster with KV Cache")
    print(f"🔢 Outputs match: {torch.equal(out_with_cache, out_without_cache)}")

    print(f"Generated tokens (with KV): {out_with_cache[0, :10].cpu().tolist()}")
    print(f"Generated tokens (without KV): {out_without_cache[0, :10].cpu().tolist()}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing different model configurations")
    print("=" * 60)
    
    # Test small configuration
    print("\n🔹 Testing SMALL configuration:")
    bench_mark(Configs.small())
    
    print("\n🔹 Testing MEDIUM configuration:")
    bench_mark(Configs.medium())
    
    # Uncomment these for testing on high-end hardware
    # print("\n🔹 Testing LARGE configuration:")
    # bench_mark(Configs.large())
    
    # print("\n🔹 Testing XLARGE configuration:")
    # bench_mark(Configs.xlarge())
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)