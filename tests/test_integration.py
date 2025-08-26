"""
测试整合后的MiniLLM，验证Flash Attention和KV Cache的功能
"""
import torch
import time
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.mini_llm import MiniLLM
from cache.kv_cache import KVCache
from config import Configs

def test_flash_attention_integration():
    """测试Flash Attention和KV Cache的整合"""
    print("测试Flash Attention + KV Cache整合")
    print("=" * 50)
    
    # 使用小配置进行测试
    config = Configs.small()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {config.device}")
    
    # 创建模型
    model = MiniLLM(config)
    model.eval()
    
    # 创建KV Cache
    kv_cache = KVCache(config, batch_size=2)
    
    # 测试数据
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=config.device)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"模型配置: dim={config.dim_model}, heads={config.num_heads}, layers={config.num_layers}")
    
    # 测试1: 不使用KV Cache的Flash Attention
    print("\n测试1: Flash Attention (无KV Cache)")
    with torch.no_grad():
        start_time = time.time()
        output1 = model(input_ids, kv_cache=None, use_flash_attention=True)
        flash_time = time.time() - start_time
    print(f"输出形状: {output1.shape}")
    print(f"Flash Attention时间: {flash_time:.4f}s")
    
    # 测试2: 标准注意力（无KV Cache）
    print("\n测试2: 标准注意力 (无KV Cache)")
    with torch.no_grad():
        start_time = time.time()
        output2 = model(input_ids, kv_cache=None, use_flash_attention=False)
        standard_time = time.time() - start_time
    print(f"输出形状: {output2.shape}")
    print(f"标准注意力时间: {standard_time:.4f}s")
    
    # 比较输出差异
    max_diff = torch.max(torch.abs(output1 - output2)).item()
    mean_diff = torch.mean(torch.abs(output1 - output2)).item()
    print(f"\nFlash vs 标准注意力比较:")
    print(f"最大差异: {max_diff:.8f}")
    print(f"平均差异: {mean_diff:.8f}")
    print(f"速度比: {standard_time/flash_time:.2f}x")
    
    # 测试3: Flash Attention + KV Cache
    print("\n测试3: Flash Attention + KV Cache")
    kv_cache.reset()  # 重置cache
    
    # 分批输入模拟推理
    chunk_size = 16
    all_outputs = []
    
    total_time = 0
    for i in range(0, seq_len, chunk_size):
        chunk = input_ids[:, i:i+chunk_size]
        
        with torch.no_grad():
            start_time = time.time()
            chunk_output = model(chunk, kv_cache=kv_cache, use_flash_attention=True)
            chunk_time = time.time() - start_time
            total_time += chunk_time
            
        all_outputs.append(chunk_output)
        print(f"处理块 {i//chunk_size + 1}: 形状={chunk.shape}, 时间={chunk_time:.4f}s")
    
    # 拼接输出
    cached_output = torch.cat(all_outputs, dim=1)
    print(f"KV Cache总时间: {total_time:.4f}s")
    print(f"KV Cache输出形状: {cached_output.shape}")
    
    # 比较KV Cache输出与原始输出
    cache_diff = torch.max(torch.abs(cached_output - output1)).item()
    cache_mean_diff = torch.mean(torch.abs(cached_output - output1)).item()
    print(f"\nKV Cache vs 原始输出比较:")
    print(f"最大差异: {cache_diff:.8f}")
    print(f"平均差异: {cache_mean_diff:.8f}")
    
    # 检查cache状态
    print(f"\nKV Cache状态:")
    for b in range(batch_size):
        print(f"Batch {b} 序列长度: {kv_cache.get_seq_len(b)}")
    
    return {
        'flash_time': flash_time,
        'standard_time': standard_time,
        'cache_time': total_time,
        'flash_vs_standard_diff': max_diff,
        'cache_vs_original_diff': cache_diff
    }


def test_generation_simulation():
    """模拟生成过程的测试"""
    print("\n\n模拟文本生成过程")
    print("=" * 50)
    
    config = Configs.small()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MiniLLM(config)
    model.eval()
    
    # 初始prompt
    batch_size = 1
    prompt_len = 32
    max_new_tokens = 64
    
    prompt_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=config.device)
    
    print(f"初始prompt长度: {prompt_len}")
    print(f"最大生成tokens: {max_new_tokens}")
    
    # 使用KV Cache进行生成
    kv_cache = KVCache(config, batch_size=batch_size)
    
    generation_times = []
    all_tokens = [prompt_ids]
    
    # 首次前向传播（prompt处理）
    with torch.no_grad():
        start_time = time.time()
        _ = model(prompt_ids, kv_cache=kv_cache, use_flash_attention=True)
        prompt_time = time.time() - start_time
    
    print(f"Prompt处理时间: {prompt_time:.4f}s")
    
    # 逐个token生成
    for step in range(max_new_tokens):
        # 模拟下一个token（实际中这里会是概率采样）
        next_token = torch.randint(0, config.vocab_size, (batch_size, 1), device=config.device)
        
        with torch.no_grad():
            start_time = time.time()
            logits = model(next_token, kv_cache=kv_cache, use_flash_attention=True)
            step_time = time.time() - start_time
            
        generation_times.append(step_time)
        all_tokens.append(next_token)
        
        if step < 5 or step % 10 == 0:  # 只打印前几步和每10步
            print(f"Step {step+1}: 时间={step_time:.4f}s, Cache长度={kv_cache.get_seq_len(0)}")
    
    # 统计
    avg_generation_time = sum(generation_times) / len(generation_times)
    total_time = prompt_time + sum(generation_times)
    tokens_per_sec = max_new_tokens / sum(generation_times)
    
    print(f"\n生成统计:")
    print(f"平均每token时间: {avg_generation_time:.4f}s")
    print(f"总时间: {total_time:.4f}s")
    print(f"生成速度: {tokens_per_sec:.2f} tokens/s")
    print(f"最终序列长度: {kv_cache.get_seq_len(0)}")
    
    return {
        'prompt_time': prompt_time,
        'avg_generation_time': avg_generation_time,
        'tokens_per_sec': tokens_per_sec,
        'total_time': total_time
    }


def benchmark_different_block_sizes():
    """测试不同Flash Attention块大小的性能"""
    print("\n\n不同块大小性能测试")
    print("=" * 50)
    
    config = Configs.small()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MiniLLM(config)
    model.eval()
    
    batch_size = 2
    seq_len = 256
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=config.device)
    
    block_sizes = [32, 64, 128, 256]
    results = {}
    
    for block_size in block_sizes:
        print(f"\n测试块大小: {block_size}")
        
        # 多次运行取平均
        times = []
        for _ in range(5):
            with torch.no_grad():
                start_time = time.time()
                _ = model(input_ids, kv_cache=None, use_flash_attention=True, 
                         flash_block_size=block_size)
                times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
        
        results[block_size] = avg_time
        print(f"平均时间: {avg_time:.4f}s (±{std_time:.4f}s)")
    
    # 找最优块大小
    best_block_size = min(results.keys(), key=lambda x: results[x])
    print(f"\n最优块大小: {best_block_size} (时间: {results[best_block_size]:.4f}s)")
    
    return results


if __name__ == "__main__":
    print("开始整合测试...")
    
    try:
        # 主要功能测试
        main_results = test_flash_attention_integration()
        
        # 生成模拟测试
        gen_results = test_generation_simulation()
        
        # 块大小测试
        block_results = benchmark_different_block_sizes()
        
        print("\n\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"Flash vs 标准注意力差异: {main_results['flash_vs_standard_diff']:.8f}")
        print(f"KV Cache vs 原始输出差异: {main_results['cache_vs_original_diff']:.8f}")
        print(f"生成速度: {gen_results['tokens_per_sec']:.2f} tokens/s")
        print(f"最优Flash块大小: {min(block_results.keys(), key=lambda x: block_results[x])}")
        
        # 检查数值稳定性
        if main_results['flash_vs_standard_diff'] < 1e-4 and main_results['cache_vs_original_diff'] < 1e-4:
            print("✅ 所有测试通过！数值稳定性良好。")
        else:
            print("⚠️ 存在数值差异，需要进一步检查。")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
