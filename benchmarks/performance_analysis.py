"""
性能对比和优化建议
Flash Attention + KV Cache 整合效果分析
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

def performance_comparison():
    """详细性能对比分析"""
    print("Flash Attention + KV Cache 性能分析")
    print("=" * 60)
    
    config = Configs.small()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试不同序列长度
    seq_lengths = [64, 128, 256, 512]
    batch_size = 2
    
    results = {
        'seq_len': [],
        'standard_time': [],
        'flash_time': [], 
        'cache_time': [],
        'standard_memory': [],
        'flash_memory': []
    }
    
    model = MiniLLM(config)
    model.eval()
    
    for seq_len in seq_lengths:
        print(f"\n测试序列长度: {seq_len}")
        print("-" * 30)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=config.device)
        
        # 清空GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 1. 标准注意力
        with torch.no_grad():
            start_time = time.time()
            _ = model(input_ids, kv_cache=None, use_flash_attention=False)
            standard_time = time.time() - start_time
        
        # 2. Flash Attention 
        with torch.no_grad():
            start_time = time.time()
            _ = model(input_ids, kv_cache=None, use_flash_attention=True, flash_block_size=128)
            flash_time = time.time() - start_time
        
        # 3. KV Cache生成模拟
        kv_cache = KVCache(config, batch_size=batch_size)
        chunk_size = 16
        cache_total_time = 0
        
        for i in range(0, seq_len, chunk_size):
            chunk = input_ids[:, i:i+chunk_size]
            with torch.no_grad():
                start_time = time.time()
                _ = model(chunk, kv_cache=kv_cache, use_flash_attention=True)
                cache_total_time += time.time() - start_time
        
        # 内存估算 (理论值)
        standard_memory_mb = (seq_len ** 2 * batch_size * config.num_heads * 4) / (1024**2)  # 注意力矩阵
        flash_memory_mb = (128 ** 2 * batch_size * config.num_heads * 4) / (1024**2)  # Flash块
        
        results['seq_len'].append(seq_len)
        results['standard_time'].append(standard_time)
        results['flash_time'].append(flash_time)
        results['cache_time'].append(cache_total_time)
        results['standard_memory'].append(standard_memory_mb)
        results['flash_memory'].append(flash_memory_mb)
        
        print(f"标准注意力: {standard_time:.4f}s")
        print(f"Flash Attention: {flash_time:.4f}s")
        print(f"KV Cache生成: {cache_total_time:.4f}s")
        print(f"理论内存节省: {(1 - flash_memory_mb/standard_memory_mb)*100:.1f}%")
        print(f"Speed up vs 标准: {standard_time/flash_time:.2f}x")
    
    # 生成报告
    print("\n" + "=" * 60)
    print("性能总结报告")
    print("=" * 60)
    
    for i, seq_len in enumerate(seq_lengths):
        speedup_flash = results['standard_time'][i] / results['flash_time'][i]
        memory_savings = (1 - results['flash_memory'][i] / results['standard_memory'][i]) * 100
        
        print(f"序列长度 {seq_len}:")
        print(f"  Flash Attention速度提升: {speedup_flash:.2f}x")
        print(f"  内存节省: {memory_savings:.1f}%")
        print(f"  KV Cache增量推理时间: {results['cache_time'][i]:.4f}s")
    
    return results


def generation_benchmark():
    """文本生成基准测试"""
    print("\n\n文本生成性能基准")
    print("=" * 60)
    
    config = Configs.small()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MiniLLM(config)
    model.eval()
    
    # 测试不同的生成长度
    prompt_len = 32
    generation_lengths = [32, 64, 128, 256]
    
    print(f"Prompt长度: {prompt_len}")
    print(f"测试生成长度: {generation_lengths}")
    
    for max_new_tokens in generation_lengths:
        print(f"\n生成 {max_new_tokens} tokens:")
        print("-" * 30)
        
        # 准备数据
        batch_size = 1
        prompt_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=config.device)
        
        # 方法1: 无KV Cache (每次都重新计算全部)
        print("方法1: 无KV Cache (重复计算)")
        total_time_no_cache = 0
        current_seq = prompt_ids
        
        for step in range(max_new_tokens):
            next_token = torch.randint(0, config.vocab_size, (batch_size, 1), device=config.device)
            current_seq = torch.cat([current_seq, next_token], dim=1)
            
            with torch.no_grad():
                start_time = time.time()
                _ = model(current_seq, kv_cache=None, use_flash_attention=True)
                total_time_no_cache += time.time() - start_time
        
        print(f"  总时间: {total_time_no_cache:.4f}s")
        print(f"  速度: {max_new_tokens/total_time_no_cache:.2f} tokens/s")
        
        # 方法2: 使用KV Cache
        print("方法2: KV Cache (增量计算)")
        kv_cache = KVCache(config, batch_size=batch_size)
        
        # 处理prompt
        with torch.no_grad():
            start_time = time.time()
            _ = model(prompt_ids, kv_cache=kv_cache, use_flash_attention=True)
            prompt_time = time.time() - start_time
        
        # 逐个生成token
        generation_time = 0
        for step in range(max_new_tokens):
            next_token = torch.randint(0, config.vocab_size, (batch_size, 1), device=config.device)
            
            with torch.no_grad():
                start_time = time.time()
                _ = model(next_token, kv_cache=kv_cache, use_flash_attention=True)
                generation_time += time.time() - start_time
        
        total_time_with_cache = prompt_time + generation_time
        
        print(f"  Prompt时间: {prompt_time:.4f}s")
        print(f"  生成时间: {generation_time:.4f}s") 
        print(f"  总时间: {total_time_with_cache:.4f}s")
        print(f"  速度: {max_new_tokens/generation_time:.2f} tokens/s")
        
        # 计算加速比
        speedup = total_time_no_cache / total_time_with_cache
        print(f"  加速比: {speedup:.2f}x")
        
    return True


def memory_analysis():
    """内存使用分析"""
    print("\n\n内存使用分析")
    print("=" * 60)
    
    config = Configs.small()
    
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    print("理论内存使用对比 (MB):")
    print("序列长度\t标准注意力\tFlash Attention\t节省")
    print("-" * 50)
    
    for seq_len in seq_lengths:
        # 注意力矩阵内存: batch_size * num_heads * seq_len^2 * 4 bytes
        standard_memory = (2 * config.num_heads * seq_len * seq_len * 4) / (1024**2)
        
        # Flash Attention块内存: batch_size * num_heads * block_size^2 * 4 bytes  
        block_size = 128
        flash_memory = (2 * config.num_heads * block_size * block_size * 4) / (1024**2)
        
        savings = (1 - flash_memory / standard_memory) * 100
        
        print(f"{seq_len}\t\t{standard_memory:.2f}\t\t{flash_memory:.2f}\t\t{savings:.1f}%")
    
    # KV Cache内存
    print(f"\nKV Cache内存使用:")
    kv_memory = (config.num_layers * 2 * config.max_batch_size * config.num_heads * 
                config.max_kv_cache_len * config.head_dim * 2) / (1024**2)  # 2 bytes for fp16
    print(f"KV Cache总内存: {kv_memory:.2f} MB")
    print(f"支持最大序列长度: {config.max_kv_cache_len}")
    print(f"支持最大批次大小: {config.max_batch_size}")
    
    return True


def optimization_recommendations():
    """优化建议"""
    print("\n\n优化建议")
    print("=" * 60)
    
    recommendations = [
        "1. Flash Attention块大小优化:",
        "   - 短序列(< 256): 使用块大小64-128",
        "   - 长序列(> 256): 使用块大小128-256", 
        "   - 根据GPU内存调整块大小",
        "",
        "2. KV Cache策略:",
        "   - 生成任务: 必须使用KV Cache",
        "   - 批处理推理: 根据序列长度决定",
        "   - 内存充足时增大max_kv_cache_len",
        "",
        "3. 数值稳定性改进:",
        "   - 考虑使用bf16而非fp16",
        "   - 对关键计算使用fp32", 
        "   - 调整温度参数以避免数值溢出",
        "",
        "4. 进一步优化方向:",
        "   - 实现grouped-query attention",
        "   - 添加rotary position embedding",
        "   - 考虑使用torch.compile优化",
        "   - 实现custom CUDA kernels"
    ]
    
    for rec in recommendations:
        print(rec)
    
    return True


if __name__ == "__main__":
    print("开始性能分析...")
    
    try:
        # 性能对比
        performance_comparison()
        
        # 生成基准
        generation_benchmark()
        
        # 内存分析  
        memory_analysis()
        
        # 优化建议
        optimization_recommendations()
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("✅ Flash Attention + KV Cache 整合成功")
        print("🚀 推荐在生成任务中使用KV Cache")
        print("💾 建议根据GPU内存调整块大小")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
