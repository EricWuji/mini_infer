#!/usr/bin/env python3
"""
修复后的Flash Attention集成测试
正确处理KV Cache的索引问题
"""

import torch
import sys
import os

from config import ModelConfig
from models.mini_llm import create_model
from cache.kv_cache import KVCache
import time


def test_flash_attention_integration():
    """测试Flash Attention与KV Cache的集成"""
    print("=== Flash Attention + KV Cache 集成测试 ===\n")
    
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU测试")
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda"
        dtype = torch.float16
    
    # 创建模型配置
    config = ModelConfig(
        dim_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=1000,
        max_seq_len=256,
        max_kv_cache_len=512,
        max_batch_size=2,
        device=device,
        dtype=dtype
    )
    
    print("模型配置:")
    print(f"  dim_model: {config.dim_model}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  设备: {config.device}")
    print()
    
    # 创建模型和缓存
    model = create_model(config)
    kv_cache = KVCache(config, batch_size=2)
    
    print("✓ 模型和缓存创建成功\n")
    
    # 测试1: 单步推理（不使用缓存）
    print("--- 测试1: 单步推理（不使用缓存） ---")
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    
    with torch.no_grad():
        logits = model(input_ids, kv_cache=None, use_flash_attention=True)
    
    print(f"✓ 输入形状: {input_ids.shape}")
    print(f"✓ 输出形状: {logits.shape}")
    print()
    
    # 测试2: 使用KV Cache的推理
    print("--- 测试2: 使用KV Cache的推理 ---")
    kv_cache.reset()  # 重置缓存
    
    # 第一步：输入较长的序列
    prompt = torch.randint(0, config.vocab_size, (2, 20), device=device)
    
    with torch.no_grad():
        logits1 = model(prompt, kv_cache=kv_cache, use_flash_attention=True)
    
    print(f"✓ 第一步 - 输入: {prompt.shape}, 输出: {logits1.shape}")
    print(f"✓ 缓存长度: {kv_cache.get_seq_len(0)}")
    
    # 第二步：增量推理（只输入一个token）
    next_token = torch.randint(0, config.vocab_size, (2, 1), device=device)
    
    with torch.no_grad():
        logits2 = model(next_token, kv_cache=kv_cache, use_flash_attention=True)
    
    print(f"✓ 第二步 - 输入: {next_token.shape}, 输出: {logits2.shape}")
    print(f"✓ 更新后缓存长度: {kv_cache.get_seq_len(0)}")
    print()
    
    # 测试3: 性能对比
    print("--- 测试3: 性能对比 ---")
    
    if device == "cuda":
        # 准备测试数据
        test_input = torch.randint(0, config.vocab_size, (2, 32), device=device)
        
        scenarios = [
            ("标准注意力", False, None),
            ("Flash Attention", True, None),
        ]
        
        results = {}
        
        for name, use_flash, cache in scenarios:
            print(f"测试 {name}...")
            
            # 预热
            for _ in range(5):
                with torch.no_grad():
                    _ = model(test_input, kv_cache=cache, use_flash_attention=use_flash)
            
            # 计时
            torch.cuda.synchronize()
            start_time = time.time()
            
            num_runs = 50
            for _ in range(num_runs):
                with torch.no_grad():
                    logits = model(test_input, kv_cache=cache, use_flash_attention=use_flash)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs * 1000
            throughput = test_input.numel() / (avg_time / 1000)
            
            results[name] = avg_time
            print(f"  平均时间: {avg_time:.2f} ms")
            print(f"  吞吐量: {throughput:.0f} tokens/s")
        
        if len(results) >= 2:
            speedup = results["标准注意力"] / results["Flash Attention"]
            print(f"  ⚡ Flash Attention加速比: {speedup:.2f}x")
    
    print()
    
    # 测试4: 增量生成模拟
    print("--- 测试4: 增量生成模拟 ---")
    
    if device == "cuda":
        # 重置缓存
        kv_cache.reset()
        
        # 初始prompt
        prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
        print(f"初始prompt长度: {prompt.size(1)}")
        
        with torch.no_grad():
            logits = model(prompt, kv_cache=kv_cache, use_flash_attention=True)
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)
        
        print(f"✓ 处理prompt完成，缓存长度: {kv_cache.get_seq_len(0)}")
        
        # 生成更多tokens
        generation_times = []
        for step in range(5):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                logits = model(next_token, kv_cache=kv_cache, use_flash_attention=True)
                next_token = torch.argmax(logits[:, -1:, :], dim=-1)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            step_time = (end_time - start_time) * 1000
            generation_times.append(step_time)
            
            print(f"  步骤 {step + 1}: {step_time:.2f} ms, 缓存长度: {kv_cache.get_seq_len(0)}")
        
        avg_gen_time = sum(generation_times) / len(generation_times)
        print(f"  ⚡ 平均生成时间: {avg_gen_time:.2f} ms/token")
        print(f"  ⚡ 生成速度: {1000 / avg_gen_time:.0f} tokens/s")
    
    print("\n=== 所有测试完成！ ===")
    return True


def test_correctness():
    """测试正确性：比较Flash Attention与标准注意力的输出"""
    print("\n=== 正确性测试 ===\n")
    
    if not torch.cuda.is_available():
        print("跳过正确性测试（需要CUDA）")
        return
    
    # 使用较小的模型以确保数值精度
    config = ModelConfig(
        dim_model=64,
        num_heads=4,
        num_layers=1,
        vocab_size=100,
        max_seq_len=128,
        max_kv_cache_len=256,
        max_batch_size=2,
        device="cuda",
        dtype=torch.float32  # 使用fp32以获得更好的数值精度
    )
    
    model = create_model(config)
    
    # 固定随机种子以确保结果可重现
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
    
    # 标准注意力输出
    with torch.no_grad():
        torch.manual_seed(42)  # 重置随机种子
        output_standard = model(input_ids, kv_cache=None, use_flash_attention=False)
    
    # Flash Attention输出
    with torch.no_grad():
        torch.manual_seed(42)  # 重置随机种子
        output_flash = model(input_ids, kv_cache=None, use_flash_attention=True)
    
    # 比较结果
    max_diff = torch.max(torch.abs(output_standard - output_flash)).item()
    mean_diff = torch.mean(torch.abs(output_standard - output_flash)).item()
    
    print(f"输出形状: {output_standard.shape}")
    print(f"最大差异: {max_diff:.8f}")
    print(f"平均差异: {mean_diff:.8f}")
    
    # 判断是否通过
    tolerance = 1e-3  # 允许的误差范围
    passed = max_diff < tolerance
    
    if passed:
        print(f"✓ 正确性测试通过 (误差 < {tolerance})")
    else:
        print(f"❌ 正确性测试失败 (误差 >= {tolerance})")
    
    return passed


if __name__ == "__main__":
    print("开始Flash Attention集成测试...\n")
    
    try:
        # 运行集成测试
        integration_success = test_flash_attention_integration()
        
        # 运行正确性测试
        correctness_success = test_correctness()
        
        print(f"\n{'='*50}")
        print("测试结果总结:")
        print(f"  集成测试: {'✓ 通过' if integration_success else '❌ 失败'}")
        print(f"  正确性测试: {'✓ 通过' if correctness_success else '❌ 失败'}")
        
        if integration_success and correctness_success:
            print("\n🎉 所有测试通过！Flash Attention集成成功！")
        else:
            print("\n⚠️  部分测试失败，请检查错误信息")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
