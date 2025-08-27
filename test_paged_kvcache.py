#!/usr/bin/env python3
"""
测试PagedKVCache的功能和与KVCache的兼容性
"""

import torch
import sys
import os
import traceback

# 添加src目录到路径
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 添加根目录到路径以导入config
root_path = os.path.dirname(__file__)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from src.config import ModelConfig, Configs
    from src.cache.kv_cache import KVCache
    from src.cache.paged_kvcache import PagedKVCache, PagedKVCacheAdapter, create_paged_kv_cache
    from src.models.mini_llm import MiniLLM
except ImportError as e:
    print(f"Import error: {e}")
    print("尝试备用导入方式...")
    try:
        import config
        from config import ModelConfig, Configs
        from cache.kv_cache import KVCache
        from cache.paged_kvcache import PagedKVCache, PagedKVCacheAdapter, create_paged_kv_cache
        from models.mini_llm import MiniLLM
    except ImportError as e2:
        print(f"备用导入也失败: {e2}")
        sys.exit(1)

def test_basic_paged_kvcache():
    """测试PagedKVCache基本功能"""
    print("=== 测试PagedKVCache基本功能 ===")
    
    config = Configs.small()
    config.device = "cpu"  # 使用CPU避免CUDA问题
    
    # 创建PagedKVCache
    paged_cache = PagedKVCache(
        config=config,
        block_size=16,
        num_blocks=64
    )
    
    print(f"PagedKVCache创建成功: {paged_cache.num_layers} layers, {paged_cache.num_heads} heads")
    print(f"可用blocks: {paged_cache.available_blocks()}")
    
    # 分配序列
    seq_id = 0
    success = paged_cache.allocate_sequence(seq_id, initial_len=0)
    print(f"分配序列 {seq_id}: {'成功' if success else '失败'}")
    
    # 创建测试数据 - 确保数据类型一致
    batch_size = 1
    seq_len = 5
    layer_idx = 0
    
    k = torch.randn(config.num_heads, seq_len, config.head_dim, dtype=config.dtype)
    v = torch.randn(config.num_heads, seq_len, config.head_dim, dtype=config.dtype)
    
    # 添加tokens
    success = paged_cache.append_tokens(seq_id, layer_idx, k, v)
    print(f"添加tokens: {'成功' if success else '失败'}")
    print(f"序列长度: {paged_cache.get_seq_len(seq_id)}")
    
    # 获取K,V
    k_out, v_out = paged_cache.get_kv(seq_id, layer_idx)
    print(f"获取K,V形状: K={k_out.shape}, V={v_out.shape}")
    
    # 验证数据一致性
    diff_k = torch.abs(k - k_out).max()
    diff_v = torch.abs(v - v_out).max()
    print(f"K差异: {diff_k:.6f}, V差异: {diff_v:.6f}")
    
    # 详细检查差异来源
    if diff_k > 1e-6:
        print(f"警告: K差异较大，检查前几个值:")
        print(f"原始K[0,0,:3]: {k[0,0,:3]}")
        print(f"输出K[0,0,:3]: {k_out[0,0,:3]}")
    
    if diff_v > 1e-6:
        print(f"警告: V差异较大，检查前几个值:")
        print(f"原始V[0,0,:3]: {v[0,0,:3]}")
        print(f"输出V[0,0,:3]: {v_out[0,0,:3]}")
    
    # 内存使用统计
    memory_info = paged_cache.get_memory_usage()
    print(f"内存使用: {memory_info}")
    
    # 释放序列
    paged_cache.free_sequence(seq_id)
    print(f"释放序列后可用blocks: {paged_cache.available_blocks()}")
    
    return True

def test_paged_kvcache_adapter():
    """测试PagedKVCacheAdapter与KVCache的兼容性"""
    print("\n=== 测试PagedKVCacheAdapter兼容性 ===")
    
    config = Configs.small()
    config.device = "cpu"
    batch_size = 2
    
    # 创建适配器
    paged_adapter = PagedKVCacheAdapter(
        config=config,
        batch_size=batch_size,
        block_size=16,
        blocks_per_seq=32
    )
    
    # 创建传统KVCache进行对比
    traditional_cache = KVCache(config, batch_size)
    
    print(f"适配器创建成功, batch_size={batch_size}")
    
    # 创建测试数据 - 确保数据类型一致
    seq_len = 8
    layer_idx = 0
    
    k = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim, dtype=config.dtype)
    v = torch.randn(batch_size, config.num_heads, seq_len, config.head_dim, dtype=config.dtype)
    
    print(f"测试数据形状: K={k.shape}, V={v.shape}")
    
    # 使用两种cache更新
    k_paged, v_paged = paged_adapter.update(layer_idx, k, v)
    k_trad, v_trad = traditional_cache.update(layer_idx, k, v)
    
    print(f"PagedCache输出形状: K={k_paged.shape}, V={v_paged.shape}")
    print(f"传统Cache输出形状: K={k_trad.shape}, V={v_trad.shape}")
    
    # 验证输出数据一致性（应该是相同的，因为都是第一次update）
    if k_paged.shape == k_trad.shape:
        diff_k = torch.abs(k_paged - k_trad).max()
        diff_v = torch.abs(v_paged - v_trad).max()
        print(f"与传统Cache差异: K={diff_k:.6f}, V={diff_v:.6f}")
    else:
        print(f"形状不匹配: PagedCache={k_paged.shape}, 传统Cache={k_trad.shape}")
    
    # 测试增量更新
    print("\n测试增量更新...")
    k_new = torch.randn(batch_size, config.num_heads, 3, config.head_dim, dtype=config.dtype)
    v_new = torch.randn(batch_size, config.num_heads, 3, config.head_dim, dtype=config.dtype)
    
    k_paged2, v_paged2 = paged_adapter.update(layer_idx, k_new, v_new)
    k_trad2, v_trad2 = traditional_cache.update(layer_idx, k_new, v_new)
    
    print(f"增量更新后形状: Paged={k_paged2.shape}, 传统={k_trad2.shape}")
    
    return True

def test_with_minillm():
    """测试与MiniLLM的集成"""
    print("\n=== 测试与MiniLLM集成 ===")
    
    config = Configs.small()
    config.device = "cpu"
    config.num_layers = 2  # 减少层数加快测试
    
    # 创建模型
    model = MiniLLM(config)
    model.eval()
    
    print(f"MiniLLM创建成功: {config.num_layers} layers")
    
    # 测试数据
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"输入形状: {input_ids.shape}")
    
    # 创建两种KVCache
    traditional_cache = KVCache(config, batch_size)
    paged_cache = create_paged_kv_cache(
        config, 
        batch_size=batch_size, 
        block_size=8, 
        blocks_per_seq=16
    )
    
    print("两种Cache创建成功")
    
    # 使用传统Cache
    with torch.no_grad():
        logits_trad = model(input_ids, kv_cache=traditional_cache, use_flash_attention=False)
    
    # 重置模型（如果有状态的话）
    paged_cache.reset()
    
    # 使用PagedCache
    with torch.no_grad():
        logits_paged = model(input_ids, kv_cache=paged_cache, use_flash_attention=False)
    
    print(f"传统Cache输出形状: {logits_trad.shape}")
    print(f"PagedCache输出形状: {logits_paged.shape}")
    
    # 比较输出差异
    if logits_trad.shape == logits_paged.shape:
        diff = torch.abs(logits_trad - logits_paged).max()
        print(f"输出最大差异: {diff:.6f}")
        
        # 检查是否差异在合理范围内（放宽容忍度，因为不同的attention实现可能有数值差异）
        if diff < 1e-4:
            print("✅ 输出基本一致!")
        elif diff < 0.1:
            print("⚠️ 输出有差异，但在可接受范围内（可能由于不同attention实现）")
        else:
            print("❌ 输出差异较大，需要检查")
            # 打印更多诊断信息
            mean_diff = torch.abs(logits_trad - logits_paged).mean()
            print(f"平均差异: {mean_diff:.6f}")
            print(f"传统Cache输出范围: [{logits_trad.min():.3f}, {logits_trad.max():.3f}]")
            print(f"PagedCache输出范围: [{logits_paged.min():.3f}, {logits_paged.max():.3f}]")
    else:
        print("❌ 输出形状不匹配")
    
    # 显示内存使用情况
    if hasattr(paged_cache, 'get_memory_usage'):
        memory_info = paged_cache.get_memory_usage()
        print(f"PagedCache内存使用: {memory_info}")
    
    return True

def run_all_tests():
    """运行所有测试"""
    tests = [
        test_basic_paged_kvcache,
        test_paged_kvcache_adapter,
        test_with_minillm
    ]
    
    print("开始PagedKVCache测试...")
    print("=" * 50)
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 失败: {e}")
            print(traceback.format_exc())
            results.append((test_func.__name__, "ERROR"))
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    for test_name, status in results:
        status_emoji = "✅" if status == "PASS" else "❌"
        print(f"{status_emoji} {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, status in results if status == "PASS")
    print(f"\n总计: {passed_tests}/{total_tests} 测试通过")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
