import torch
import torch.nn.functional as F
from typing import Tuple
import math
import time


def standard_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    用于比较的标准注意力实现
    """
    scale = q.size(-1) ** -0.5
    scores = (q @ k.T) * scale
    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ v
    return output

def tiled_attention(
    q: torch.Tensor, # [seq_len, dim] 查询矩阵
    k: torch.Tensor, # [seq_len, dim] 键矩阵
    v: torch.Tensor, # [seq_len, dim] 值矩阵
    block_size: int = 128
) -> torch.Tensor:
    """
    Flash Attention 2 实现，包含分块（tiling）和在线 softmax（online softmax）
    
    该实现采用了 Flash Attention 的关键思想：
    1. 分块（Tiling）：将计算划分为多个块，以减少内存使用
    2. 在线 Softmax（Online Softmax）：增量式计算 softmax 统计量，避免存储大型中间矩阵
    3. 重计算（Recomputation）：通过重新计算注意力权重来换取内存，而非存储它们
    
    相比朴素注意力的主要改进：
    - 内存复杂度从 O(N²) 降低至 O(N)
    - 通过在线追踪最大值保证数值稳定性
    - 通过顺序处理块实现内存高效
    
    Args:
        q, k, v: [seq_len, dim] - 查询（Query）、键（Key）、值（Value）矩阵
        block_size: int - 每个块的大小（在内存和计算之间进行权衡）
        
    Return:
        output: [seq_len, dim] - 注意力输出
        
    算法细节：
    - 对于每个查询块 Q_i，遍历所有键值块 K_j, V_j
    - 维护运行时统计量（最大值 m_i 和总和 l_i）以保证数值稳定性
    - 使用在线 softmax 更新，避免存储完整的注意力矩阵
    - 最终的归一化确保 softmax 计算正确
    
    张量形状说明：
    - q_i: [min(block_size, seq_len-i), dim] - 当前查询块
    - k_j: [min(block_size, seq_len-j), dim] - 当前键块
    - v_j: [min(block_size, seq_len-j), dim] - 当前值块
    - s_ij: [min(block_size, seq_len-i), min(block_size, seq_len-j)] - 注意力分数矩阵
    - o_i: [min(block_size, seq_len-i), dim] - 当前查询块的输出
    - l_i: [min(block_size, seq_len-i)] - 当前查询块的行和
    - m_i: [min(block_size, seq_len-i)] - 当前查询块的行最大值
    - p_ij: [min(block_size, seq_len-i), min(block_size, seq_len-j)] - 注意力权重矩阵
    """

    seq_len, dim = q.shape
    scale = dim ** -0.5
    o = torch.zeros_like(q, dtype=torch.float32)  # 使用float32提高数值稳定性
    l = torch.zeros(seq_len, device=q.device, dtype=torch.float32)  # 行和
    m = torch.full((seq_len,), -torch.inf, device=q.device, dtype=torch.float32)  # 行最大值

    # 将输入转换为float32进行计算
    q = q.float()
    k = k.float()
    v = v.float()

    # 查询块的外循环
    for i in range(0, seq_len, block_size):
        q_i = q[i:i + block_size] * scale  # [block_size, dim]
        o_i = torch.zeros(min(block_size, seq_len - i), dim, device=q.device, dtype=torch.float32)
        l_i = torch.zeros(min(block_size, seq_len - i), device=q.device, dtype=torch.float32)
        m_i = torch.full((min(block_size, seq_len - i),), -torch.inf, device=q.device, dtype=torch.float32)
        
        # 键值块的内循环
        for j in range(0, seq_len, block_size):
            k_j = k[j:j + block_size]  # [block_size, dim]
            v_j = v[j:j + block_size]  # [block_size, dim]
            
            # 计算注意力分数
            s_ij = q_i @ k_j.T  # [q_block_size, k_block_size]
            
            # 在线softmax更新 - Flash Attention的关键思想
            m_ij = s_ij.max(dim=-1)[0]  # [q_block_size] - 当前块的最大值
            
            # 更新行最大值
            m_i_new = torch.maximum(m_i, m_ij)
            
            # 计算具有数值稳定性的指数值
            alpha = torch.exp(m_i - m_i_new)  # [q_block_size] - 之前块的重新缩放因子
            beta = torch.exp(m_ij - m_i_new)   # [q_block_size] - 未使用但保留以便理解

            
            # 更新输出和归一化
            o_i = o_i * alpha.unsqueeze(-1)  # 重新缩放之前的输出
            
            # 计算当前块的注意力权重
            p_ij = torch.exp(s_ij - m_i_new.unsqueeze(-1))  # [q_block_size, k_block_size]
            
            # 用当前块的贡献更新输出
            o_i = o_i + (p_ij @ v_j)  # [q_block_size, dim]
            
            # 更新归一化因子
            l_i = l_i * alpha + p_ij.sum(dim=-1)
            m_i = m_i_new
        
        # 存储带有最终归一化的结果
        actual_block_size = min(block_size, seq_len - i)
        o[i:i + actual_block_size] = o_i / l_i.unsqueeze(-1)
        l[i:i + actual_block_size] = l_i
        m[i:i + actual_block_size] = m_i

    return o.to(q.dtype)


def test_flash_attention():
    """
    验证Flash Attention实现的测试函数
    """
    torch.manual_seed(42)
    
    print("测试Flash Attention 2实现")
    print("=" * 50)
    
    # 使用不同配置进行测试
    test_configs = [
        {"seq_len": 128, "dim": 32, "block_size": 32},
        {"seq_len": 256, "dim": 64, "block_size": 64},
        {"seq_len": 512, "dim": 128, "block_size": 128},
        {"seq_len": 1024, "dim": 64, "block_size": 256},
    ]
    
    all_passed = True
    
    for i, config in enumerate(test_configs):
        print(f"\n测试 {i+1}: seq_len={config['seq_len']}, dim={config['dim']}, block_size={config['block_size']}")
        
        # 生成随机输入
        q = torch.randn(config["seq_len"], config["dim"])
        k = torch.randn(config["seq_len"], config["dim"])
        v = torch.randn(config["seq_len"], config["dim"])
        
        # 计算输出
        flash_output = tiled_attention(q, k, v, block_size=config["block_size"])
        standard_output = standard_attention(q, k, v)
        
        # 检查数值精度
        max_diff = torch.max(torch.abs(flash_output - standard_output)).item()
        mean_diff = torch.mean(torch.abs(flash_output - standard_output)).item()
        
        print(f"  最大差异: {max_diff:.8f}")
        print(f"  平均差异: {mean_diff:.8f}")
        
        # 检查输出是否接近
        is_close = torch.allclose(flash_output, standard_output, atol=1e-4, rtol=1e-4)
        
        if is_close:
            print(f"  ✅ 测试 {i+1} 通过!")
        else:
            print(f"  ❌ 测试 {i+1} 失败!")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有Flash Attention测试都通过了!")
    else:
        print("💥 部分测试失败!")
        
    return all_passed


def benchmark_memory_usage():
    """
    比较标准注意力和flash attention的内存使用情况基准测试
    """
    import time
    import gc
    
    print("\n内存使用基准测试")
    print("=" * 40)
    
    # 测试配置 - 递增的序列长度
    configs = [512, 1024, 2048]  # seq_lens
    dim = 128
    block_size = 128
    
    for seq_len in configs:
        print(f"\n序列长度: {seq_len}, 维度: {dim}")
        print("-" * 30)
        
        # 生成数据
        q = torch.randn(seq_len, dim)
        k = torch.randn(seq_len, dim)  
        v = torch.randn(seq_len, dim)
        
        # 清空缓存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # 计时Flash Attention
        torch.manual_seed(42)
        start_time = time.time()
        flash_output = tiled_attention(q, k, v, block_size=block_size)
        flash_time = time.time() - start_time
        
        # 计时标准注意力
        torch.manual_seed(42)
        start_time = time.time()
        standard_output = standard_attention(q, k, v)
        standard_time = time.time() - start_time
        
        # 验证正确性
        max_diff = torch.max(torch.abs(flash_output - standard_output)).item()
        
        print(f"Flash Attention时间:    {flash_time:.4f}s")
        print(f"标准注意力时间: {standard_time:.4f}s")
        print(f"速度比: {standard_time/flash_time:.2f}x")
        print(f"最大差异: {max_diff:.2e}")
        
        # 内存使用估算
        standard_memory = seq_len * seq_len * 4  # 注意力矩阵的字节数
        flash_memory = block_size * block_size * 4  # 注意力块的字节数
        memory_savings = (standard_memory - flash_memory) / standard_memory * 100
        
        print(f"预计内存节省: {memory_savings:.1f}%")
        print(f"标准内存（注意力矩阵）: {standard_memory/1024/1024:.1f} MB")
        print(f"Flash内存（注意力块）: {flash_memory/1024:.1f} KB")


def demo_usage():
    """
    演示Flash Attention实现的典型用法
    """
    print("\nFlash Attention 2使用演示")
    print("=" * 40)
    
    # 例子：模拟一个小型transformer注意力层
    batch_size = 4
    seq_len = 512
    dim = 128
    num_heads = 8
    head_dim = dim // num_heads
    
    print(f"模拟多头注意力:")
    print(f"- 批次大小: {batch_size}")
    print(f"- 序列长度: {seq_len}")
    print(f"- 隐藏维度: {dim}")  
    print(f"- 头数: {num_heads}")
    print(f"- 头维度: {head_dim}")
    
    # 生成随机输入（通常这些来自嵌入）
    x = torch.randn(batch_size, seq_len, dim)
    
    # 模拟多头注意力计算
    total_time = 0
    
    for b in range(batch_size):
        for h in range(num_heads):
            # 提取特定头的Q, K, V（简化版 - 通常来自线性投影）
            start_dim = h * head_dim
            end_dim = start_dim + head_dim
            
            q = x[b, :, start_dim:end_dim]  # [seq_len, head_dim]
            k = x[b, :, start_dim:end_dim]  # [seq_len, head_dim]  
            v = x[b, :, start_dim:end_dim]  # [seq_len, head_dim]
            
            # 应用Flash Attention
            start_time = time.time()
            output = tiled_attention(q, k, v, block_size=64)
            total_time += time.time() - start_time
    
    print(f"\n总计算时间: {total_time:.4f}s")
    print(f"每个头的平均时间: {total_time/(batch_size*num_heads):.4f}s")
    print("✅ 多头注意力模拟完成!")


if __name__ == "__main__":
    # 运行所有测试和基准测试
    test_flash_attention()
    # benchmark_memory_usage()
    # demo_usage()
