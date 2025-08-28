import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_attention_2_kernel(
    Q, K, V, O,  # Input and output tensors
    L, M,  # LSE (log-sum-exp) and max values
    stride_qz, stride_qh, stride_qm, stride_qk,  # Q strides
    stride_kz, stride_kh, stride_kn, stride_kk,  # K strides  
    stride_vz, stride_vh, stride_vn, stride_vk,  # V strides
    stride_oz, stride_oh, stride_om, stride_on,  # O strides
    stride_lz, stride_lh, stride_lm,  # L strides
    stride_mz, stride_mh, stride_mm,  # M strides
    Z, H, N_CTX, HEAD_DIM,  # Tensor dimensions
    scale,  # Attention scale factor
    # BLOCK_M 是对Query 进行行分块
    # BLOCK_N 是对kv 进行列分块
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Flash Attention 2 kernel implementation in Triton.
    
    This kernel implements the memory-efficient attention algorithm with improved
    work partitioning compared to Flash Attention 1.
    
    Args:
        Q, K, V: Query, Key, Value tensors [batch, heads, seq_len, head_dim]
        O: Output tensor [batch, heads, seq_len, head_dim]
        L: Log-sum-exp values [batch, heads, seq_len]
        M: Max values [batch, heads, seq_len]
        Various stride parameters for tensor indexing
        Z, H, N_CTX, HEAD_DIM: Tensor dimensions
        scale: Attention scale factor (1/sqrt(head_dim))
        BLOCK_M, BLOCK_N, BLOCK_DMODEL: Block sizes for tiling
        IS_CAUSAL: Whether to apply causal masking
    """
    # Get program IDs
    start_m = tl.program_id(0) # 行索引
    off_hz = tl.program_id(1) # head & batch 的组合ID
    off_z = off_hz // H # batch index
    off_h = off_hz % H # head index
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # query 的 offset
    offs_n = tl.arange(0, BLOCK_N) # kv 的 offset
    offs_d = tl.arange(0, BLOCK_DMODEL) # head_dim 的 offset
    
    # Initialize pointers to Q, O, L, M for this block
    q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh + 
                  offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    o_ptrs = O + (off_z * stride_oz + off_h * stride_oh + 
                  offs_m[:, None] * stride_om + offs_d[None, :] * stride_on)
    l_ptrs = L + (off_z * stride_lz + off_h * stride_lh + offs_m * stride_lm)
    m_ptrs = M + (off_z * stride_mz + off_h * stride_mh + offs_m * stride_mm)
    
    # Initialize accumulator and statistics
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q block and convert to float32 for computation
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    q = q.to(tl.float32)
    
    # Determine loop bounds for causal masking
    if IS_CAUSAL:
        # For causal attention, we only process blocks up to the current row block
        loop_end = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)
    else:
        loop_end = N_CTX
    
    # Loop over K, V blocks  
    for start_n in range(0, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Calculate the actual block range
        offs_n_curr = start_n + offs_n
        
        # Load K, V blocks
        k_ptrs = K + (off_z * stride_kz + off_h * stride_kh + 
                      offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v_ptrs = V + (off_z * stride_vz + off_h * stride_vh + 
                      offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk)

        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)

        # Convert to float32 for computation
        k = k.to(tl.float32)
        v = v.to(tl.float32)
        
        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))
        qk = qk * scale  # Scale by 1/sqrt(d_k)
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Apply padding mask
        padding_mask = offs_n_curr[None, :] < N_CTX
        qk = tl.where(padding_mask, qk, float("-inf"))
        
        # Compute new max values
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        
        # Compute softmax scaling factors
        alpha = tl.exp(m_i - m_ij)
        beta = tl.exp(qk - m_ij[:, None])
        
        # Update accumulator with previous values
        acc = acc * alpha[:, None]
        
        # Add new contribution
        acc += tl.dot(beta, v)
        
        # Update statistics
        l_i = l_i * alpha + tl.sum(beta, 1)
        m_i = m_ij
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store results
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX)
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)


def flash_attention_2(q, k, v, causal=False):
    """
    Flash Attention 2 implementation using Triton.
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim] 
        v: Value tensor [batch_size, num_heads, seq_len, head_dim]
        causal: Whether to apply causal (triangular) masking
        
    Returns:
        output: Attention output [batch_size, num_heads, seq_len, head_dim]
        lse: Log-sum-exp values [batch_size, num_heads, seq_len] 
        max_vals: Max values [batch_size, num_heads, seq_len]
    """
    # Get tensor dimensions
    BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape
    
    # Ensure tensors are contiguous and on GPU
    q = q.contiguous()
    k = k.contiguous() 
    v = v.contiguous()
    
    # Create output tensors
    output = torch.empty_like(q)
    lse = torch.empty((BATCH, N_HEAD, N_CTX), device=q.device, dtype=torch.float32)
    max_vals = torch.empty((BATCH, N_HEAD, N_CTX), device=q.device, dtype=torch.float32)
    
    # Define block sizes (tuned for performance)
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = HEAD_DIM
    
    # Calculate attention scale
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # Calculate grid dimensions
    grid = lambda meta: (
        triton.cdiv(N_CTX, BLOCK_M),  # Number of blocks in M dimension
        BATCH * N_HEAD,               # Batch * num_heads
    )
    
    # Launch kernel
    flash_attention_2_kernel[grid](
        q, k, v, output,
        lse, max_vals,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides  
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3), 
        # O strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # L strides
        lse.stride(0), lse.stride(1), lse.stride(2),
        # M strides 
        max_vals.stride(0), max_vals.stride(1), max_vals.stride(2),
        # Dimensions
        BATCH, N_HEAD, N_CTX, HEAD_DIM,
        # Scale
        scale,
        # Block sizes
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        # Config
        IS_CAUSAL=causal,
    )
    
    return output, lse, max_vals


def benchmark_flash_attention_2():
    """
    Simple benchmark to test Flash Attention 2 implementation.
    """
    import time
    
    # Test parameters
    BATCH = 2
    N_HEAD = 8  
    N_CTX = 2048
    HEAD_DIM = 64
    
    # Create random input tensors
    torch.manual_seed(42)
    q = torch.randn(BATCH, N_HEAD, N_CTX, HEAD_DIM, device='cuda', dtype=torch.float16)
    k = torch.randn(BATCH, N_HEAD, N_CTX, HEAD_DIM, device='cuda', dtype=torch.float16)
    v = torch.randn(BATCH, N_HEAD, N_CTX, HEAD_DIM, device='cuda', dtype=torch.float16)
    
    # Warm up
    for _ in range(10):
        output, lse, max_vals = flash_attention_2(q, k, v, causal=True)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    num_runs = 100
    for _ in range(num_runs):
        output, lse, max_vals = flash_attention_2(q, k, v, causal=True)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
    print(f"Flash Attention 2 average time: {avg_time:.3f} ms")
    print(f"Output shape: {output.shape}")
    print(f"LSE shape: {lse.shape}")
    print(f"Max vals shape: {max_vals.shape}")
    
    return output, lse, max_vals


if __name__ == "__main__":
    # Run benchmark if script is executed directly
    if torch.cuda.is_available():
        benchmark_flash_attention_2()
    else:
        print("CUDA not available. Please run on a GPU.")
        
    # Simple correctness test
    print("\n=== Correctness Test ===")
    
    # Use larger dimensions that meet Triton's requirements (>=16)
    batch, heads, seq_len, head_dim = 1, 2, 32, 16
    q = torch.randn(batch, heads, seq_len, head_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    k = torch.randn(batch, heads, seq_len, head_dim, device='cuda' if torch.cuda.is_available() else 'cpu')  
    v = torch.randn(batch, heads, seq_len, head_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        # Test Flash Attention 2
        output_fa2, lse, max_vals = flash_attention_2(q, k, v, causal=True)
        print(f"Flash Attention 2 output range: [{output_fa2.min():.4f}, {output_fa2.max():.4f}]")
        
        # Compare with naive attention (for small sequences)
        def naive_attention(q, k, v, causal=True):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            if causal:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
                scores.masked_fill_(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)
        
        output_naive = naive_attention(q, k, v, causal=True)
        print(f"Naive attention output range: [{output_naive.min():.4f}, {output_naive.max():.4f}]")
        
        # Check if results are close
        max_diff = torch.max(torch.abs(output_fa2 - output_naive)).item()
        print(f"Max difference: {max_diff:.6f}")
        print(f"Results match: {max_diff < 1e-2}")  # Slightly relaxed tolerance for fp16
    else:
        print("CUDA not available for correctness test.")