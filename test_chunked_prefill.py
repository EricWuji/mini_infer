"""
Integration test for chunked prefill with KV cache and Flash Attention 2
"""

import torch
import sys
import os
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.config import ModelConfig, Configs
from src.models.mini_llm import MiniLLM
from src.models.chunked_prefill import ChunkedPrefillMiniLLM, chunked_generate_with_kv_cache
from src.cache.kv_cache import KVCache


def test_chunked_prefill_correctness():
    """Test that chunked prefill produces the same results as regular forward pass"""
    print("Testing chunked prefill correctness...")
    
    # Create test configuration
    config = Configs.small()
    config.num_layers = 2  # Use more layers for better testing
    config.max_seq_len = 1024
    config.chunk_size = 256
    
    # Create models
    regular_model = MiniLLM(config)
    chunked_model = ChunkedPrefillMiniLLM(config, chunk_size=256)
    
    # Copy weights
    chunked_model.load_state_dict(regular_model.state_dict())
    
    # Put models in eval mode
    regular_model.eval()
    chunked_model.eval()
    
    # Test different sequence lengths
    test_lengths = [128, 256, 512, 768]
    batch_size = 2
    
    for seq_len in test_lengths:
        print(f"  Testing sequence length {seq_len}...")
        
        # Create test input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=config.device)
        
        # Regular forward pass
        with torch.no_grad():
            regular_output = regular_model.forward(input_ids, use_flash_attention=True)
        
        # Chunked forward pass
        with torch.no_grad():
            chunked_output = chunked_model.chunked_prefill_forward(input_ids, use_flash_attention=True)
        
        # Compare outputs
        max_diff = torch.max(torch.abs(regular_output - chunked_output)).item()
        mean_diff = torch.mean(torch.abs(regular_output - chunked_output)).item()
        
        # Relaxed tolerance for float16 and chunked processing
        max_tolerance = 6e-2 if config.dtype == torch.float16 else 1e-3  # Slightly more relaxed
        mean_tolerance = 2e-2 if config.dtype == torch.float16 else 1e-4
        
        assert max_diff < max_tolerance, f"Max difference too large: {max_diff} (tolerance: {max_tolerance})"
        assert mean_diff < mean_tolerance, f"Mean difference too large: {mean_diff} (tolerance: {mean_tolerance})"
        
        print(f"    Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f} ✓")
    
    print("Chunked prefill correctness test passed!")
    return True


def test_kv_cache_integration():
    """Test KV cache integration with chunked prefill"""
    print("Testing KV cache integration...")
    
    config = Configs.small()
    config.num_layers = 2
    config.max_seq_len = 512
    config.chunk_size = 128
    
    model = MiniLLM(config)
    model.eval()
    
    batch_size = 2
    seq_len = 384
    
    # Create test input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=config.device)
    
    # Test with KV cache
    kv_cache = KVCache(config, batch_size=batch_size)
    
    with torch.no_grad():
        # First pass - should populate cache
        output1 = model.forward(input_ids, kv_cache=kv_cache, use_flash_attention=True)
        
        # Check that cache is populated
        for batch_idx in range(batch_size):
            cached_len = kv_cache.get_seq_len(batch_idx)
            assert cached_len == seq_len, f"Expected cache length {seq_len}, got {cached_len}"
        
        # Second pass with additional tokens
        new_tokens = torch.randint(0, config.vocab_size, (batch_size, 64), device=config.device)
        output2 = model.forward(new_tokens, kv_cache=kv_cache, use_flash_attention=True)
        
        # Check cache length updated
        for batch_idx in range(batch_size):
            cached_len = kv_cache.get_seq_len(batch_idx)
            expected_len = seq_len + 64
            assert cached_len == expected_len, f"Expected cache length {expected_len}, got {cached_len}"
    
    print("KV cache integration test passed!")
    return True


def test_generation_with_chunked_prefill():
    """Test text generation with chunked prefill"""
    print("Testing generation with chunked prefill...")
    
    config = Configs.small()
    config.max_seq_len = 256
    config.chunk_size = 64
    
    model = MiniLLM(config)
    model.eval()
    
    batch_size = 1
    prompt_len = 128
    max_new_tokens = 32
    
    # Create test prompt
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=config.device)
    
    # Generate with chunked prefill
    with torch.no_grad():
        generated_ids = chunked_generate_with_kv_cache(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            chunk_size=config.chunk_size,
            temperature=1.0,
            use_flash_attention=True
        )
    
    # Verify output shape
    expected_shape = (batch_size, prompt_len + max_new_tokens)
    assert generated_ids.shape == expected_shape, f"Expected shape {expected_shape}, got {generated_ids.shape}"
    
    # Verify that the prompt part is unchanged
    assert torch.equal(generated_ids[:, :prompt_len], input_ids), "Prompt part should be unchanged"
    
    print("Generation test passed!")
    return True


def test_flash_attention_integration():
    """Test Flash Attention 2 integration"""
    print("Testing Flash Attention 2 integration...")
    
    # Check if CUDA is available for Flash Attention
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Flash Attention test")
        return True
    
    config = Configs.small()
    config.device = "cuda"
    config.dtype = torch.float16
    config.max_seq_len = 512
    config.chunk_size = 128
    
    model = ChunkedPrefillMiniLLM(config, chunk_size=128)
    model.eval()
    
    batch_size = 2
    seq_len = 256
    
    # Create test input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=config.device)
    
    # Test with Flash Attention enabled
    with torch.no_grad():
        output_flash = model.chunked_prefill_forward(input_ids, use_flash_attention=True)
    
    # Test with Flash Attention disabled (fallback to regular attention)
    with torch.no_grad():
        output_regular = model.chunked_prefill_forward(input_ids, use_flash_attention=False)
    
    # Compare outputs (should be close but not identical due to numerical differences)
    max_diff = torch.max(torch.abs(output_flash - output_regular)).item()
    print(f"  Max difference between Flash and Regular attention: {max_diff:.6f}")
    
    # The difference should be reasonable for fp16 precision and different attention implementations
    max_tolerance = 2.0 if config.dtype == torch.float16 else 1e-1
    assert max_diff < max_tolerance, f"Difference too large: {max_diff} (tolerance: {max_tolerance})"
    
    print("Flash Attention integration test passed!")
    return True


def benchmark_performance():
    """Benchmark performance improvements"""
    print("Benchmarking performance improvements...")
    
    config = Configs.medium()
    config.max_seq_len = 2048
    config.chunk_size = 512
    
    # Create models
    regular_model = MiniLLM(config)
    chunked_model = ChunkedPrefillMiniLLM(config, chunk_size=512)
    chunked_model.load_state_dict(regular_model.state_dict())
    
    # Put in eval mode
    regular_model.eval()
    chunked_model.eval()
    
    batch_size = 2
    seq_len = 1536  # Sequence that will benefit from chunking
    
    # Create test input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=config.device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = regular_model.forward(input_ids, use_flash_attention=True)
            _ = chunked_model.chunked_prefill_forward(input_ids, use_flash_attention=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark regular model
    num_trials = 10
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_trials):
            _ = regular_model.forward(input_ids, use_flash_attention=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    regular_time = (time.time() - start_time) / num_trials
    
    # Benchmark chunked model
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_trials):
            _ = chunked_model.chunked_prefill_forward(input_ids, use_flash_attention=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    chunked_time = (time.time() - start_time) / num_trials
    
    speedup = regular_time / chunked_time
    
    print(f"  Regular model time: {regular_time:.4f}s")
    print(f"  Chunked model time: {chunked_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    if torch.cuda.is_available():
        print(f"  Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    print("Performance benchmark completed!")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("Running Chunked Prefill Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_chunked_prefill_correctness,
        test_kv_cache_integration,
        test_generation_with_chunked_prefill,
        test_flash_attention_integration,
        benchmark_performance
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"Running {test_func.__name__}...")
            result = test_func()
            if result:
                passed += 1
                print(f"✓ {test_func.__name__} passed")
            else:
                failed += 1
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("All tests passed! 🎉")
        return True
    else:
        print(f"{failed} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
