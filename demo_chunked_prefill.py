"""
Chunked Prefill Demo

This demo showcases the chunked prefill implementation with KV cache
and Flash Attention 2 for improved throughput in MiniLLM.
"""

import torch
import time
import sys
import os
from typing import List

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.config import ModelConfig, Configs
from src.models.mini_llm import MiniLLM
from src.models.chunked_prefill import (
    ChunkedPrefillMiniLLM, 
    chunked_generate_with_kv_cache,
    BatchedChunkedPrefill,
    benchmark_chunked_prefill
)
from src.cache.kv_cache import KVCache


def demo_basic_chunked_prefill():
    """Demonstrate basic chunked prefill functionality"""
    print("=" * 60)
    print("Basic Chunked Prefill Demo")
    print("=" * 60)
    
    # Create model configuration
    config = Configs.medium()
    config.chunk_size = 256
    config.use_flash_attention = True
    config.max_seq_len = 2048
    
    print(f"Model Config:")
    print(f"  - Vocabulary Size: {config.vocab_size}")
    print(f"  - Model Dimension: {config.dim_model}")
    print(f"  - Number of Heads: {config.num_heads}")
    print(f"  - Number of Layers: {config.num_layers}")
    print(f"  - Max Sequence Length: {config.max_seq_len}")
    print(f"  - Chunk Size: {config.chunk_size}")
    print(f"  - Device: {config.device}")
    print()
    
    # Create models
    regular_model = MiniLLM(config)
    chunked_model = ChunkedPrefillMiniLLM(config, chunk_size=config.chunk_size)
    
    # Copy weights to chunked model
    chunked_model.load_state_dict(regular_model.state_dict())
    
    # Create test input
    batch_size = 2
    seq_len = 1024
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=config.device)
    
    print(f"Test Input Shape: {input_ids.shape}")
    print()
    
    # Test regular forward pass
    print("Testing Regular Forward Pass...")
    start_time = time.time()
    
    with torch.no_grad():
        regular_output = regular_model.forward(
            input_ids=input_ids,
            use_flash_attention=config.use_flash_attention
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    regular_time = time.time() - start_time
    
    print(f"  Output Shape: {regular_output.shape}")
    print(f"  Time: {regular_time:.4f} seconds")
    print(f"  Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB" if torch.cuda.is_available() else "")
    print()
    
    # Test chunked prefill
    print("Testing Chunked Prefill...")
    start_time = time.time()
    
    with torch.no_grad():
        chunked_output = chunked_model.chunked_prefill_forward(
            input_ids=input_ids,
            use_flash_attention=config.use_flash_attention
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    chunked_time = time.time() - start_time
    
    print(f"  Output Shape: {chunked_output.shape}")
    print(f"  Time: {chunked_time:.4f} seconds")
    print(f"  Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB" if torch.cuda.is_available() else "")
    print(f"  Speedup: {regular_time / chunked_time:.2f}x")
    print()
    
    # Verify outputs are close
    max_diff = torch.max(torch.abs(regular_output - chunked_output)).item()
    print(f"Maximum Output Difference: {max_diff:.6f}")
    print(f"Outputs Match: {max_diff < 1e-3}")
    print()


def demo_chunked_generation():
    """Demonstrate text generation with chunked prefill"""
    print("=" * 60)
    print("Chunked Generation Demo")
    print("=" * 60)
    
    # Create smaller model for generation demo
    config = Configs.small()
    config.chunk_size = 128
    config.use_flash_attention = True
    config.max_seq_len = 512
    
    model = MiniLLM(config)
    
    # Create test prompt
    batch_size = 1
    prompt_len = 256
    max_new_tokens = 50
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=config.device)
    
    print(f"Generation Config:")
    print(f"  - Prompt Length: {prompt_len}")
    print(f"  - Max New Tokens: {max_new_tokens}")
    print(f"  - Chunk Size: {config.chunk_size}")
    print(f"  - Temperature: 1.0")
    print()
    
    # Generate text with chunked prefill
    print("Generating text with chunked prefill...")
    start_time = time.time()
    
    generated_ids = chunked_generate_with_kv_cache(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        chunk_size=config.chunk_size,
        temperature=1.0,
        use_flash_attention=config.use_flash_attention
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_time = time.time() - start_time
    
    print(f"Generated Sequence Shape: {generated_ids.shape}")
    print(f"Generation Time: {generation_time:.4f} seconds")
    print(f"Tokens per Second: {max_new_tokens / generation_time:.2f}")
    print()


def demo_batched_processing():
    """Demonstrate batched chunked prefill processing"""
    print("=" * 60)
    print("Batched Chunked Prefill Demo")
    print("=" * 60)
    
    # Create model
    config = Configs.small()
    config.chunk_size = 128
    model = MiniLLM(config)
    
    # Create batched processor
    processor = BatchedChunkedPrefill(
        model=model,
        chunk_size=config.chunk_size,
        max_batch_size=4
    )
    
    # Create variable-length sequences
    sequence_lengths = [64, 128, 256, 384, 512]
    input_batch = []
    
    for seq_len in sequence_lengths:
        seq = torch.randint(0, config.vocab_size, (seq_len,), device=config.device)
        input_batch.append(seq)
    
    print(f"Processing {len(input_batch)} sequences with lengths: {sequence_lengths}")
    print()
    
    # Process batch
    start_time = time.time()
    results = processor.process_batch(input_batch, use_flash_attention=True)
    processing_time = time.time() - start_time
    
    print("Results:")
    for i, (seq_len, result) in enumerate(zip(sequence_lengths, results)):
        print(f"  Sequence {i+1}: Input length {seq_len} -> Output shape {result.shape}")
    
    print(f"\nTotal Processing Time: {processing_time:.4f} seconds")
    print(f"Average Time per Sequence: {processing_time / len(input_batch):.4f} seconds")
    print()


def demo_performance_comparison():
    """Compare performance of different approaches"""
    print("=" * 60)
    print("Performance Comparison Demo")
    print("=" * 60)
    
    # Create model
    config = Configs.medium()
    config.use_flash_attention = True
    model = MiniLLM(config)
    
    # Run benchmark
    results = benchmark_chunked_prefill(
        model=model,
        sequence_lengths=[512, 1024, 2048],
        chunk_sizes=[128, 256, 512],
        batch_size=2,
        num_trials=5
    )
    
    # Print summary
    print("Performance Summary:")
    print("-" * 40)
    for seq_len, seq_results in results.items():
        regular_time = seq_results.get('regular', 0)
        print(f"\nSequence Length {seq_len}:")
        print(f"  Regular: {regular_time:.4f}s")
        
        best_chunked_time = float('inf')
        best_chunk_size = None
        
        for method, time_val in seq_results.items():
            if method.startswith('chunked_'):
                chunk_size = method.split('_')[1]
                speedup = regular_time / time_val if time_val > 0 else 0
                print(f"  Chunk {chunk_size}: {time_val:.4f}s (speedup: {speedup:.2f}x)")
                
                if time_val < best_chunked_time:
                    best_chunked_time = time_val
                    best_chunk_size = chunk_size
        
        if best_chunk_size:
            best_speedup = regular_time / best_chunked_time
            print(f"  Best: Chunk {best_chunk_size} with {best_speedup:.2f}x speedup")


def main():
    """Run all demos"""
    print("MiniLLM Chunked Prefill Demo")
    print("This demo showcases the chunked prefill optimization with KV cache and Flash Attention 2")
    print()
    
    try:
        # Run demos
        demo_basic_chunked_prefill()
        demo_chunked_generation()
        demo_batched_processing()
        demo_performance_comparison()
        
        print("=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
