"""
Optimized Inference Engine for MiniLLM

This module provides a high-level inference engine that combines:
- Chunked prefill for improved throughput
- KV cache for efficient generation
- Flash Attention 2 for memory optimization
- Batch processing for better GPU utilization
"""

import torch
from torch import nn
from typing import List, Optional, Union, Dict, Any, Tuple
import time
from dataclasses import dataclass

from src.config import ModelConfig
from src.models.mini_llm import MiniLLM
from src.models.chunked_prefill import ChunkedPrefillMiniLLM, chunked_generate_with_kv_cache
from src.cache.kv_cache import KVCache


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = True
    pad_token_id: int = 0
    eos_token_id: Optional[int] = None
    chunk_size: Optional[int] = None  # If None, use model default


class OptimizedInferenceEngine:
    """
    High-performance inference engine for MiniLLM with chunked prefill optimization
    """
    
    def __init__(self, 
                 model: Union[MiniLLM, ChunkedPrefillMiniLLM],
                 config: Optional[ModelConfig] = None):
        """
        Initialize the inference engine
        
        Args:
            model: MiniLLM or ChunkedPrefillMiniLLM instance
            config: Optional model configuration (inferred from model if not provided)
        """
        self.model = model
        self.config = config or model.config
        
        # Convert to chunked model if needed
        if not isinstance(model, ChunkedPrefillMiniLLM):
            self.chunked_model = ChunkedPrefillMiniLLM(
                self.config, 
                chunk_size=self.config.chunk_size
            )
            self.chunked_model.load_state_dict(model.state_dict())
        else:
            self.chunked_model = model
        
        self.chunked_model.eval()
        
        # Performance tracking
        self.stats = {
            'total_tokens_processed': 0,
            'total_time': 0.0,
            'num_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def generate(self,
                input_ids: torch.Tensor,
                generation_config: Optional[GenerationConfig] = None,
                use_kv_cache: bool = True,
                return_timing_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Generate text using optimized inference
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            generation_config: Generation configuration
            use_kv_cache: Whether to use KV cache
            return_timing_info: Whether to return timing information
            
        Returns:
            generated_ids: Generated token IDs
            timing_info: Optional timing information dictionary
        """
        if generation_config is None:
            generation_config = GenerationConfig()
        
        start_time = time.time()
        batch_size = input_ids.shape[0]
        
        # Determine chunk size
        chunk_size = generation_config.chunk_size or self.config.chunk_size
        
        timing_info = {
            'prefill_time': 0.0,
            'generation_time': 0.0,
            'total_time': 0.0,
            'tokens_per_second': 0.0,
            'prefill_tokens_per_second': 0.0
        }
        
        with torch.no_grad():
            if use_kv_cache:
                # Use chunked generation with KV cache
                prefill_start = time.time()
                
                generated_ids = chunked_generate_with_kv_cache(
                    model=self.chunked_model,
                    input_ids=input_ids,
                    max_new_tokens=generation_config.max_new_tokens,
                    chunk_size=chunk_size,
                    temperature=generation_config.temperature,
                    top_k=generation_config.top_k,
                    top_p=generation_config.top_p,
                    use_flash_attention=self.config.use_flash_attention
                )
                
                total_time = time.time() - start_time
                timing_info['total_time'] = total_time
                timing_info['tokens_per_second'] = generation_config.max_new_tokens / total_time
                
            else:
                # Fallback to regular generation without KV cache
                current_ids = input_ids
                
                prefill_start = time.time()
                
                for _ in range(generation_config.max_new_tokens):
                    # Get logits for current sequence
                    if current_ids.shape[1] > chunk_size:
                        logits = self.chunked_model.chunked_prefill_forward(
                            current_ids,
                            use_flash_attention=self.config.use_flash_attention,
                            return_last_token_only=True
                        )
                    else:
                        logits = self.chunked_model.forward(
                            current_ids,
                            use_flash_attention=self.config.use_flash_attention
                        )
                        logits = logits[:, -1:, :]  # Last token only
                    
                    # Apply temperature
                    if generation_config.temperature != 1.0:
                        logits = logits / generation_config.temperature
                    
                    # Apply top-k filtering
                    if generation_config.top_k is not None:
                        top_k_logits, top_k_indices = torch.topk(logits, generation_config.top_k, dim=-1)
                        logits = torch.full_like(logits, float('-inf'))
                        logits.scatter_(-1, top_k_indices, top_k_logits)
                    
                    # Apply top-p filtering
                    if generation_config.top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                        sorted_indices_to_remove[:, :, 1:] = sorted_indices_to_remove[:, :, :-1].clone()
                        sorted_indices_to_remove[:, :, 0] = False
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                        logits = logits.masked_fill(indices_to_remove, float('-inf'))
                    
                    # Sample next token
                    if generation_config.do_sample:
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, 1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    # Append to sequence
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    
                    # Check for EOS token
                    if (generation_config.eos_token_id is not None and 
                        (next_token == generation_config.eos_token_id).all()):
                        break
                
                generated_ids = current_ids
                total_time = time.time() - start_time
                timing_info['total_time'] = total_time
                timing_info['tokens_per_second'] = generation_config.max_new_tokens / total_time
        
        # Update statistics
        self.stats['total_tokens_processed'] += generated_ids.numel()
        self.stats['total_time'] += timing_info['total_time']
        self.stats['num_requests'] += 1
        
        if return_timing_info:
            return generated_ids, timing_info
        else:
            return generated_ids
    
    def batch_generate(self,
                      input_batch: List[torch.Tensor],
                      generation_config: Optional[GenerationConfig] = None,
                      max_batch_size: Optional[int] = None) -> List[torch.Tensor]:
        """
        Generate text for a batch of variable-length inputs
        
        Args:
            input_batch: List of input tensors with different lengths
            generation_config: Generation configuration
            max_batch_size: Maximum batch size for processing
            
        Returns:
            List of generated sequences
        """
        if generation_config is None:
            generation_config = GenerationConfig()
        
        max_batch_size = max_batch_size or self.config.max_batch_size
        results = []
        
        # Process in mini-batches
        for i in range(0, len(input_batch), max_batch_size):
            mini_batch = input_batch[i:i + max_batch_size]
            
            # Pad sequences to same length
            max_len = max(seq.size(0) for seq in mini_batch)
            padded_batch = []
            
            for seq in mini_batch:
                if seq.size(0) < max_len:
                    # Pad with pad_token_id
                    padded_seq = torch.cat([
                        seq,
                        torch.full((max_len - seq.size(0),), 
                                 generation_config.pad_token_id,
                                 dtype=seq.dtype, device=seq.device)
                    ])
                else:
                    padded_seq = seq
                padded_batch.append(padded_seq)
            
            # Stack into batch tensor
            batch_tensor = torch.stack(padded_batch, dim=0)
            
            # Generate for batch
            generated_batch = self.generate(
                batch_tensor, 
                generation_config=generation_config,
                use_kv_cache=True
            )
            
            # Extract individual results
            for j, original_seq in enumerate(mini_batch):
                original_len = original_seq.size(0)
                # Remove padding and extract generated part
                generated_seq = generated_batch[j]
                results.append(generated_seq)
        
        return results
    
    def benchmark(self,
                 batch_sizes: List[int] = [1, 2, 4, 8],
                 sequence_lengths: List[int] = [128, 256, 512, 1024],
                 max_new_tokens: int = 100,
                 num_trials: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Benchmark inference performance across different configurations
        
        Args:
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of input sequence lengths to test
            max_new_tokens: Number of tokens to generate
            num_trials: Number of trials for averaging
            
        Returns:
            Performance results dictionary
        """
        results = {}
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens)
        
        print("Benchmarking Optimized Inference Engine")
        print("=" * 50)
        
        for batch_size in batch_sizes:
            if batch_size > self.config.max_batch_size:
                continue
                
            results[f'batch_{batch_size}'] = {}
            
            for seq_len in sequence_lengths:
                if seq_len > self.config.max_seq_len:
                    continue
                
                print(f"Testing batch_size={batch_size}, seq_len={seq_len}")
                
                # Create test input
                input_ids = torch.randint(
                    0, self.config.vocab_size,
                    (batch_size, seq_len),
                    device=self.config.device
                )
                
                # Warm up
                for _ in range(2):
                    _ = self.generate(input_ids, generation_config)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Benchmark
                times = []
                for _ in range(num_trials):
                    start_time = time.time()
                    _, timing_info = self.generate(
                        input_ids, 
                        generation_config, 
                        return_timing_info=True
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.time() - start_time)
                
                avg_time = sum(times) / len(times)
                tokens_per_second = max_new_tokens / avg_time
                
                results[f'batch_{batch_size}'][f'seq_{seq_len}'] = {
                    'avg_time': avg_time,
                    'tokens_per_second': tokens_per_second,
                    'throughput': batch_size * tokens_per_second
                }
                
                print(f"  Average time: {avg_time:.4f}s")
                print(f"  Tokens/sec: {tokens_per_second:.2f}")
                print(f"  Throughput: {batch_size * tokens_per_second:.2f} tokens/sec")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.stats.copy()
        if stats['total_time'] > 0:
            stats['average_tokens_per_second'] = stats['total_tokens_processed'] / stats['total_time']
        else:
            stats['average_tokens_per_second'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.stats = {
            'total_tokens_processed': 0,
            'total_time': 0.0,
            'num_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }


def create_inference_engine(
    model_config: Optional[ModelConfig] = None,
    chunk_size: Optional[int] = None,
    enable_flash_attention: bool = True
) -> OptimizedInferenceEngine:
    """
    Create an optimized inference engine
    
    Args:
        model_config: Model configuration
        chunk_size: Chunk size for prefill (if None, uses config default)
        enable_flash_attention: Whether to enable Flash Attention 2
        
    Returns:
        OptimizedInferenceEngine instance
    """
    from src.config import Configs
    
    if model_config is None:
        model_config = Configs.medium()
    
    if chunk_size is not None:
        model_config.chunk_size = chunk_size
    
    model_config.use_flash_attention = enable_flash_attention
    
    # Create chunked model directly
    model = ChunkedPrefillMiniLLM(model_config, chunk_size=model_config.chunk_size)
    
    return OptimizedInferenceEngine(model, model_config)


# Example usage and testing
if __name__ == "__main__":
    # Test the inference engine
    print("Testing Optimized Inference Engine")
    
    # Create engine
    engine = create_inference_engine()
    
    # Create test input
    batch_size = 2
    seq_len = 256
    input_ids = torch.randint(0, engine.config.vocab_size, (batch_size, seq_len), 
                             device=engine.config.device)
    
    # Test generation
    generation_config = GenerationConfig(max_new_tokens=50, temperature=1.0)
    
    print("Generating text...")
    generated_ids, timing_info = engine.generate(
        input_ids, 
        generation_config=generation_config,
        return_timing_info=True
    )
    
    print(f"Generated shape: {generated_ids.shape}")
    print(f"Generation time: {timing_info['total_time']:.4f}s")
    print(f"Tokens per second: {timing_info['tokens_per_second']:.2f}")
    
    # Test batch generation
    print("\nTesting batch generation...")
    input_batch = [
        torch.randint(0, engine.config.vocab_size, (128,), device=engine.config.device),
        torch.randint(0, engine.config.vocab_size, (256,), device=engine.config.device),
        torch.randint(0, engine.config.vocab_size, (192,), device=engine.config.device),
    ]
    
    batch_results = engine.batch_generate(input_batch, generation_config)
    print(f"Batch results shapes: {[r.shape for r in batch_results]}")
    
    # Show statistics
    print("\nEngine statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
