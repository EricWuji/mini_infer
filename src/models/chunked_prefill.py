"""
Chunked Prefill Implementation for MiniLLM

This module implements chunked prefill optimization to improve throughput
by processing long sequences in smaller chunks, which allows for better
memory utilization and parallelization.
"""

import torch
from torch import nn
from typing import Optional, Tuple, List
import math

from src.cache.kv_cache import KVCache
from src.config import ModelConfig
from src.models.mini_llm import MiniLLM


class ChunkedPrefillMiniLLM(MiniLLM):
    """
    MiniLLM with chunked prefill optimization
    """
    
    def __init__(self, config: ModelConfig = None, chunk_size: int = 512):
        """
        Initialize ChunkedPrefillMiniLLM
        
        Args:
            config: Model configuration
            chunk_size: Size of each chunk for prefill processing
        """
        super().__init__(config)
        self.chunk_size = chunk_size
        
    def chunked_prefill_forward(self, 
                               input_ids: torch.Tensor,
                               kv_cache: Optional[KVCache] = None,
                               use_flash_attention: bool = True,
                               return_last_token_only: bool = False) -> torch.Tensor:
        """
        Forward pass with chunked prefill optimization
        
        Args:
            input_ids: [batch_size, seq_len] input token IDs
            kv_cache: KV Cache object for storing key/value pairs
            use_flash_attention: Whether to use Flash Attention 2
            return_last_token_only: If True, only return logits for the last token
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] or [batch_size, 1, vocab_size] if return_last_token_only
        """
        batch_size, total_seq_len = input_ids.shape
        device = input_ids.device
        
        # If sequence is shorter than chunk size, use regular forward pass
        if total_seq_len <= self.chunk_size:
            logits = self.forward(
                input_ids=input_ids,
                kv_cache=kv_cache,
                use_flash_attention=use_flash_attention
            )
            if return_last_token_only:
                return logits[:, -1:, :]  # [batch_size, 1, vocab_size]
            return logits
        
        # Initialize KV cache if not provided
        if kv_cache is None:
            kv_cache = KVCache(self.config, batch_size=batch_size)
            
        # Reset cache to ensure clean state
        kv_cache.reset()
        
        # Process in chunks
        all_logits = []
        num_chunks = math.ceil(total_seq_len / self.chunk_size)
        
        for chunk_idx in range(num_chunks):
            start_pos = chunk_idx * self.chunk_size
            end_pos = min((chunk_idx + 1) * self.chunk_size, total_seq_len)
            
            # Extract chunk
            chunk_input = input_ids[:, start_pos:end_pos]
            chunk_len = end_pos - start_pos
            
            # Process chunk with KV cache
            # The forward method will handle the cache positioning correctly
            chunk_logits = self.forward(
                input_ids=chunk_input,
                kv_cache=kv_cache,
                use_flash_attention=use_flash_attention
            )
            
            # Store logits if needed
            if not return_last_token_only or chunk_idx == num_chunks - 1:
                all_logits.append(chunk_logits)
        
        if return_last_token_only:
            # Return only the last token's logits
            return all_logits[-1][:, -1:, :]  # [batch_size, 1, vocab_size]
        else:
            # Concatenate all logits
            return torch.cat(all_logits, dim=1)  # [batch_size, total_seq_len, vocab_size]


def chunked_generate_with_kv_cache(
    model: MiniLLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    chunk_size: int = 512,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    use_flash_attention: bool = True
) -> torch.Tensor:
    """
    Generate text using chunked prefill with KV cache optimization
    
    Args:
        model: MiniLLM model instance
        input_ids: [batch_size, seq_len] input token IDs
        max_new_tokens: Maximum number of new tokens to generate
        chunk_size: Size of chunks for prefill processing
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        use_flash_attention: Whether to use Flash Attention 2
        
    Returns:
        generated_ids: [batch_size, seq_len + max_new_tokens] generated token IDs
    """
    batch_size, input_len = input_ids.shape
    device = input_ids.device
    
    # Initialize KV cache
    kv_cache = KVCache(model.config, batch_size=batch_size)
    
    # Phase 1: Chunked prefill for input tokens
    if input_len > chunk_size:
        # Process input in chunks
        num_chunks = math.ceil(input_len / chunk_size)
        
        for chunk_idx in range(num_chunks):
            start_pos = chunk_idx * chunk_size
            end_pos = min((chunk_idx + 1) * chunk_size, input_len)
            
            chunk_input = input_ids[:, start_pos:end_pos]
            
            # Process chunk (we don't need logits for prefill, just update cache)
            with torch.no_grad():
                _ = model.forward(
                    input_ids=chunk_input,
                    kv_cache=kv_cache,
                    use_flash_attention=use_flash_attention
                )
        
        # Get logits for the last input token for generation
        last_logits = model.forward(
            input_ids=input_ids[:, -1:],  # Only last token
            kv_cache=kv_cache,
            use_flash_attention=use_flash_attention
        )
    else:
        # Input is short enough to process in one go
        last_logits = model.forward(
            input_ids=input_ids,
            kv_cache=kv_cache,
            use_flash_attention=use_flash_attention
        )
        last_logits = last_logits[:, -1:, :]  # Only last token
    
    # Phase 2: Auto-regressive generation
    generated_tokens = []
    current_logits = last_logits
    
    for _ in range(max_new_tokens):
        # Apply temperature scaling
        if temperature != 1.0:
            current_logits = current_logits / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            top_k_logits, top_k_indices = torch.topk(current_logits, top_k, dim=-1)
            current_logits = torch.full_like(current_logits, float('-inf'))
            current_logits.scatter_(-1, top_k_indices, top_k_logits)
        
        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(current_logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, :, 1:] = sorted_indices_to_remove[:, :, :-1].clone()
            sorted_indices_to_remove[:, :, 0] = False
            
            # Apply the mask
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            current_logits = current_logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample next token
        probs = torch.softmax(current_logits, dim=-1)
        next_token = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, 1)
        generated_tokens.append(next_token)
        
        # Get next logits using the generated token
        current_logits = model.forward(
            input_ids=next_token,
            kv_cache=kv_cache,
            use_flash_attention=use_flash_attention
        )
    
    # Combine input and generated tokens
    if generated_tokens:
        generated_sequence = torch.cat([input_ids] + generated_tokens, dim=1)
    else:
        generated_sequence = input_ids
    
    return generated_sequence


class BatchedChunkedPrefill:
    """
    Batched chunked prefill processor for improved throughput
    """
    
    def __init__(self, 
                 model: MiniLLM,
                 chunk_size: int = 512,
                 max_batch_size: int = 32):
        """
        Initialize batched chunked prefill processor
        
        Args:
            model: MiniLLM model instance
            chunk_size: Size of chunks for processing
            max_batch_size: Maximum batch size for processing
        """
        self.model = model
        self.chunk_size = chunk_size
        self.max_batch_size = max_batch_size
        
    def process_batch(self,
                     input_batch: List[torch.Tensor],
                     use_flash_attention: bool = True) -> List[torch.Tensor]:
        """
        Process a batch of sequences with chunked prefill
        
        Args:
            input_batch: List of input tensors [seq_len] for each sequence
            use_flash_attention: Whether to use Flash Attention 2
            
        Returns:
            List of output logits for each sequence
        """
        results = []
        
        # Process in mini-batches
        for i in range(0, len(input_batch), self.max_batch_size):
            mini_batch = input_batch[i:i + self.max_batch_size]
            
            # Pad sequences to the same length within the mini-batch
            max_len = max(seq.size(0) for seq in mini_batch)
            padded_batch = []
            attention_masks = []
            
            for seq in mini_batch:
                if seq.size(0) < max_len:
                    # Pad with zeros (assuming 0 is padding token)
                    padded_seq = torch.cat([
                        seq, 
                        torch.zeros(max_len - seq.size(0), dtype=seq.dtype, device=seq.device)
                    ])
                    mask = torch.cat([
                        torch.ones(seq.size(0), dtype=torch.bool, device=seq.device),
                        torch.zeros(max_len - seq.size(0), dtype=torch.bool, device=seq.device)
                    ])
                else:
                    padded_seq = seq
                    mask = torch.ones(seq.size(0), dtype=torch.bool, device=seq.device)
                
                padded_batch.append(padded_seq)
                attention_masks.append(mask)
            
            # Stack into batch tensor
            batch_tensor = torch.stack(padded_batch, dim=0)  # [batch_size, max_len]
            
            # Create chunked model if needed
            if not isinstance(self.model, ChunkedPrefillMiniLLM):
                chunked_model = ChunkedPrefillMiniLLM(self.model.config, self.chunk_size)
                chunked_model.load_state_dict(self.model.state_dict())
            else:
                chunked_model = self.model
            
            # Process with chunked prefill
            with torch.no_grad():
                batch_logits = chunked_model.chunked_prefill_forward(
                    input_ids=batch_tensor,
                    use_flash_attention=use_flash_attention
                )
            
            # Extract results for each sequence in the mini-batch
            for j, (original_seq, mask) in enumerate(zip(mini_batch, attention_masks)):
                original_len = original_seq.size(0)
                # Only take logits for the original sequence length
                seq_logits = batch_logits[j, :original_len, :]
                results.append(seq_logits)
        
        return results


def benchmark_chunked_prefill(
    model: MiniLLM,
    sequence_lengths: List[int] = [512, 1024, 2048, 4096],
    chunk_sizes: List[int] = [128, 256, 512, 1024],
    batch_size: int = 4,
    num_trials: int = 10
):
    """
    Benchmark chunked prefill performance
    
    Args:
        model: MiniLLM model instance
        sequence_lengths: List of sequence lengths to test
        chunk_sizes: List of chunk sizes to test
        batch_size: Batch size for testing
        num_trials: Number of trials for averaging
    """
    import time
    
    device = model.device
    results = {}
    
    print("Benchmarking Chunked Prefill Performance")
    print("=" * 50)
    
    for seq_len in sequence_lengths:
        results[seq_len] = {}
        
        # Generate random input
        input_ids = torch.randint(0, model.config.vocab_size, 
                                (batch_size, seq_len), 
                                device=device)
        
        # Test regular forward pass
        model.eval()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        for _ in range(num_trials):
            with torch.no_grad():
                _ = model.forward(input_ids, use_flash_attention=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        regular_time = (time.time() - start_time) / num_trials
        
        results[seq_len]['regular'] = regular_time
        print(f"Seq Length {seq_len}: Regular = {regular_time:.4f}s")
        
        # Test chunked prefill with different chunk sizes
        for chunk_size in chunk_sizes:
            if chunk_size >= seq_len:
                continue
                
            chunked_model = ChunkedPrefillMiniLLM(model.config, chunk_size)
            chunked_model.load_state_dict(model.state_dict())
            chunked_model.eval()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(num_trials):
                with torch.no_grad():
                    _ = chunked_model.chunked_prefill_forward(
                        input_ids, use_flash_attention=True
                    )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            chunked_time = (time.time() - start_time) / num_trials
            
            results[seq_len][f'chunked_{chunk_size}'] = chunked_time
            speedup = regular_time / chunked_time
            print(f"  Chunk {chunk_size} = {chunked_time:.4f}s (speedup: {speedup:.2f}x)")
        
        print()
    
    return results
