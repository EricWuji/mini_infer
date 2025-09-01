"""
Debug script for chunked prefill implementation
"""
import torch
import sys
sys.path.insert(0, '/home/wuyinqi/mini_infer')

from src.config import Configs
from src.models.mini_llm import MiniLLM
from src.models.chunked_prefill import ChunkedPrefillMiniLLM
from src.cache.kv_cache import KVCache


def debug_chunked_vs_regular():
    """Debug the difference between chunked and regular processing"""
    print("Debugging chunked vs regular processing")
    
    # Use the exact same config as the failing test
    config = Configs.small()
    config.num_layers = 2  # Use more layers for better testing
    config.max_seq_len = 1024
    config.chunk_size = 256
    config.dtype = torch.float16  # This is the default
    
    # Create models
    regular_model = MiniLLM(config)
    chunked_model = ChunkedPrefillMiniLLM(config, chunk_size=256)
    chunked_model.load_state_dict(regular_model.state_dict())
    
    # Put in eval mode
    regular_model.eval()
    chunked_model.eval()
    
    # Test with a sequence that will be chunked (match the failing test)
    batch_size = 2
    seq_len = 512  # This will be split into 2 chunks of 256 each
    
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=config.device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Will be split into {seq_len // config.chunk_size} chunks of size {config.chunk_size}")
    
    # Regular processing - use regular attention first
    with torch.no_grad():
        regular_output = regular_model.forward(input_ids, use_flash_attention=False)
    
    print(f"Regular output shape: {regular_output.shape}")
    print(f"Regular output stats: min={regular_output.min():.6f}, max={regular_output.max():.6f}, mean={regular_output.mean():.6f}")
    
    # Test chunked model implementation without Flash Attention first
    print("\nTesting ChunkedPrefillMiniLLM implementation without Flash Attention:")
    with torch.no_grad():
        chunked_impl_output = chunked_model.chunked_prefill_forward(input_ids, use_flash_attention=False)
    
    print(f"ChunkedPrefillMiniLLM output shape: {chunked_impl_output.shape}")
    print(f"ChunkedPrefillMiniLLM output stats: min={chunked_impl_output.min():.6f}, max={chunked_impl_output.max():.6f}, mean={chunked_impl_output.mean():.6f}")
    
    # Compare with regular
    impl_diff = torch.abs(regular_output - chunked_impl_output)
    impl_max_diff = torch.max(impl_diff).item()
    impl_mean_diff = torch.mean(impl_diff).item()
    
    print(f"Implementation difference (no Flash Attention):")
    print(f"  Max difference: {impl_max_diff:.6f}")
    print(f"  Mean difference: {impl_mean_diff:.6f}")
    
    # Now test with Flash Attention
    print("\nTesting with Flash Attention:")
    with torch.no_grad():
        regular_flash_output = regular_model.forward(input_ids, use_flash_attention=True)
        chunked_flash_output = chunked_model.chunked_prefill_forward(input_ids, use_flash_attention=True)
    
    # Compare Flash results
    flash_diff = torch.abs(regular_flash_output - chunked_flash_output)
    flash_max_diff = torch.max(flash_diff).item()
    flash_mean_diff = torch.mean(flash_diff).item()
    
    print(f"Flash Attention difference:")
    print(f"  Max difference: {flash_max_diff:.6f}")
    print(f"  Mean difference: {flash_mean_diff:.6f}")
    
    return impl_max_diff


if __name__ == "__main__":
    debug_chunked_vs_regular()
