# MiniLLM with Configurable Context Length

This repository contains a configurable implementation of MiniLLM with KV Cache support. The system is designed to easily scale with hardware upgrades by simply changing configuration parameters.

## 🚀 Quick Start

### Basic Usage
```python
from config import Configs
from MiniLLM import create_model
from KVCache import KVCache

# Create a small model (default)
config = Configs.small()
model = create_model(config)

# Create KV cache
kv_cache = KVCache(config)
```

### Running Benchmarks

#### Option 1: Use the benchmark script
```bash
# Test small configuration
python run_benchmark.py --config small

# Test medium configuration  
python run_benchmark.py --config medium

# Test with custom parameters
python run_benchmark.py --config custom --max-seq-len 4096 --max-kv-cache-len 8192
```

#### Option 2: Run the test directly
```bash
python test_kvcache.py
```

## 📊 Available Configurations

### Small (Default - Testing/Development)
- Max sequence length: 1,024
- Max KV cache length: 2,048
- Model dimension: 512
- Attention heads: 8
- Memory usage: ~Low

### Medium (Mid-range Hardware)
- Max sequence length: 2,048
- Max KV cache length: 4,096
- Model dimension: 1,024
- Attention heads: 16
- Memory usage: ~Medium

### Large (High-end Hardware)
- Max sequence length: 4,096
- Max KV cache length: 8,192
- Model dimension: 2,048
- Attention heads: 32
- Memory usage: ~High

### XLarge (Future Hardware Upgrades)
- Max sequence length: 8,192
- Max KV cache length: 16,384
- Model dimension: 4,096
- Attention heads: 64
- Memory usage: ~Very High

## 🔧 Customizing for Your Hardware

### Method 1: Modify config.py
Edit the configuration in `config.py`:
```python
# For your specific hardware requirements
CUSTOM_CONFIG = ModelConfig(
    max_seq_len=8192,        # Your desired context length
    max_kv_cache_len=16384,  # 2x seq_len recommended
    dim_model=2048,          # Adjust based on GPU memory
    num_heads=32,            # Keep divisible by dim_model
    vocab_size=50000,        # Your tokenizer vocab size
)
```

### Method 2: Create config programmatically
```python
from config import ModelConfig

# Create your custom configuration
my_config = ModelConfig(
    max_seq_len=16384,       # 16K context length
    max_kv_cache_len=32768,  # 32K KV cache
    dim_model=4096,
    num_heads=64,
    vocab_size=100000
)

model = create_model(my_config)
```

## 💾 Memory Usage Guidelines

| Configuration | GPU Memory (approx) | Context Length | Use Case |
|---------------|-------------------|----------------|----------|
| Small         | 2-4 GB           | 1K            | Development/Testing |
| Medium        | 8-12 GB          | 2K            | Small-scale inference |
| Large         | 16-24 GB         | 4K            | Production inference |
| XLarge        | 32+ GB           | 8K+           | Future scaling |

## 🔄 Upgrading for New Hardware

When you upgrade your hardware, simply:

1. **Update your configuration**:
   ```python
   # Change from small to large configuration
   config = Configs.large()  # Instead of Configs.small()
   ```

2. **Or create custom configuration**:
   ```python
   config = ModelConfig(
       max_seq_len=32768,      # Your new context length
       max_kv_cache_len=65536, # 2x recommended
       # ... other parameters
   )
   ```

3. **Test the configuration**:
   ```bash
   python run_benchmark.py --config large
   ```

## 🎯 Key Features

- **Centralized Configuration**: All model parameters in one place
- **Backward Compatibility**: Legacy parameter support
- **Automatic Validation**: Config validation prevents errors
- **Memory Efficient**: KV Cache with vectorized operations
- **Scalable**: Easy hardware upgrades
- **Flexible**: Custom configurations supported

## 📈 Performance Notes

- KV Cache provides 2-3x speedup for long sequences
- Memory usage scales quadratically with context length
- Attention computation is the main bottleneck for very long sequences
- Batch size should be adjusted based on available GPU memory

## 🤝 Contributing

Feel free to add new predefined configurations or optimize the implementation for different hardware setups!
