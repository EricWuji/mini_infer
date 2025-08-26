# MiniLLM - Lightweight LLM with Flash Attention & KV Cache

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, lightweight implementation of a Large Language Model with integrated Flash Attention 2 and KV Cache for efficient inference.

## ✨ Features

- 🚀 **Flash Attention 2**: Memory-efficient attention computation with up to 99.6% memory savings
- 💾 **KV Cache**: High-speed incremental inference for text generation (148+ tokens/s)
- 🔧 **Modular Design**: Clean architecture with separated concerns
- ⚡ **GPU Optimized**: CUDA support with automatic mixed precision
- 📊 **Production Ready**: Comprehensive testing and benchmarking
- 🔄 **Backward Compatible**: Easy migration from existing implementations

## 📦 Installation

### From Source
```bash
git clone https://github.com/EricWuji/mini_infer.git
cd mini_infer
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/EricWuji/mini_infer.git
cd mini_infer
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Basic Usage
```python
from src import MiniLLM, KVCache, Configs

# Create model with default small configuration
config = Configs.small()
model = MiniLLM(config)

# Create KV Cache for efficient generation
kv_cache = KVCache(config, batch_size=2)

# Forward pass with Flash Attention + KV Cache
output = model(
    input_ids,
    kv_cache=kv_cache,
    use_flash_attention=True,
    flash_block_size=128
)
```

### Text Generation
```python
from examples.demo_usage import TextGenerator

# Create generator
generator = TextGenerator(use_flash_attention=True)

# Generate text
result = generator.generate(
    prompt_ids=prompt_ids,
    max_new_tokens=50,
    temperature=0.8
)
```

## 📁 Project Structure

```
mini_infer/
├── src/                     # Source code
│   ├── models/             # Model implementations
│   │   └── mini_llm.py     # Main MiniLLM model
│   ├── attention/          # Attention mechanisms
│   │   ├── flash_attention.py  # Flash Attention 2 implementation
│   │   └── tiled_attention.py  # Original tiled attention
│   ├── cache/              # Caching systems
│   │   └── kv_cache.py     # KV Cache implementation
│   └── config.py           # Configuration classes
├── tests/                  # Unit tests
│   ├── test_integration.py # Integration tests
│   └── test_kvcache.py     # KV Cache tests
├── benchmarks/             # Performance benchmarks
│   ├── performance_analysis.py
│   └── benchmark_kv_cache.py
├── examples/               # Usage examples
│   ├── demo_usage.py       # Complete usage demo
│   └── example_usage.py    # Basic examples
├── docs/                   # Documentation
│   ├── INTEGRATION_SUMMARY.md
│   ├── FLASH_ATTENTION_FIXES.md
│   └── KV_CACHE_OPTIMIZATION.md
└── README.md               # This file
```

## 🎯 Performance

### Memory Optimization
| Sequence Length | Standard Attention | Flash Attention | Memory Savings |
|----------------|------------------|-----------------|----------------|
| 256            | 4.00 MB          | 1.00 MB         | 75.0%          |
| 512            | 16.00 MB         | 1.00 MB         | 93.8%          |
| 1024           | 64.00 MB         | 1.00 MB         | 98.4%          |
| 2048           | 256.00 MB        | 1.00 MB         | 99.6%          |

### Generation Speed
- **KV Cache Generation**: 148+ tokens/s
- **Memory Usage**: 64MB KV Cache supports 2048-length sequences  
- **Batch Processing**: Up to 16 concurrent sequences

## 🧪 Testing

Run all tests:
```bash
# Basic integration test
python tests/test_integration.py

# Performance analysis
python benchmarks/performance_analysis.py

# Usage examples
python examples/demo_usage.py
```

## 📊 Benchmarks

Compare different configurations:
```bash
cd benchmarks
python performance_analysis.py
```

This will test:
- Flash Attention vs Standard Attention
- Different block sizes optimization
- KV Cache vs non-cached inference
- Memory usage analysis

## ⚙️ Configuration

### Model Configurations
```python
from src.config import Configs

# Predefined configurations
small_config = Configs.small()    # 512 dim, 8 heads
medium_config = Configs.medium()  # 1024 dim, 16 heads  
large_config = Configs.large()    # 2048 dim, 32 heads

# Custom configuration
from src.config import ModelConfig

custom_config = ModelConfig(
    dim_model=1024,
    num_heads=16,
    num_layers=12,
    max_seq_len=2048,
    vocab_size=50000
)
```

### Flash Attention Block Size Optimization
- **Short sequences (<256)**: Use block size 64-128
- **Long sequences (>256)**: Use block size 128-256
- **Optimal**: Block size 256 performs best in most cases

## 🔧 Advanced Usage

### Custom Attention Mechanisms
```python
from src.attention.flash_attention import flash_attention_with_kv_cache

# Direct flash attention call
output = flash_attention_with_kv_cache(
    q=query_tensor,
    k=key_tensor, 
    v=value_tensor,
    causal_mask=mask,
    block_size=128
)
```

### KV Cache Management
```python
from src.cache.kv_cache import KVCache

# Create cache
cache = KVCache(config, batch_size=4)

# Reset specific batches
cache.reset(batch_indices=[0, 2])

# Get sequence length
seq_len = cache.get_seq_len(batch_idx=0)
```

## 📖 Documentation

- [Integration Summary](docs/INTEGRATION_SUMMARY.md) - Complete integration overview
- [Flash Attention Details](docs/FLASH_ATTENTION_FIXES.md) - Technical implementation details
- [KV Cache Optimization](docs/KV_CACHE_OPTIMIZATION.md) - Cache system design

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Flash Attention implementation inspired by the original paper by Tri Dao et al.
- KV Cache design follows Hugging Face Transformers conventions
- Project structure follows modern Python packaging standards

## 📞 Support

If you encounter any issues or have questions:

1. Check the [documentation](docs/)
2. Run the test suite to verify your setup
3. Open an issue on GitHub with detailed information

---

**Made with ❤️ by EricWuji**
