"""
简单的测试脚本 - 验证新项目结构
"""
import os
import sys

# 确保我们可以导入src模块
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

def test_imports():
    """测试所有主要组件的导入"""
    print("🧪 测试导入...")
    
    try:
        from config import ModelConfig, Configs, DEFAULT_CONFIG
        print("✅ 配置模块导入成功")
        
        from cache.kv_cache import KVCache
        print("✅ KV Cache模块导入成功")
        
        from models.mini_llm import MiniLLM, create_model
        print("✅ MiniLLM模块导入成功")
        
        from attention.flash_attention import flash_attention_with_kv_cache, multi_head_flash_attention
        print("✅ Flash Attention模块导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    try:
        import torch
        from config import Configs
        from models.mini_llm import MiniLLM
        from cache.kv_cache import KVCache
        
        # 创建小配置
        config = Configs.small()
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 创建模型和缓存
        model = MiniLLM(config)
        kv_cache = KVCache(config, batch_size=2)
        
        # 创建测试输入
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=config.device)
        
        # 测试前向传播
        with torch.no_grad():
            output = model(input_ids, kv_cache=None, use_flash_attention=False)
            print(f"✅ 标准注意力输出形状: {output.shape}")
            
            output_flash = model(input_ids, kv_cache=None, use_flash_attention=True)
            print(f"✅ Flash Attention输出形状: {output_flash.shape}")
            
            # 测试KV Cache
            kv_cache.reset()
            output_cache = model(input_ids[:, :16], kv_cache=kv_cache, use_flash_attention=True)
            print(f"✅ KV Cache输出形状: {output_cache.shape}")
        
        print("✅ 基本功能测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_project_structure():
    """验证项目结构"""
    print("\n🧪 验证项目结构...")
    
    expected_structure = {
        'src/': ['config.py', '__init__.py'],
        'src/models/': ['mini_llm.py', '__init__.py'],
        'src/attention/': ['flash_attention.py', 'tiled_attention.py', '__init__.py'],
        'src/cache/': ['kv_cache.py', '__init__.py'],
        'tests/': ['test_integration.py', 'test_kvcache.py'],
        'benchmarks/': ['performance_analysis.py', 'benchmark_kv_cache.py', 'run_benchmark.py'],
        'examples/': ['demo_usage.py', 'example_usage.py'],
        'docs/': ['INTEGRATION_SUMMARY.md', 'FLASH_ATTENTION_FIXES.md', 'KV_CACHE_OPTIMIZATION.md']
    }
    
    all_good = True
    for directory, files in expected_structure.items():
        if not os.path.exists(directory):
            print(f"❌ 目录缺失: {directory}")
            all_good = False
            continue
            
        for file in files:
            file_path = os.path.join(directory, file)
            if not os.path.exists(file_path):
                print(f"⚠️ 文件缺失: {file_path}")
            else:
                print(f"✅ {file_path}")
    
    if all_good:
        print("✅ 项目结构验证通过!")
    
    return all_good

if __name__ == "__main__":
    print("🚀 MiniLLM 新项目结构测试")
    print("=" * 50)
    
    success = True
    
    # 测试项目结构
    success &= test_project_structure()
    
    # 测试导入
    success &= test_imports()
    
    # 测试基本功能
    success &= test_basic_functionality()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！新项目结构工作正常！")
        print("\n📋 使用说明:")
        print("1. 从项目根目录导入: from src import MiniLLM, KVCache")
        print("2. 运行示例: python examples/demo_usage.py")
        print("3. 运行基准测试: python benchmarks/performance_analysis.py")
        print("4. 使用Makefile: make help")
    else:
        print("❌ 部分测试失败，请检查项目结构")
