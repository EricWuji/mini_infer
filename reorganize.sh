#!/bin/bash

# 项目重构清理脚本
# 这个脚本会备份旧文件并清理项目根目录

echo "🚀 开始项目重构清理..."

# 创建备份目录
mkdir -p backup_old_structure
echo "📁 创建备份目录: backup_old_structure"

# 备份旧的根级文件
echo "💾 备份旧文件..."
mv MiniLLM.py backup_old_structure/ 2>/dev/null || true
mv KVCache.py backup_old_structure/ 2>/dev/null || true  
mv flash_attention_kv.py backup_old_structure/ 2>/dev/null || true
mv tiled_attn.py backup_old_structure/ 2>/dev/null || true
mv config.py backup_old_structure/ 2>/dev/null || true

# 备份测试文件
mv test_integration.py backup_old_structure/ 2>/dev/null || true
mv test_kvcache.py backup_old_structure/ 2>/dev/null || true

# 备份基准测试文件
mv benchmark_kv_cache.py backup_old_structure/ 2>/dev/null || true
mv run_benchmark.py backup_old_structure/ 2>/dev/null || true
mv performance_analysis.py backup_old_structure/ 2>/dev/null || true

# 备份示例文件
mv demo_usage.py backup_old_structure/ 2>/dev/null || true
mv example_usage.py backup_old_structure/ 2>/dev/null || true

# 备份文档文件
mv FLASH_ATTENTION_FIXES.md backup_old_structure/ 2>/dev/null || true
mv KV_CACHE_OPTIMIZATION.md backup_old_structure/ 2>/dev/null || true
mv INTEGRATION_SUMMARY.md backup_old_structure/ 2>/dev/null || true

# 更新gitignore和README
mv README.md backup_old_structure/README_old.md 2>/dev/null || true
cp README_NEW.md README.md
mv .gitignore backup_old_structure/gitignore_old 2>/dev/null || true  
cp .gitignore_new .gitignore

# 清理临时文件
rm -f README_NEW.md .gitignore_new

echo "✅ 项目重构完成!"
echo ""
echo "📊 新的项目结构:"
echo "├── src/                    # 源代码"
echo "│   ├── models/            # 模型实现"
echo "│   ├── attention/         # 注意力机制"  
echo "│   ├── cache/             # 缓存系统"
echo "│   └── config.py          # 配置"
echo "├── tests/                 # 测试"
echo "├── benchmarks/            # 基准测试"
echo "├── examples/              # 使用示例"
echo "├── docs/                  # 文档"
echo "└── backup_old_structure/  # 旧文件备份"
echo ""
echo "🛠️ 下一步操作:"
echo "1. conda activate inference"
echo "2. python tests/test_integration.py  # 验证功能"
echo "3. python examples/demo_usage.py     # 运行示例"
echo "4. make help                         # 查看可用命令"
