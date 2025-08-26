#!/usr/bin/env python3
"""
测试启动器 - 正确设置Python路径并运行测试
"""
import sys
import os
import importlib.util

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 动态导入并运行测试模块
def run_test(test_file):
    """动态加载并运行测试文件"""
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    test_module = importlib.util.module_from_spec(spec)
    
    # 设置模块的上下文
    test_module.__file__ = test_file
    sys.modules["test_module"] = test_module
    
    try:
        spec.loader.exec_module(test_module)
        
        # 如果有main函数，运行它
        if hasattr(test_module, '__name__') and test_module.__name__ == 'test_module':
            # 查找并运行main代码
            if hasattr(test_module, 'main'):
                test_module.main()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # 可以指定测试文件，或运行所有测试
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        if os.path.exists(test_file):
            print(f"🧪 运行测试: {test_file}")
            run_test(test_file)
        else:
            print(f"❌ 测试文件不存在: {test_file}")
    else:
        print("🧪 运行所有测试...")
        
        # 运行所有测试文件
        test_files = [
            "tests/test_integration.py",
            "tests/test_kvcache.py"
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"\n🧪 运行测试: {test_file}")
                success = run_test(test_file)
                if not success:
                    print(f"❌ 测试失败: {test_file}")
                    break
            else:
                print(f"⚠️ 跳过不存在的测试文件: {test_file}")
        
        print("\n✅ 所有测试完成!")
