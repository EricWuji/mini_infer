"""
MiniLLM with Flash Attention and KV Cache - 使用示例
展示如何在实际应用中使用整合后的功能
"""
import torch
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.mini_llm import MiniLLM
from cache.kv_cache import KVCache
from config import Configs

class TextGenerator:
    """基于MiniLLM的文本生成器"""
    
    def __init__(self, config=None, use_flash_attention=True):
        """
        初始化生成器
        
        Args:
            config: 模型配置，默认使用small配置
            use_flash_attention: 是否使用Flash Attention
        """
        if config is None:
            config = Configs.small()
        
        self.config = config
        self.model = MiniLLM(config)
        self.model.eval()
        self.use_flash_attention = use_flash_attention
        
        print(f"已加载模型:")
        print(f"  设备: {config.device}")
        print(f"  维度: {config.dim_model}")
        print(f"  头数: {config.num_heads}")
        print(f"  层数: {config.num_layers}")
        print(f"  Flash Attention: {'启用' if use_flash_attention else '禁用'}")
    
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, do_sample=True):
        """
        生成文本
        
        Args:
            prompt_ids: [batch_size, prompt_len] prompt的token IDs
            max_new_tokens: 生成的最大token数
            temperature: 温度参数
            do_sample: 是否随机采样
            
        Returns:
            generated_ids: [batch_size, prompt_len + generated_len] 生成的token IDs
        """
        batch_size, prompt_len = prompt_ids.shape
        device = prompt_ids.device
        
        # 创建KV Cache
        kv_cache = KVCache(self.config, batch_size=batch_size)
        
        print(f"开始生成:")
        print(f"  Prompt长度: {prompt_len}")
        print(f"  最大生成token: {max_new_tokens}")
        print(f"  批次大小: {batch_size}")
        
        # 处理prompt
        with torch.no_grad():
            logits = self.model(
                prompt_ids, 
                kv_cache=kv_cache, 
                use_flash_attention=self.use_flash_attention
            )
        
        # 存储所有生成的token
        all_tokens = [prompt_ids]
        
        # 逐个生成token
        for step in range(max_new_tokens):
            # 从最后一个位置的logits中采样
            next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                # 随机采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪心解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            all_tokens.append(next_token)
            
            # 获取下一个token的logits（用于下一步生成）
            with torch.no_grad():
                logits = self.model(
                    next_token,
                    kv_cache=kv_cache,
                    use_flash_attention=self.use_flash_attention
                )
            
            # 每10步打印进度
            if (step + 1) % 10 == 0 or step < 5:
                print(f"  生成进度: {step + 1}/{max_new_tokens}")
        
        # 拼接所有token
        generated_ids = torch.cat(all_tokens, dim=1)
        
        print(f"生成完成! 总长度: {generated_ids.shape[1]}")
        return generated_ids
    
    def batch_inference(self, input_ids_list):
        """
        批量推理（非生成任务）
        
        Args:
            input_ids_list: list of [seq_len] 多个序列的token IDs
            
        Returns:
            outputs: list of [seq_len, vocab_size] 每个序列的logits
        """
        # 填充到相同长度
        max_len = max(len(ids) for ids in input_ids_list)
        batch_size = len(input_ids_list)
        
        padded_inputs = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.config.device)
        
        for i, ids in enumerate(input_ids_list):
            padded_inputs[i, :len(ids)] = torch.tensor(ids, device=self.config.device)
        
        print(f"批量推理: {batch_size}个序列，最大长度{max_len}")
        
        with torch.no_grad():
            outputs = self.model(
                padded_inputs,
                kv_cache=None,  # 批量推理通常不使用KV cache
                use_flash_attention=self.use_flash_attention
            )
        
        # 分割回原始长度
        result_outputs = []
        for i, ids in enumerate(input_ids_list):
            result_outputs.append(outputs[i, :len(ids), :])
        
        return result_outputs


def demo_text_generation():
    """演示文本生成功能"""
    print("=" * 60)
    print("文本生成演示")
    print("=" * 60)
    
    # 创建生成器
    generator = TextGenerator(use_flash_attention=True)
    
    # 模拟prompt
    batch_size = 2
    prompt_len = 20
    vocab_size = generator.config.vocab_size
    
    # 随机生成prompt (实际使用中这里是编码后的文本)
    prompt_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), 
                              device=generator.config.device)
    
    print(f"Prompt IDs形状: {prompt_ids.shape}")
    
    # 生成文本
    generated_ids = generator.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=30,
        temperature=0.8,
        do_sample=True
    )
    
    print(f"生成结果形状: {generated_ids.shape}")
    print("生成完成!")
    
    return generated_ids


def demo_batch_inference():
    """演示批量推理功能"""
    print("\n" + "=" * 60)
    print("批量推理演示")
    print("=" * 60)
    
    generator = TextGenerator(use_flash_attention=True)
    
    # 模拟不同长度的输入序列
    sequences = [
        list(range(10, 25)),     # 长度15
        list(range(20, 40)),     # 长度20
        list(range(5, 35)),      # 长度30
    ]
    
    print("输入序列:")
    for i, seq in enumerate(sequences):
        print(f"  序列{i+1}: 长度={len(seq)}")
    
    # 批量推理
    outputs = generator.batch_inference(sequences)
    
    print("推理结果:")
    for i, output in enumerate(outputs):
        print(f"  序列{i+1} 输出形状: {output.shape}")
    
    return outputs


def performance_demo():
    """性能对比演示"""
    print("\n" + "=" * 60)
    print("性能对比演示")
    print("=" * 60)
    
    config = Configs.small()
    
    # Flash Attention vs 标准注意力
    print("测试Flash Attention vs 标准注意力:")
    
    flash_generator = TextGenerator(config, use_flash_attention=True)
    standard_generator = TextGenerator(config, use_flash_attention=False)
    
    # 测试数据
    batch_size = 2
    prompt_len = 32
    prompt_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), 
                              device=config.device)
    
    import time
    
    # Flash Attention生成
    print("\nFlash Attention生成:")
    start_time = time.time()
    flash_result = flash_generator.generate(prompt_ids, max_new_tokens=20)
    flash_time = time.time() - start_time
    print(f"用时: {flash_time:.4f}s")
    
    # 标准注意力生成
    print("\n标准注意力生成:")
    start_time = time.time() 
    standard_result = standard_generator.generate(prompt_ids, max_new_tokens=20)
    standard_time = time.time() - start_time
    print(f"用时: {standard_time:.4f}s")
    
    # 对比
    print(f"\n性能对比:")
    print(f"Flash Attention用时: {flash_time:.4f}s")
    print(f"标准注意力用时: {standard_time:.4f}s")
    if flash_time < standard_time:
        print(f"Flash Attention更快 {standard_time/flash_time:.2f}x")
    else:
        print(f"标准注意力更快 {flash_time/standard_time:.2f}x")


if __name__ == "__main__":
    print("MiniLLM Flash Attention + KV Cache 使用示例")
    
    try:
        # 演示文本生成
        demo_text_generation()
        
        # 演示批量推理
        demo_batch_inference()
        
        # 演示性能对比
        performance_demo()
        
        print("\n" + "=" * 60)
        print("✅ 所有演示完成!")
        print("🚀 MiniLLM已成功整合Flash Attention和KV Cache")
        print("📝 可以用于实际的文本生成和推理任务")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
