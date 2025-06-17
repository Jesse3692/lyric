import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    加载训练好的模型和分词器
    """
    try:
        # 尝试加载为GPT2模型
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
    except:
        # 如果失败，尝试加载为其他模型
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # 确保有pad_token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_lyrics(
    model, 
    tokenizer, 
    prompt="", 
    max_length=200, 
    num_return_sequences=1,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    do_sample=True
):
    """
    生成歌词
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        prompt: 提示文本，作为生成的起点
        max_length: 生成文本的最大长度
        num_return_sequences: 返回的序列数量
        temperature: 温度参数，控制随机性
        top_k: top-k采样参数
        top_p: top-p采样参数
        repetition_penalty: 重复惩罚参数
        do_sample: 是否使用采样
        
    Returns:
        生成的歌词列表
    """
    # 将模型设置为评估模式
    model.eval()
    
    # 检查是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 对提示文本进行编码
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)
    
    # 生成参数
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # 解码生成的序列
    generated_lyrics = []
    for generated_sequence in output_sequences:
        text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        generated_lyrics.append(text)
    
    return generated_lyrics

def format_lyrics(text, line_length=20):
    """
    格式化歌词，使其更易于阅读
    """
    # 按标点符号分割
    import re
    sentences = re.split(r'([，。！？、：；])', text)
    
    # 重组句子，保留标点
    formatted_lines = []
    current_line = ""
    
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            part = sentences[i]
            # 添加标点（如果有）
            if i + 1 < len(sentences):
                part += sentences[i + 1]
            
            # 如果当前行加上新部分超过行长度，开始新行
            if len(current_line) + len(part) > line_length:
                if current_line:
                    formatted_lines.append(current_line)
                current_line = part
            else:
                current_line += part
    
    # 添加最后一行
    if current_line:
        formatted_lines.append(current_line)
    
    # 每4行添加一个空行（形成歌词的段落）
    result = []
    for i, line in enumerate(formatted_lines):
        result.append(line)
        if (i + 1) % 4 == 0:
            result.append("")
    
    return "\n".join(result)

def generate_lyrics_with_style(model_dir, prompt="", **kwargs):
    """
    使用特定风格的模型生成歌词
    """
    model, tokenizer = load_model(model_dir)
    generated_texts = generate_lyrics(model, tokenizer, prompt, **kwargs)
    formatted_lyrics = [format_lyrics(text) for text in generated_texts]
    return formatted_lyrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成歌词")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--prompt", type=str, default="", help="提示文本")
    parser.add_argument("--max_length", type=int, default=200, help="生成文本的最大长度")
    parser.add_argument("--num_sequences", type=int, default=1, help="生成的序列数量")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--output_file", type=str, help="输出文件路径")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(args.model_path)
    
    # 生成歌词
    generated_lyrics = generate_lyrics(
        model, 
        tokenizer, 
        prompt=args.prompt,
        max_length=args.max_length,
        num_return_sequences=args.num_sequences,
        temperature=args.temperature
    )
    
    # 格式化歌词
    formatted_lyrics = [format_lyrics(text) for text in generated_lyrics]
    
    # 输出结果
    for i, lyrics in enumerate(formatted_lyrics):
        print(f"\n--- 生成的歌词 {i+1} ---\n")
        print(lyrics)
        print("\n" + "-" * 40)
    
    # 保存到文件（如果指定）
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, lyrics in enumerate(formatted_lyrics):
                f.write(f"\n--- 生成的歌词 {i+1} ---\n\n")
                f.write(lyrics)
                f.write("\n\n" + "-" * 40 + "\n")
        print(f"歌词已保存到 {args.output_file}")