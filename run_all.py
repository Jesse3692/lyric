import os
import logging
import pandas as pd
import re
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    TextDataset, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
import gradio as gr

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
CSV_FILE = "113be125e2eb2adc8de3baf0fbc88e8b_4_8.csv"
OUTPUT_DIR = "processed_lyrics"
MODEL_DIR = "trained_model"
EPOCHS = 3
BATCH_SIZE = 4
MODEL_NAME = "gpt2"  # 或者使用中文模型如 "uer/gpt2-chinese-cluecorpussmall"

# 1. 数据预处理函数
def clean_lyrics(text):
    """清洗歌词文本，只保留中英文和标点符号"""
    if pd.isna(text):
        return ""
    
    # 保留中文、英文、数字、常见标点
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、：；''""（）【】《》\s]'
    cleaned_text = re.sub(pattern, '', str(text))
    # 去除多余空白
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def process_csv_lyrics(csv_file, output_dir):
    """处理CSV格式的歌词数据"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    logger.info(f"正在读取CSV文件: {csv_file}")
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码
        try:
            df = pd.read_csv(csv_file, encoding='gbk')
        except:
            df = pd.read_csv(csv_file, encoding='latin1')
    
    logger.info(f"CSV文件包含 {len(df)} 行数据")
    
    # 查看CSV文件的列
    logger.info("CSV文件的列名: %s", df.columns.tolist())
    
    # 确定歌词列
    lyrics_column = None
    possible_lyrics_columns = ['lyrics', 'lyric', '歌词', 'text', '内容']
    
    for col in possible_lyrics_columns:
        if col in df.columns:
            lyrics_column = col
            break
    
    if lyrics_column is None:
        # 如果没有找到明确的歌词列，假设最长的文本列是歌词
        text_lengths = {}
        for col in df.columns:
            if df[col].dtype == 'object':  # 只考虑文本列
                avg_length = df[col].astype(str).str.len().mean()
                text_lengths[col] = avg_length
        
        if text_lengths:
            lyrics_column = max(text_lengths, key=text_lengths.get)
    
    logger.info(f"使用 '{lyrics_column}' 列作为歌词")
    
    # 清洗歌词
    logger.info("正在清洗歌词...")
    df['cleaned_lyrics'] = df[lyrics_column].apply(clean_lyrics)
    
    # 保存所有清洗后的歌词到一个文件
    all_lyrics_file = os.path.join(output_dir, "all_lyrics.txt")
    with open(all_lyrics_file, 'w', encoding='utf-8') as f:
        for lyrics in df['cleaned_lyrics']:
            if lyrics and len(lyrics) > 10:  # 只保存非空且长度合理的歌词
                f.write(lyrics + "\n\n")
    
    logger.info(f"所有歌词已保存到: {all_lyrics_file}")
    return all_lyrics_file

# 2. 模型训练函数
def load_dataset(data_path, tokenizer, block_size=128):
    """加载数据集"""
    logger.info(f"正在从 {data_path} 加载数据集...")
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=block_size,
    )
    return dataset

def train_model(data_path, output_dir, num_train_epochs=3, per_device_train_batch_size=4):
    """训练歌词生成模型"""
    # 检查是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载预训练模型和分词器
    if MODEL_NAME == "gpt2":
        # 使用英文GPT-2模型
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        # 设置特殊token
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # 使用中文预训练模型
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    train_dataset = load_dataset(data_path, tokenizer)
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=10000,
        save_total_limit=2,
        logging_steps=500,
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # 开始训练
    logger.info("开始训练模型...")
    trainer.train()
    
    # 保存模型和分词器
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"模型已保存到 {output_dir}")
    
    return model, tokenizer

# 3. 歌词生成函数
def generate_lyrics(model, tokenizer, prompt="", max_length=200, temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.2):
    """生成歌词"""
    # 将模型设置为评估模式
    model.eval()
    
    # 检查是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 确保提示不为空，如果为空，添加一个空格
    if not prompt:
        prompt = " "
    
    # 对提示文本进行编码
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成参数
    output_sequences = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        no_repeat_ngram_size=2,  # 避免重复的n-gram
    )
    
    # 解码生成的序列
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text
def format_lyrics(text, line_length=20):
    """格式化歌词，使其更易于阅读"""
    # 按标点符号分割
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

# 4. Gradio界面函数
def create_ui(model, tokenizer):
    """创建Gradio界面"""
    def generate_lyrics_ui(prompt, max_length, temperature, top_k, top_p, repetition_penalty):
        try:
            generated_text = generate_lyrics(
                model, 
                tokenizer, 
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            formatted_lyrics = format_lyrics(generated_text)
            return formatted_lyrics
        except Exception as e:
            return f"生成歌词时出错: {str(e)}"
    
    with gr.Blocks(title="歌词生成器") as app:
        gr.Markdown("# 歌词生成器")
        gr.Markdown("输入提示，生成独特的歌词！")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    lines=2, 
                    label="提示文本", 
                    placeholder="输入一些文字作为歌词的开头...",
                    info="可以为空，模型会自动生成完整歌词"
                )
                
                with gr.Accordion("高级设置", open=False):
                    max_length = gr.Slider(
                        minimum=50, 
                        maximum=500, 
                        value=200, 
                        step=10, 
                        label="最大长度"
                    )
                    temperature = gr.Slider(
                        minimum=0.1, 
                        maximum=2.0, 
                        value=1.0, 
                        step=0.1, 
                        label="温度 (创造性)",
                        info="较高的值会产生更多样化但可能不太连贯的文本"
                    )
                    top_k = gr.Slider(
                        minimum=1, 
                        maximum=100, 
                        value=50, 
                        step=1, 
                        label="Top-K"
                    )
                    top_p = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.95, 
                        step=0.05, 
                        label="Top-P (核采样)"
                    )
                    repetition_penalty = gr.Slider(
                        minimum=1.0, 
                        maximum=2.0, 
                        value=1.2, 
                        step=0.1, 
                        label="重复惩罚",
                        info="较高的值会减少重复"
                    )
                
                generate_button = gr.Button("生成歌词", variant="primary")
            
            with gr.Column(scale=2):
                output = gr.Textbox(
                    lines=20, 
                    label="生成的歌词", 
                    interactive=False
                )
        
        # 事件处理
        generate_button.click(
            fn=generate_lyrics_ui,
            inputs=[
                prompt, 
                max_length, 
                temperature, 
                top_k, 
                top_p, 
                repetition_penalty
            ],
            outputs=[output]
        )
        
        # 示例
        gr.Examples(
            examples=[
                ["我想要一首关于爱情的歌"],
                ["夜空中最亮的星"],
                ["雨下整夜"],
            ],
            inputs=[prompt],
        )
    
    return app

def load_trained_model(model_dir):
    """加载已训练好的模型和分词器"""
    logger.info(f"正在加载已训练好的模型: {model_dir}")

    try:
        # 检查是否有GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        # 加载模型和分词器
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)

        # 确保pad_token设置正确
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        # 将模型移至适当的设备
        model.to(device)

        logger.info("模型和分词器加载成功")
        return model, tokenizer

    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        raise

# 主函数：执行完整流程
def main():
    # # 1. 处理CSV文件
    # logger.info("=== 步骤1: 处理CSV文件 ===")
    # lyrics_file = process_csv_lyrics(CSV_FILE, OUTPUT_DIR)
    
    # # 2. 训练模型
    # logger.info("=== 步骤2: 训练模型 ===")
    # model, tokenizer = train_model(
    #     data_path=lyrics_file,
    #     output_dir=MODEL_DIR,
    #     num_train_epochs=EPOCHS,
    #     per_device_train_batch_size=BATCH_SIZE
    # )
    
    # 加载已训练好的模型
    logger.info("=== 加载已训练好的模型 ===")
    model, tokenizer = load_trained_model(MODEL_DIR)
    # 3. 生成一些示例歌词
    logger.info("=== 步骤3: 生成示例歌词 ===")
    prompts = ["", "爱情", "思念", "夜晚", "雨"]
    for prompt in prompts:
        logger.info(f"使用提示 '{prompt}' 生成歌词...")
        generated_text = generate_lyrics(model, tokenizer, prompt=prompt)
        formatted_lyrics = format_lyrics(generated_text)
        
        # 打印生成的歌词
        logger.info(f"\n--- 提示: '{prompt}' ---\n")
        logger.info(formatted_lyrics)
        logger.info("\n" + "-" * 40)
    
    # 4. 启动Gradio界面
    logger.info("=== 步骤4: 启动Gradio界面 ===")
    app = create_ui(model, tokenizer)
    app.launch()

if __name__ == "__main__":
    main()