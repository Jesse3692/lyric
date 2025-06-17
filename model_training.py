import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    TextDataset, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(data_path, tokenizer, block_size=128):
    """
    加载数据集
    """
    logger.info(f"正在从 {data_path} 加载数据集...")
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=block_size,
    )
    return dataset

def train_model(
    model_name="gpt2",
    data_path="processed_lyrics/all_lyrics.txt",
    output_dir="trained_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    save_steps=10000,
    save_total_limit=2,
    logging_steps=500,
    block_size=128
):
    """
    训练歌词生成模型
    """
    # 检查是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载预训练模型和分词器
    if model_name == "gpt2":
        # 使用英文GPT-2模型
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        # 设置特殊token
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # 使用中文预训练模型，如BERT-base-chinese或其他
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    train_dataset = load_dataset(data_path, tokenizer, block_size)
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
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

def train_multiple_models(data_dir, base_output_dir, model_name="gpt2", **kwargs):
    """
    训练多个模型，每个对应不同的歌手或地区
    """
    # 遍历数据目录中的所有txt文件
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            category_name = file[:-4]  # 去掉.txt后缀
            data_path = os.path.join(data_dir, file)
            output_dir = os.path.join(base_output_dir, category_name)
            
            logger.info(f"开始训练 {category_name} 模型...")
            train_model(
                model_name=model_name,
                data_path=data_path,
                output_dir=output_dir,
                **kwargs
            )
            logger.info(f"{category_name} 模型训练完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练歌词生成模型")
    parser.add_argument("--model_name", type=str, default="gpt2", help="预训练模型名称")
    parser.add_argument("--data_path", type=str, required=True, help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="trained_model", help="模型输出目录")
    parser.add_argument("--batch_size", type=int, default=4, help="每个设备的训练批次大小")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--block_size", type=int, default=128, help="文本块大小")
    parser.add_argument("--train_multiple", action="store_true", help="是否训练多个模型")
    parser.add_argument("--data_dir", type=str, help="多个数据集的目录")
    
    args = parser.parse_args()
    
    if args.train_multiple:
        if not args.data_dir:
            parser.error("训练多个模型时需要指定 --data_dir")
        train_multiple_models(
            data_dir=args.data_dir,
            base_output_dir=args.output_dir,
            model_name=args.model_name,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            block_size=args.block_size
        )
    else:
        train_model(
            model_name=args.model_name,
            data_path=args.data_path,
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            block_size=args.block_size
        )