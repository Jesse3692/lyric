import argparse
import os
import logging
from data_preprocessing import process_lyrics_dataset, categorize_lyrics_by_metadata
from model_training import train_model, train_multiple_models
from lyrics_generator import generate_lyrics_with_style

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="歌词生成模型训练和使用")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 数据预处理命令
    preprocess_parser = subparsers.add_parser("preprocess", help="预处理歌词数据")
    preprocess_parser.add_argument("--lyrics_dir", type=str, help="原始歌词目录")
    preprocess_parser.add_argument("--output_dir", type=str, required=True, help="处理后的输出目录")
    preprocess_parser.add_argument("--metadata_file", type=str, help="元数据文件（可选）")
    preprocess_parser.add_argument("--categorize", action="store_true", help="是否按元数据分类")
    preprocess_parser.add_argument("--category_field", type=str, default="singer", help="分类字段")
    preprocess_parser.add_argument("--csv_file", type=str, help="CSV格式歌词文件（可选）")
    preprocess_parser.add_argument("--singer_column", type=str, help="CSV中的歌手列名（可选）")
    preprocess_parser.add_argument("--region_column", type=str, help="CSV中的地区列名（可选）")
    
    # 模型训练命令
    train_parser = subparsers.add_parser("train", help="训练歌词生成模型")
    train_parser.add_argument("--model_name", type=str, default="gpt2", help="预训练模型名称")
    train_parser.add_argument("--data_path", type=str, help="训练数据路径")
    train_parser.add_argument("--output_dir", type=str, default="trained_model", help="模型输出目录")
    train_parser.add_argument("--batch_size", type=int, default=4, help="每个设备的训练批次大小")
    train_parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    train_parser.add_argument("--train_multiple", action="store_true", help="是否训练多个模型")
    train_parser.add_argument("--data_dir", type=str, help="多个数据集的目录")
    
    # 生成歌词命令
    generate_parser = subparsers.add_parser("generate", help="生成歌词")
    generate_parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    generate_parser.add_argument("--prompt", type=str, default="", help="提示文本")
    generate_parser.add_argument("--max_length", type=int, default=200, help="生成文本的最大长度")
    generate_parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    generate_parser.add_argument("--output_file", type=str, help="输出文件路径")
    
    # UI命令
    ui_parser = subparsers.add_parser("ui", help="启动Gradio界面")
    ui_parser.add_argument("--model_dir", type=str, default="trained_models", help="模型目录")
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        if args.csv_file:
            # 处理CSV格式的歌词数据
            from csv_lyrics_processor import process_csv_lyrics
            process_csv_lyrics(
                args.csv_file, 
            args.output_dir, 
                args.singer_column, 
                args.region_column
        )
        elif args.lyrics_dir:
            # 处理歌词数据集
            processed_files = process_lyrics_dataset(
                args.lyrics_dir, 
                args.output_dir, 
                args.metadata_file
            )
            logger.info(f"成功处理 {len(processed_files)} 个歌词文件")
            
            # 按元数据分类（如果需要）
            if args.categorize:
                categories = categorize_lyrics_by_metadata(
                    args.output_dir, 
                    args.category_field
            )
                logger.info(f"按 {args.category_field} 分类完成，共有 {len(categories)} 个类别")
        else:
            logger.error("请指定 --lyrics_dir 或 --csv_file")
    
    elif args.command == "train":
        if args.train_multiple:
            if not args.data_dir:
                parser.error("训练多个模型时需要指定 --data_dir")
            train_multiple_models(
                data_dir=args.data_dir,
                base_output_dir=args.output_dir,
                model_name=args.model_name,
                per_device_train_batch_size=args.batch_size,
                num_train_epochs=args.epochs
            )
        else:
            if not args.data_path:
                parser.error("训练单个模型时需要指定 --data_path")
            train_model(
                model_name=args.model_name,
                data_path=args.data_path,
                output_dir=args.output_dir,
                per_device_train_batch_size=args.batch_size,
                num_train_epochs=args.epochs
            )
    
    elif args.command == "generate":
        # 导入生成模块
        from lyrics_generator import load_model, generate_lyrics, format_lyrics
        
        # 加载模型
        model, tokenizer = load_model(args.model_path)
        
        # 生成歌词
        generated_lyrics = generate_lyrics(
            model, 
            tokenizer, 
            prompt=args.prompt,
            max_length=args.max_length,
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
            logger.info(f"歌词已保存到 {args.output_file}")
    
    elif args.command == "ui":
        # 导入UI模块
        from app import create_ui
        
        # 创建并启动UI
        app = create_ui()
        app.launch()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
