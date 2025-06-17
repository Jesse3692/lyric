import re
import os
import json
from tqdm import tqdm

def clean_lyrics(text):
    """
    清洗歌词文本，只保留中英文和标点符号
    """
    # 保留中文、英文、数字、常见标点
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、：；''""（）【】《》\s]'
    cleaned_text = re.sub(pattern, '', text)
    # 去除多余空白
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def process_lyrics_file(file_path, output_dir, metadata=None):
    """
    处理单个歌词文件并保存清洗后的结果
    
    Args:
        file_path: 歌词文件路径
        output_dir: 输出目录
        metadata: 元数据字典，包含歌手、地区等信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 清洗歌词
        cleaned_content = clean_lyrics(content)
        
        # 创建输出文件名
        base_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, f"cleaned_{base_name}")
        
        # 保存清洗后的歌词
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        # 如果有元数据，保存元数据
        if metadata:
            metadata_file = os.path.join(output_dir, f"meta_{base_name}.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return output_file
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def process_lyrics_dataset(lyrics_dir, output_dir, metadata_file=None):
    """
    处理整个歌词数据集
    
    Args:
        lyrics_dir: 歌词文件目录
        output_dir: 输出目录
        metadata_file: 元数据文件路径，包含歌手、地区等信息
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载元数据（如果有）
    metadata_dict = {}
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
    
    # 获取所有歌词文件
    lyrics_files = []
    for root, _, files in os.walk(lyrics_dir):
        for file in files:
            if file.endswith('.txt'):
                lyrics_files.append(os.path.join(root, file))
    
    # 处理所有歌词文件
    processed_files = []
    for file_path in tqdm(lyrics_files, desc="处理歌词文件"):
        file_name = os.path.basename(file_path)
        file_metadata = metadata_dict.get(file_name, {})
        processed_file = process_lyrics_file(file_path, output_dir, file_metadata)
        if processed_file:
            processed_files.append(processed_file)
    
    print(f"成功处理 {len(processed_files)}/{len(lyrics_files)} 个歌词文件")
    return processed_files

def categorize_lyrics_by_metadata(output_dir, category_field='singer'):
    """
    根据元数据对歌词进行分类，例如按歌手、地区等
    
    Args:
        output_dir: 输出目录，包含清洗后的歌词和元数据
        category_field: 分类字段，如'singer', 'region'等
    """
    categories = {}
    
    # 遍历所有元数据文件
    for file in os.listdir(output_dir):
        if file.startswith('meta_') and file.endswith('.json'):
            with open(os.path.join(output_dir, file), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 获取分类值
            category_value = metadata.get(category_field, 'unknown')
            
            # 对应的歌词文件
            lyrics_file = os.path.join(output_dir, f"cleaned_{file[5:-5]}")
            if os.path.exists(lyrics_file):
                if category_value not in categories:
                    categories[category_value] = []
                categories[category_value].append(lyrics_file)
    
    # 为每个分类创建一个合并文件
    for category, files in categories.items():
        category_dir = os.path.join(output_dir, category_field)
        os.makedirs(category_dir, exist_ok=True)
        
        merged_content = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                merged_content.append(f.read())
        
        # 保存合并后的文件
        with open(os.path.join(category_dir, f"{category}.txt"), 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(merged_content))
    
    return categories

if __name__ == "__main__":
    # 示例用法
    lyrics_dir = "lyrics_data"  # 原始歌词目录
    output_dir = "processed_lyrics"  # 处理后的输出目录
    metadata_file = "lyrics_metadata.json"  # 元数据文件（可选）
    
    # 处理歌词数据集
    processed_files = process_lyrics_dataset(lyrics_dir, output_dir, metadata_file)
    
    # 按歌手分类
    singer_categories = categorize_lyrics_by_metadata(output_dir, 'singer')
    print(f"按歌手分类完成，共有 {len(singer_categories)} 个歌手类别")
    
    # 按地区分类
    region_categories = categorize_lyrics_by_metadata(output_dir, 'region')
    print(f"按地区分类完成，共有 {len(region_categories)} 个地区类别")