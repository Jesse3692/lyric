import pandas as pd
import os
import re
import json
from tqdm import tqdm

def clean_lyrics(text):
    """
    清洗歌词文本，只保留中英文和标点符号
    """
    if pd.isna(text):
        return ""
    
    # 保留中文、英文、数字、常见标点
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、：；''""（）【】《》\s]'
    cleaned_text = re.sub(pattern, '', str(text))
    # 去除多余空白
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def process_csv_lyrics(csv_file, output_dir, singer_column=None, region_column=None):
    """
    处理CSV格式的歌词数据
    
    Args:
        csv_file: CSV文件路径
        output_dir: 输出目录
        singer_column: 歌手列名（可选）
        region_column: 地区列名（可选）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_file}")
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码
        try:
            df = pd.read_csv(csv_file, encoding='gbk')
        except:
            df = pd.read_csv(csv_file, encoding='latin1')
    
    print(f"CSV文件包含 {len(df)} 行数据")
    
    # 查看CSV文件的列
    print("CSV文件的列名:", df.columns.tolist())
    
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
    
    print(f"使用 '{lyrics_column}' 列作为歌词")
    
    # 清洗歌词
    print("正在清洗歌词...")
    df['cleaned_lyrics'] = df[lyrics_column].apply(clean_lyrics)
    
    # 保存所有清洗后的歌词到一个文件
    all_lyrics_file = os.path.join(output_dir, "all_lyrics.txt")
    with open(all_lyrics_file, 'w', encoding='utf-8') as f:
        for lyrics in tqdm(df['cleaned_lyrics'], desc="保存所有歌词"):
            if lyrics and len(lyrics) > 10:  # 只保存非空且长度合理的歌词
                f.write(lyrics + "\n\n")
    
    print(f"所有歌词已保存到: {all_lyrics_file}")
    
    # 如果指定了歌手列，按歌手分类
    if singer_column and singer_column in df.columns:
        singer_dir = os.path.join(output_dir, "by_singer")
        os.makedirs(singer_dir, exist_ok=True)
        
        singers = df[singer_column].dropna().unique()
        print(f"按歌手分类，共有 {len(singers)} 个歌手")
        
        for singer in tqdm(singers, desc="按歌手处理"):
            singer_lyrics = df[df[singer_column] == singer]['cleaned_lyrics']
            if len(singer_lyrics) < 5:  # 跳过歌词太少的歌手
                continue
                
            singer_file = os.path.join(singer_dir, f"{singer}.txt")
            with open(singer_file, 'w', encoding='utf-8') as f:
                for lyrics in singer_lyrics:
                    if lyrics and len(lyrics) > 10:
                        f.write(lyrics + "\n\n")
    
    # 如果指定了地区列，按地区分类
    if region_column and region_column in df.columns:
        region_dir = os.path.join(output_dir, "by_region")
        os.makedirs(region_dir, exist_ok=True)
        
        regions = df[region_column].dropna().unique()
        print(f"按地区分类，共有 {len(regions)} 个地区")
        
        for region in tqdm(regions, desc="按地区处理"):
            region_lyrics = df[df[region_column] == region]['cleaned_lyrics']
            if len(region_lyrics) < 5:  # 跳过歌词太少的地区
                continue
                
            region_file = os.path.join(region_dir, f"{region}.txt")
            with open(region_file, 'w', encoding='utf-8') as f:
                for lyrics in region_lyrics:
                    if lyrics and len(lyrics) > 10:
                        f.write(lyrics + "\n\n")
    
    return all_lyrics_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="处理CSV格式的歌词数据")
    parser.add_argument("--csv_file", type=str, required=True, help="CSV文件路径")
    parser.add_argument("--output_dir", type=str, default="processed_lyrics", help="输出目录")
    parser.add_argument("--singer_column", type=str, help="歌手列名（可选）")
    parser.add_argument("--region_column", type=str, help="地区列名（可选）")
    
    args = parser.parse_args()
    
    process_csv_lyrics(
        args.csv_file, 
        args.output_dir, 
        args.singer_column, 
        args.region_column
    )