import gradio as gr
import os
import torch
from lyrics_generator import load_model, generate_lyrics, format_lyrics

# 全局变量，用于存储加载的模型和分词器
models = {}
model_paths = {}

def load_all_models(base_dir):
    """
    加载目录中的所有模型
    """
    global models, model_paths
    
    # 查找所有模型目录
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "pytorch_model.bin")):
            model_paths[item] = item_path
    
    return list(model_paths.keys())

def load_selected_model(model_name):
    """
    加载选定的模型
    """
    global models, model_paths
    
    if model_name not in models:
        try:
            model, tokenizer = load_model(model_paths[model_name])
            models[model_name] = (model, tokenizer)
            return f"模型 '{model_name}' 加载成功！"
        except Exception as e:
            return f"加载模型时出错: {str(e)}"
    else:
        return f"模型 '{model_name}' 已加载"

def generate_lyrics_ui(model_name, prompt, max_length, temperature, top_k, top_p, repetition_penalty, num_sequences):
    """
    生成歌词的UI函数
    """
    global models
    
    if model_name not in models:
        return f"错误：模型 '{model_name}' 未加载，请先加载模型"
    
    model, tokenizer = models[model_name]
    
    try:
        generated_texts = generate_lyrics(
            model, 
            tokenizer, 
            prompt=prompt,
            max_length=max_length,
            num_return_sequences=num_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        formatted_lyrics = [format_lyrics(text) for text in generated_texts]
        
        # 将生成的歌词合并为一个字符串，用分隔符分开
        result = ""
        for i, lyrics in enumerate(formatted_lyrics):
            result += f"\n--- 生成的歌词 {i+1} ---\n\n"
            result += lyrics
            result += "\n\n" + "-" * 40 + "\n"
        
        return result
    except Exception as e:
        return f"生成歌词时出错: {str(e)}"

def create_ui():
    """
    创建Gradio界面
    """
    # 加载所有可用模型
    model_dir = "trained_models"  # 模型目录
    available_models = load_all_models(model_dir)
    
    with gr.Blocks(title="歌词生成器") as app:
        gr.Markdown("# 歌词生成器")
        gr.Markdown("选择模型风格，输入提示，生成独特的歌词！")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    label="选择模型风格",
                    info="不同的模型代表不同的歌手或地区风格"
                )
                load_button = gr.Button("加载选定的模型")
                load_status = gr.Textbox(label="加载状态", interactive=False)
                
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
                    num_sequences = gr.Slider(
                        minimum=1, 
                        maximum=5, 
                        value=1, 
                        step=1, 
                        label="生成数量"
                    )
                
                generate_button = gr.Button("生成歌词", variant="primary")
            
            with gr.Column(scale=2):
                output = gr.Textbox(
                    lines=20, 
                    label="生成的歌词", 
                    interactive=False
                )
        
        # 事件处理
        load_button.click(
            fn=load_selected_model,
            inputs=[model_dropdown],
            outputs=[load_status]
        )
        
        generate_button.click(
            fn=generate_lyrics_ui,
            inputs=[
                model_dropdown, 
                prompt, 
                max_length, 
                temperature, 
                top_k, 
                top_p, 
                repetition_penalty, 
                num_sequences
            ],
            outputs=[output]
        )
        
        # 示例
        gr.Examples(
            examples=[
                ["默认模型", "我想要一首关于爱情的歌"],
                ["默认模型", "夜空中最亮的星"],
                ["默认模型", "雨下整夜"],
            ],
            inputs=[model_dropdown, prompt],
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch()