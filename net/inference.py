# inference.py
"""
如何使用训练好的 NPNetV 模型进行推理
"""

import torch
from transformers import CLIPTextModel, CLIPTokenizer

# 确保 os, config, model 存在
import os
import config  # 需要 config.py 来获取模型维度
from model import NPNetV # 需要 model.py 来获取模型架构

# --- 1. 定义你的输入 ---
PROMPT = "a black dog wearing halloween costume" # (你想生成的提示词)
MODEL_PATH = "npnet_v_final.pth" # (你刚刚训练好的模型)
DEVICE = config.DEVICE


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 没找到模型文件 '{MODEL_PATH}'")
        return

    print("--- 1. 加载模型和 Tokenizer ---")
    
    # --- 加载 CLIP (用于编码文本) ---
    # 必须使用和训练时完全相同的编码器
    print(f"加载文本编码器: {config.TEXT_ENCODER_MODEL}")
    tokenizer = CLIPTokenizer.from_pretrained(config.TEXT_ENCODER_MODEL)
    text_encoder = CLIPTextModel.from_pretrained(
        config.TEXT_ENCODER_MODEL, 
        use_safetensors=True,
        dtype=torch.float32 # 保持和训练时一致
    ).to(DEVICE)
    text_encoder.eval()
    
    # --- 加载 NPNetV (我们的噪声优化器) ---
    print("加载 NPNetV 模型架构...")
    npnet = NPNetV(
        channels=config.CHANNELS,
        t=config.TEMPORAL_DIM,
        h=config.HEIGHT,
        w=config.WIDTH,
        freq_decay=config.FREQ_DECAY
    ).to(DEVICE)
    
    print(f"加载训练好的权重: {MODEL_PATH}")
    npnet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    npnet.eval() # 切换到评估模式 (非常重要)

    print("\n--- 2. 准备输入 ---")
    
    # --- 准备文本嵌入 E_txt ---
    with torch.no_grad():
        inputs = tokenizer(
            [PROMPT], # 放入一个列表中
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        # E_txt 形状 (B=1, D=768)
        E_txt = text_encoder(inputs.input_ids.to(DEVICE))[1] 
    
    print(f"提示词 '{PROMPT}' 已编码为 E_txt, 形状: {E_txt.shape}")

    # --- 准备初始高斯噪声 x_T ---
    # (B, C, T, H, W)
    x_T = torch.randn(
        1, # Batch size = 1
        config.CHANNELS,
        config.TEMPORAL_DIM,
        config.HEIGHT,
        config.WIDTH
    ).to(DEVICE)
    
    # (确保 x_T 也是 float32)
    x_T = x_T.float()
    
    print(f"已生成随机初始噪声 x_T, 形状: {x_T.shape}")

    # --- 3. 执行噪声优化 ---
    print("\n--- 3. 正在运行 NPNetV... ---")
    
    with torch.no_grad(): # 推理时不需要梯度
        x_star_T = npnet(x_T, E_txt)
        
    print("🎉 成功！已生成优化后的噪声 x_star_T 🎉")
    print(f"最终输出形状: {x_star_T.shape}")

    # --- 4. 后续步骤 ---
    print("\n--- 4. 如何使用 ---")
    print("你现在应该将 'x_star_T' (而不是 'x_T')")
    print("作为初始潜变量，输入到你的 T2V 扩散模型 (如 VideoCrafter2)")
    print("的采样循环 (e.g., DDIM) 中去。")

if __name__ == "__main__":
    main()