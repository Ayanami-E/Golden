# config.py
import torch

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 训练超参 =====
LR = 2e-4                 # AdamW 基础学习率
BATCH_SIZE = 32
NUM_WORKERS = 4
LOG_STEP = 50
NUM_EPOCHS = 3

# —— 动态时序正则（关键） ——
TAU_START = 0.10          # 训练初期权重大，稳住时间一致性
TAU_END   = 0.02          # 后期减小，避免总损失被 L_temp 主导
LOSS_TEMP_LOW_FREQ_K = 4  # 时间低通保留的前 K 个系数

# 数据目录
WEAK_PAIRS_DIR   = r"/root/autodl-tmp/VideoCrafter/Work/Work/outputs/all_weak_pairs/"
GOLDEN_PAIRS_DIR = r"/root/autodl-tmp/VideoCrafter/Work/outputs/all_golden_pairs/"
WEAK_SUPERVISION_RATIO = 0.8

# （仅做日志/初始化参考；模型可处理任意尺寸）
CHANNELS = 4
TEMPORAL_DIM = 16
HEIGHT = 32
WIDTH  = 32

# 频域支路
FREQ_DECAY = 1.2

# 文本编码器
TEXT_ENCODER_MODEL_DIM = 512  # CLIP-ViT-B/32 隐藏维
