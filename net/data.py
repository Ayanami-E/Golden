# data.py
"""
从磁盘加载预计算的噪声对 (Section 3.3)
(已修复：使用 last_hidden_state 的 mean pooling)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import config

class PrecomputedNoiseDataset(Dataset):
    """
    一个用于加载预计算噪声对的自定义数据集。
    (已更新，用于加载 'prompt' 字符串)
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.file_paths = sorted(list(self.directory.glob("*.pt"))) 

        if not self.file_paths:
            raise FileNotFoundError(
                f"在 {directory} 中没有找到 .pt 文件。"
                "请检查您的 config.py 路径以及文件扩展名是否正确。"
            )
        print(f"在 {directory} 中找到了 {len(self.file_paths)} 个预计算样本。")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        try:
            # weights_only=False 允许加载包含字典和张量的 pickle 文件
            data = torch.load(file_path, map_location='cpu', weights_only=False) 

            # --- (重要) 键已更新！ ---
            x_T = data['x_T']
            x_T_target = data['x_T_target']
            prompt_text = data['prompt'] # 这是一个字符串

            # 返回原始文本，我们将在训练循环中批量编码
            return x_T, x_T_target, prompt_text

        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            print("请确保文件格式正确，并且包含 'x_T', 'x_T_target', 'prompt' 键。")
            return None # 返回 None，让 dataloader 忽略此样本

def safe_collate(batch):
    """一个安全的 collate_fn，用于过滤掉加载失败的 (None) 样本"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None, None # 返回 None 以便在训练循环中跳过
    return torch.utils.data.dataloader.default_collate(batch)

def infinite_iterator(dataloader):
    """一个将 dataloader 转换为无限循环迭代器的辅助函数"""
    while True:
        for batch in dataloader:
            yield batch

def create_data_iterators():
    """
    (已更新) 创建 weak 和 golden 数据的 Dataset, DataLoader, 和无限迭代器。
    使用 safe_collate 来跳过损坏的文件。
    """
    weak_dataset = PrecomputedNoiseDataset(config.WEAK_PAIRS_DIR)
    golden_dataset = PrecomputedNoiseDataset(config.GOLDEN_PAIRS_DIR)

    weak_loader = DataLoader(
        weak_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True, 
        drop_last=True,
        collate_fn=safe_collate  # <-- (新增)
    )

    golden_loader = DataLoader(
        golden_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate  # <-- (新增)
    )

    weak_iter = infinite_iterator(weak_loader)
    golden_iter = infinite_iterator(golden_loader)

    print("Weak 和 Golden 数据迭代器创建成功。")
    return weak_iter, golden_iter

def get_next_batch(weak_iter, golden_iter, tokenizer, text_encoder):
    """
    (已修复) 从迭代器获取一个批次，并即时编码文本。
    使用 last_hidden_state 的 mean pooling 得到 768 维嵌入。
    """
    if torch.rand(1).item() < config.WEAK_SUPERVISION_RATIO:
        batch = next(weak_iter)
    else:
        batch = next(golden_iter)

    x_T, x_T_target, prompt_texts = batch

    # 如果批次为空 (由于 safe_collate 过滤掉了所有样本)
    if x_T is None:
        return None, None, None

    # --- (修复) 即时编码文本 ---
    with torch.no_grad():
        # 1. 标记化: (B, Seq_Len)
        inputs = tokenizer(
            prompt_texts, 
            padding="max_length", # 填充到最大长度
            truncation=True,      # 截断
            max_length=tokenizer.model_max_length, # (例如 77 for CLIP)
            return_tensors="pt"
        )
        
        # 2. 编码: (B, Seq_Len, Dim)
        text_encoder_output = text_encoder(inputs.input_ids.to(config.DEVICE))
        
        # CLIP 输出:
        # - last_hidden_state: (B, Seq_Len, 512) for clip-vit-base-patch32
        # - pooler_output: (B, 512) - 这是 [CLS] token 的输出
        
        # 问题：pooler_output 是 512 维，但 config 期望 768 维
        # 解决方案：使用 last_hidden_state 的 mean pooling
        last_hidden_state = text_encoder_output.last_hidden_state  # (B, Seq_Len, Hidden_Dim)
        
        # Mean pooling over sequence length
        # 使用 attention_mask 来避免对 padding tokens 求平均
        attention_mask = inputs.attention_mask.to(config.DEVICE)  # (B, Seq_Len)
        
        # Expand mask to match hidden state dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum over sequence length with mask
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Mean pooling
        E_txt = sum_embeddings / sum_mask  # (B, Hidden_Dim)

    # 将数据移动到 GPU
    return x_T.to(config.DEVICE), E_txt.to(config.DEVICE), x_T_target.to(config.DEVICE)