# utils.py
"""
辅助工具和模型
"""

import torch
import torch.nn as nn
import config

class MockFrozenUNet(nn.Module):
    """
    (MOCK) 这是一个模拟的冻结T2V UNet。
    您应该用您自己的模型 (例如 VideoCrafter2) 替换它。
    """
    def __init__(self, channels=config.CHANNELS):
        super().__init__()
        # 假装有一个 UNet，它接收 (B, C, T, H, W) 和 (B, D_text)
        # 并输出同形状的噪声预测
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
        # 确保它不被训练
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, latent, timestep, text_embed):
        # 这是一个假的 UNet 前向传播
        return self.conv(latent)
    
    def eval(self):
        # 确保模型始终处于评估模式
        super().eval()
        return self

def get_frozen_base_model():
    """
    (MOCK) 加载您冻结的基础模型。
    """
    print("加载 (模拟的) 冻结基础模型...")
    model = MockFrozenUNet().to(config.DEVICE)
    model.eval() # 确保它处于评估模式
    return model