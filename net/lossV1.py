# loss.py
"""
训练目标（鲁棒版）
- 主损失：Charbonnier（对离群值更稳）
- 时序正则：匹配预测与目标的低频“时间梯度”
- 支持外部传入 tau（用于余弦衰减）
- 全流程 float32
"""

import torch
import torch.nn as nn
import torch_dct


class CharbonnierLoss(nn.Module):
    """L(x,y) = mean( sqrt( (x-y)^2 + eps^2 ) )"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


class NPNetVLoss(nn.Module):
    def __init__(self, tau: float = 0.1, temporal_low_freq_k: int = 4, charbonnier_eps: float = 1e-6):
        super().__init__()
        self.register_buffer("tau_default", torch.tensor(float(tau)))
        self.k = int(temporal_low_freq_k)
        self.main_loss = CharbonnierLoss(eps=charbonnier_eps)

    def _temporal_low_pass_filter(self, x_f32: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,H,W) -> 对 T 做 DCT 低通
        x_perm = x_f32.permute(0, 1, 3, 4, 2)     # (B,C,H,W,T)
        x_freq = torch_dct.dct(x_perm, norm='ortho')  # 最后一维是 T
        mask = torch.zeros_like(x_freq)
        mask[..., :self.k] = 1.0
        x_low = torch_dct.idct(x_freq * mask, norm='ortho')
        x_low = x_low.permute(0, 1, 4, 2, 3).contiguous()
        return x_low

    def forward(self, x_star_T: torch.Tensor, x_target_T: torch.Tensor, tau_override: float = None):
        x_star_T   = x_star_T.float()
        x_target_T = x_target_T.float()

        # 1) 主损失（鲁棒）
        L_main = self.main_loss(x_star_T, x_target_T)

        # 2) 时间正则（低频时间梯度匹配）
        x_star_low   = self._temporal_low_pass_filter(x_star_T)
        x_target_low = self._temporal_low_pass_filter(x_target_T)
        d_pred  = x_star_low[:, :, 1:] - x_star_low[:, :, :-1]
        d_gt    = x_target_low[:, :, 1:] - x_target_low[:, :, :-1]
        L_temp  = torch.mean((d_pred - d_gt) ** 2)

        tau = self.tau_default.item() if tau_override is None else float(tau_override)
        total = L_main + tau * L_temp
        return total, L_main, L_temp
