# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct


class NPNetVLoss(nn.Module):
    """
    NPNet-V2 对应的 loss:
    - 主损失: 只在时间维的低频成分上对齐 (Charbonnier)
    - 正则: 惩罚预测结果的时间高频能量，鼓励 temporal smoothness
    """

    def __init__(self, tau: float = 0.1, temporal_low_freq_k: int = 4, charbonnier_eps: float = 1e-6):
        super().__init__()
        # 基础 τ，用于没有 override 时的权重
        self.register_buffer("tau_base", torch.tensor(float(tau)))
        # 取前 K 个 temporal 频率作为低频
        self.temporal_low_freq_k = temporal_low_freq_k
        self.charbonnier_eps = charbonnier_eps

    # ---- 1D DCT / IDCT 沿时间维 ----
    def _temporal_dct(self, x):
        """
        x: (B, C, T, H, W)
        return: freq: (B, C, H, W, T)  —— DCT 是在最后一维上做的
        """
        x_perm = x.permute(0, 1, 3, 4, 2).contiguous()  # (B,C,H,W,T)
        x_freq = torch_dct.dct(x_perm, norm="ortho")
        return x_freq

    def _temporal_idct(self, x_freq):
        """
        x_freq: (B, C, H, W, T)
        return: x: (B, C, T, H, W)
        """
        x_perm = torch_dct.idct(x_freq, norm="ortho")  # (B,C,H,W,T)
        x = x_perm.permute(0, 1, 4, 2, 3).contiguous()
        return x

    def forward(self, x_pred, x_target, tau_override=None):
        """
        x_pred   : NPNet 输出的噪声  x*_T  (B,C,T,H,W)
        x_target : 监督的目标噪声    x_T_target (B,C,T,H,W)
        tau_override : 训练脚本里用 cosine_tau 动态传进来的 τ

        return:
            total_loss, L_main, L_temp
        """
        B, C, T, H, W = x_pred.shape
        K = min(self.temporal_low_freq_k, T)

        # === 1. 时间维 DCT，得到频域表示 ===
        pred_freq = self._temporal_dct(x_pred)   # (B,C,H,W,T)
        tgt_freq  = self._temporal_dct(x_target)

        device = x_pred.device

        # === 2. 构造低频 / 高频 mask ===
        mask_t = torch.zeros(T, device=device)
        mask_t[:K] = 1.0                         # 前 K 个是低频
        mask_low = mask_t.view(1, 1, 1, 1, T)    # (1,1,1,1,T)
        mask_high = 1.0 - mask_low

        # === 3. 低频分量用于主重建损失 ===
        pred_freq_low = pred_freq * mask_low
        tgt_freq_low  = tgt_freq * mask_low

        x_pred_low = self._temporal_idct(pred_freq_low)  # (B,C,T,H,W)
        x_tgt_low  = self._temporal_idct(tgt_freq_low)

        diff_low = x_pred_low - x_tgt_low

        # Charbonnier loss （带 sqrt 的 L1）
        L_main = torch.sqrt(diff_low * diff_low + self.charbonnier_eps).mean()

        # === 4. 高频能量正则（鼓励 temporal smoothness） ===
        high_freq_pred = pred_freq * mask_high
        L_temp = torch.sqrt(high_freq_pred * high_freq_pred + self.charbonnier_eps).mean()

        # === 5. τ 权重 ===
        if tau_override is None:
            tau = float(self.tau_base.item())
        else:
            tau = float(tau_override)

        loss = L_main + tau * L_temp

        return loss, L_main, L_temp
