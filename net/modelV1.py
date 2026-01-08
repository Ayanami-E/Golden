"""
NPNet-V (Proposal Version)
- 全 3D DCT(T,H,W)
- 软低通 mask(ft,fh,fw)
- 残差支路：3D CNN + FiLM + SE
- Temporal-Frequency 精细调制 Wr(ft;E_txt)
- 残差头 zero-init（保持稳定）
"""

import torch
import torch.nn as nn
import torch_dct
from typing import Tuple
import config


# -----------------------------
# Utilities
# -----------------------------
def _groups(c: int) -> int:
    if c >= 64:
        return 16
    if c >= 32:
        return 8
    return 4


def dct3(x):
    """
    x: (B, C, T, H, W)
    torch_dct 只能在最后一维做 DCT，所以我们多次 permute
    """
    # DCT over W
    x = torch_dct.dct(x, norm='ortho')  # last dim = W

    # DCT over H
    x = x.transpose(-2, -1).contiguous()
    x = torch_dct.dct(x, norm='ortho')
    x = x.transpose(-2, -1).contiguous()

    # DCT over T
    x = x.transpose(-3, -1).contiguous()  # move T to last
    x = torch_dct.dct(x, norm='ortho')
    x = x.transpose(-3, -1).contiguous()  # move back

    return x


def idct3(x):
    """
    x: (B, C, T, H, W)
    """
    # IDCT over W
    x = torch_dct.idct(x, norm='ortho')

    # IDCT over H
    x = x.transpose(-2, -1).contiguous()
    x = torch_dct.idct(x, norm='ortho')
    x = x.transpose(-2, -1).contiguous()

    # IDCT over T
    x = x.transpose(-3, -1).contiguous()
    x = torch_dct.idct(x, norm='ortho')
    x = x.transpose(-3, -1).contiguous()

    return x


# -----------------------------
# Squeeze-Excite
# -----------------------------
class SqueezeExcite3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


# -----------------------------
# FiLM
# -----------------------------
class FiLM(nn.Module):
    def __init__(self, text_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, 2 * channels)
    def forward(self, x, cond):
        B, C = x.shape[:2]
        gb = self.proj(cond)
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma.view(B, C, 1, 1, 1)
        beta  = beta.view(B, C, 1, 1, 1)
        return (1 + gamma) * x + beta


# -----------------------------
# Residual 3D Block
# -----------------------------
class ResBlock3D(nn.Module):
    def __init__(self, channels: int, text_dim: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        p = dilation
        g = _groups(channels)

        self.conv1 = nn.Conv3d(channels, channels, 3, padding=p, dilation=dilation)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=p, dilation=dilation)
        self.norm2 = nn.GroupNorm(g, channels)

        self.se = SqueezeExcite3D(channels)
        self.film = FiLM(text_dim, channels)

        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x, E_txt):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        h = self.se(h)
        h = self.film(h, E_txt)
        h = self.act(h)
        return x + h


# -----------------------------
# Residual Branch (CNN)
# -----------------------------
class ResidualBranch(nn.Module):
    def __init__(self, in_ch: int, text_dim: int, width: int = 64, depth: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU()
        )

        blocks = []
        for i in range(depth):
            dilation = 1 if i < depth // 2 else 2
            blocks.append(ResBlock3D(width, text_dim, dilation=dilation, dropout=0.05))
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, in_ch, 1)
        )

        nn.init.zeros_(self.head[-1].weight)
        if self.head[-1].bias is not None:
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, E_txt):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h, E_txt)
        return self.head(h)


# -----------------------------
# Proposal Core: Temporal Frequency Modulation
# -----------------------------
class TemporalFrequencyModulator(nn.Module):
    def __init__(self, text_dim: int, channels: int, T: int):
        super().__init__()
        self.W_freq = nn.Linear(text_dim, channels * T)

    def forward(self, r, E_txt):
        """
        r: (B,C,T,H,W)
        return: r_out (after Ft → modulation → inverse Ft)
        """
        B, C, T, H, W = r.shape

        # === 1D DCT along temporal dimension T ===
        # torch_dct 只能对最后一维做 DCT，所以把 T 挪到最后一维
        # (B,C,T,H,W) -> (B,C,H,W,T)
        r_perm = r.permute(0, 1, 3, 4, 2).contiguous()
        r_freq_perm = torch_dct.dct(r_perm, norm='ortho')  # DCT over T
        # 再换回 (B,C,T,H,W)
        r_freq = r_freq_perm.permute(0, 1, 4, 2, 3).contiguous()

        # Wr(ft;E_txt) → shape (B,C,T,1,1)，按时间频率调节
        w = self.W_freq(E_txt).view(B, C, T, 1, 1)
        w = torch.sigmoid(w)

        # modulation in frequency domain
        mod = r_freq * w  # (B,C,T,H,W)

        # === inverse 1D DCT along T ===
        mod_perm = mod.permute(0, 1, 3, 4, 2).contiguous()   # (B,C,H,W,T)
        r_out_perm = torch_dct.idct(mod_perm, norm='ortho')  # IDCT over T
        r_out = r_out_perm.permute(0, 1, 4, 2, 3).contiguous()  # (B,C,T,H,W)

        return r_out



# -----------------------------
# 3D Frequency Branch
# -----------------------------
class FrequencyBranch(nn.Module):
    def __init__(self, channels: int, decay: float = 1.0):
        super().__init__()
        self.channels = channels
        self.decay = decay

    def _mask(self, T, H, W, device):
        ft = torch.linspace(0, 1, T, device=device)
        fh = torch.linspace(0, 1, H, device=device)
        fw = torch.linspace(0, 1, W, device=device)

        Ft, Fh, Fw = torch.meshgrid(ft, fh, fw, indexing="ij")
        dist = Ft + Fh + Fw
        return torch.exp(-self.decay * dist).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, C, T, H, W = x.shape
        spec = dct3(x)

        mask = self._mask(T, H, W, x.device)
        spec_low = spec * mask

        out = idct3(spec_low)
        return out


# -----------------------------
# NPNet-V (Proposal Version)
# -----------------------------
class NPNetV(nn.Module):
    def __init__(self, channels: int, T: int, H: int, W: int, freq_decay: float = 1.0):
        super().__init__()
        text_dim = config.TEXT_ENCODER_MODEL_DIM

        self.freq_branch = FrequencyBranch(channels, decay=freq_decay)
        self.residual_branch = ResidualBranch(channels, text_dim, 64, 8)
        self.tfreq_mod = TemporalFrequencyModulator(text_dim, channels, T)

        self.alpha = nn.Parameter(torch.tensor(0.6))  # freq
        self.beta  = nn.Parameter(torch.tensor(0.8))  # residual

    def forward(self, x_T, E_txt):
        x_T = x_T.float()
        E_txt = E_txt.float()

        x_spec = self.freq_branch(x_T)
        r_base = self.residual_branch(x_T, E_txt)
        r_mod  = self.tfreq_mod(r_base, E_txt)

        return self.alpha * x_spec + self.beta * (x_T + r_mod)
