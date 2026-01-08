import os
import torch
import glob
import math

# ====== 你需要提供的 ======
from model import dct3, idct3   # <-- 你自己的 3D DCT 实现
# ==========================

def extract_direction(x_T, x_Tg, low_ratio=0.25, top_k=1, clamp_r=3.0, alpha=0.3):
    """
    输入:
        x_T   : 原噪声  [C,T,H,W]
        x_Tg  : golden噪声 [C,T,H,W]

    输出:
        d     : semantic direction  [C,T,H,W]
    """

    # 1. 残差
    D = x_Tg - x_T   # [C,T,H,W]

    # 2. DCT
    D_dct = dct3(D)  # [C,T,H,W]

    C, T, H, W = D_dct.shape
    hcut = int(H * low_ratio)
    wcut = int(W * low_ratio)

    # 3. 低频截断
    D_low = D_dct[:, :, :hcut, :wcut]   # [C,T,hcut,wcut]
    M = D_low.reshape(C*T, hcut*wcut)   # reshape 成矩阵

    # 6. 重建完整频域 (其它高频=0)
    D_dct_new = torch.zeros_like(D_dct)
    D_dct_new[:, :, :hcut, :wcut] = D_low_k

    # 7. 逆DCT → 得到 direction
    d = idct3(D_dct_new)

    # 8. 限幅 + 缩放
    d = d.clamp(-clamp_r, clamp_r)
    d = alpha * d

    return d


def main():
    indir  = "data/all_golden_pairs"
    outdir = "data/phase2_dirs"
    os.makedirs(outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(indir, "*.pt")))
    print(f"Found {len(files)} golden samples.")

    for f in files:
        basename = os.path.basename(f)
        idx = basename.split(".")[0]

        outpath = os.path.join(outdir, f"{idx}.pt")
        if os.path.exists(outpath):
            print(f"[skip] {outpath}")
            continue

        ckpt = torch.load(f, map_location="cpu")

        x_T  = ckpt["x_T"].float()
        x_Tg = ckpt["x_T_target"].float()
        prompt = ckpt["prompt"]

        d = extract_direction(
            x_T, x_Tg,
            low_ratio=0.25,
            top_k=1,
            clamp_r=3.0,
            alpha=0.3
        )

        torch.save(
            {
                "prompt": prompt,
                "direction": d.half(),
                "meta_old": ckpt["meta"]
            },
            outpath
        )

        print(f"[ok] saved {outpath}")


if __name__ == "__main__":
    main()
