# inference_net.py
"""
使用训练好的 NPNetV 模型进行推理，并与原始噪声进行对比。

特点：
- 与 train 相同的 CLIP 文本编码方式（attention mask + mean pooling）
- 与 train 相同的本地 CLIP 路径（可通过 --clip_path 覆盖）
- 冻结并 eval() 文本编码器
- NPNetV 给出 x_star = NPNet(x_T, E_txt)
- 按全局强度系数 npnet_strength 做混合：
    x_T_in = (1 - s) * x_T + s * x_star
  s=0  等价于不用 NPNet
  s=1  完全使用 NPNet 输出
"""

import os
import sys
import argparse
import glob
import time

from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from pytorch_lightning import seed_everything

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

sys.path.append(script_dir)      # net/ 目录，导入 config, model
sys.path.append(project_root)    # 项目根目录，导入 scripts, utils

# NPNetV 相关
from transformers import CLIPTextModel, CLIPTokenizer
import config as npnet_config
from model import NPNetV

# T2V 模型相关
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling
from utils.utils import instantiate_from_config


def load_pair(pt_path):
    """Load .pt file (weak or golden pairs) into CPU tensors + metadata dict."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    return data


def get_parser():
    parser = argparse.ArgumentParser(description="Generate videos using NPNetV (no lambda-search)")

    parser.add_argument("--seed", type=int, default=20251116)

    # T2V base model
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt")

    # NPNetV 模型
    parser.add_argument(
        "--npnet_ckpt_path",
        type=str,
        default="npnet_v_final.pth",
        help="Path to the trained NPNetV checkpoint."
    )

    # NPNetV 强度：0 = 完全不用 NPNetV, 1 = 完全使用 NPNetV 输出
    parser.add_argument(
        "--npnet_strength",
        type=float,
        default=1.0,
        help="How strongly NPNetV modifies the initial noise: x_in = (1-s)*x_T + s*x_star.",
    )

    # 与训练一致的本地 CLIP 路径
    parser.add_argument(
        "--clip_path",
        type=str,
        default="/root/autodl-tmp/VideoCrafter/local_clip_model",
        help="Local path to CLIP tokenizer & text encoder used for NPNetV training."
    )

    # 噪声对目录 & 输出目录
    parser.add_argument(
        "--pairs_dir",
        type=str,
        default="/root/autodl-tmp/VideoCrafter/Work/outputs/weak_pairs/",
        help="Directory containing .pt files (weak_pairs or golden_pairs)."
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="Work/outputs/npnet_videos",
        help="Directory to save the comparison videos."
    )

    parser.add_argument("--savefps", type=int, default=10)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5)

    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--runtime_bs", type=int, default=1, help="Batch size for inference")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)

    print("=" * 60)
    print("Generate Videos using NPNetV vs. Baseline (no lambda-search)")
    print("=" * 60)

    # ===== 1. Load T2V model =====
    print("\n[1/4] Loading T2V (base) model...")
    t2v_config = OmegaConf.load(args.config)
    model_config = t2v_config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config).cuda()

    assert os.path.exists(args.ckpt_path), f"T2V ckpt not found: {args.ckpt_path}"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    device = next(model.parameters()).device
    print(f" <i>✓ T2V Model loaded on {device}")

    # ===== 2. Load NPNetV + CLIP =====
    print("\n[2/4] Loading NPNetV & text encoder...")

    local_clip_path = args.clip_path
    assert os.path.exists(local_clip_path), f"CLIP path not found: {local_clip_path}"
    print(f" <i>Loading text encoder from local path: {local_clip_path}")

    npnet_tokenizer = CLIPTokenizer.from_pretrained(local_clip_path)
    npnet_text_encoder = CLIPTextModel.from_pretrained(
        local_clip_path,
        torch_dtype=torch.float32
    ).to(device)
    npnet_text_encoder.eval()
    for p in npnet_text_encoder.parameters():
        p.requires_grad = False

    assert os.path.exists(args.npnet_ckpt_path), f"NPNetV ckpt not found: {args.npnet_ckpt_path}"

    npnet = NPNetV(
        channels=npnet_config.CHANNELS,
        T=npnet_config.TEMPORAL_DIM,
        H=npnet_config.HEIGHT,
        W=npnet_config.WIDTH,
        freq_decay=npnet_config.FREQ_DECAY
    ).to(device)

    state = torch.load(args.npnet_ckpt_path, map_location=device)
    npnet.load_state_dict(state, strict=True)
    npnet.eval()

    print(f" <i>✓ NPNetV Model loaded from {args.npnet_ckpt_path}")

    # clamp npnet_strength
    s = float(args.npnet_strength)
    s = max(0.0, min(1.0, s))
    print(f"\n[Info] npnet_strength = {s} (0=no effect, 1=full NPNet)")

    # ===== 3. Load the pairs list =====
    print(f"\n[3/4] Loading pairs...")
    pt_files = sorted(glob.glob(os.path.join(args.pairs_dir, "*.pt")))
    total_files = len(pt_files)
    print(f" <i>✓ Found {total_files} pairs in {args.pairs_dir}")

    # Output directories
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "baseline_xT"), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "npnet_x_star_T"), exist_ok=True)

    # ===== 4. Batched generation =====
    print("\n[4/4] Generating videos in batches of", args.runtime_bs)
    start_time = time.time()
    runtime_bs = args.runtime_bs

    pbar_all = tqdm(total=total_files, desc="Total progress", position=0)

    with torch.no_grad():
        for start_idx in range(0, total_files, runtime_bs):
            batch_paths = pt_files[start_idx:start_idx + runtime_bs]
            cur_bs = len(batch_paths)
            batch_desc = f"[Batch {start_idx}-{start_idx + cur_bs - 1}]"
            pbar_batch = tqdm(total=4, desc=batch_desc, position=1, leave=False)

            # -------------------------------------------------
            # Step 0. 读这一批的数据
            xT_list = []
            prompts_batch = []
            metas_batch = []

            for p in batch_paths:
                data = load_pair(p)
                x_T = data["x_T"].float().unsqueeze(0)   # (1, C, T, H, W)
                xT_list.append(x_T)
                prompts_batch.append(data["prompt"])
                metas_batch.append(data.get("meta", "N/A"))

            x_T_batch = torch.cat(xT_list, dim=0).to(device)   # (B, C, T, H, W)
            B, C, T, H, W = x_T_batch.shape
            noise_shape = [B, C, T, H, W]

            # -------------------------------------------------
            # Step 1. NPNet 前向，得到 x_star_raw
            pbar_batch.update(1)
            pbar_batch.set_description(f"{batch_desc} running NPNet")

            inputs = npnet_tokenizer(
                prompts_batch,
                padding="max_length",
                truncation=True,
                max_length=npnet_tokenizer.model_max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = npnet_text_encoder(input_ids=inputs["input_ids"])
            hidden = out.last_hidden_state.float()                     # (B, L, D)
            mask = inputs["attention_mask"].unsqueeze(-1).float()      # (B, L, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            E_txt = pooled.float()                                     # (B, D)

            x_star_raw = npnet(x_T_batch, E_txt)                       # (B, C, T, H, W)

            if s > 0.0:
                x_in_batch = (1.0 - s) * x_T_batch + s * x_star_raw
            else:
                x_in_batch = x_T_batch.clone()

            # -------------------------------------------------
            # Step 2. 准备 T2V 的条件
            text_emb = model.get_learned_conditioning(prompts_batch).to(device)
            cond = {"c_crossattn": [text_emb]}

            pbar_batch.update(1)
            pbar_batch.set_description(f"{batch_desc} sampling baseline")

            # -------------------------------------------------
            # Step 3. Baseline: 原始 x_T
            samples_xT = batch_ddim_sampling(
                model, cond, noise_shape,
                n_samples=args.n_samples,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                cfg_scale=args.unconditional_guidance_scale,
                x_T=x_T_batch,
            )

            pbar_batch.update(1)
            pbar_batch.set_description(f"{batch_desc} sampling NPNet")

            # -------------------------------------------------
            # Step 4. Ours: NPNet refined x_T
            samples_x_star_T = batch_ddim_sampling(
                model, cond, noise_shape,
                n_samples=args.n_samples,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                cfg_scale=args.unconditional_guidance_scale,
                x_T=x_in_batch,
            )

            pbar_batch.update(1)
            pbar_batch.set_description(f"{batch_desc} saving")

            # -------------------------------------------------
            # Step 5. 保存结果
            filenames = [f"{start_idx + j:06d}" for j in range(cur_bs)]

            save_videos(
                samples_xT,
                os.path.join(args.savedir, "baseline_xT"),
                filenames,
                fps=args.savefps
            )
            save_videos(
                samples_x_star_T,
                os.path.join(args.savedir, "npnet_x_star_T"),
                filenames,
                fps=args.savefps
            )

            for fname, prompt, meta in zip(filenames, prompts_batch, metas_batch):
                with open(os.path.join(args.savedir, f"{fname}.txt"), "w") as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Metadata: {meta}\n")

            pbar_batch.close()
            pbar_all.update(cur_bs)

    pbar_all.close()

    elapsed = time.time() - start_time
    print(f"\n✓ Generated {total_files * 2} videos in {elapsed:.2f}s")
    print(f" <i>Saved to: {args.savedir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
