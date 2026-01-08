# make_golden_pairs.py (正确版本 + 断点续传)
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, torch, numpy as np
from typing import List
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from utils.utils import instantiate_from_config
from scripts.evaluation.funcs import load_model_checkpoint, load_prompts
from lvdm.models.samplers.ddim import DDIMSampler


def read_prompts(path: str) -> List[str]:
    try:
        return load_prompts(path)
    except:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines


@torch.no_grad()
def ddim_forward_K_steps(sampler, x_T, cond, uc, cfg_scale, K_steps):
    """
    正确的前向去噪 K 步
    使用全局索引，从 ddim_timesteps 的末尾开始
    """
    device = x_T.device
    batch_size = x_T.shape[0]
    
    # 使用全局的 DDIM schedule
    total_steps = len(sampler.ddim_timesteps)
    
    if K_steps > total_steps:
        raise ValueError(f"K_steps={K_steps} > total_ddim_steps={total_steps}")
    
    # ✅ 从最后 K 个时间步开始（最高噪声）
    start_idx = total_steps - K_steps  # 例如：50 - 10 = 40
    
    x_t = x_T
    
    print(f"      Forward: steps [{start_idx}, {total_steps-1}]")
    print(f"      Timesteps: t={sampler.ddim_timesteps[start_idx]} → t={sampler.ddim_timesteps[-1]}")
    
    # ✅ 使用全局索引遍历
    for global_idx in range(start_idx, total_steps):
        t_cur = sampler.ddim_timesteps[global_idx]
        t_tensor = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)
        
        # ✅ 直接使用全局索引的 alpha 值
        alpha_t = sampler.ddim_alphas[global_idx]
        alpha_prev = sampler.ddim_alphas_prev[global_idx]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = sampler.ddim_sqrt_one_minus_alphas[global_idx]
        
        # 预测噪声（强引导）
        if cfg_scale > 1.0 and uc is not None:
            noise_cond = sampler.model.apply_model(x_t, t_tensor, cond)
            noise_uncond = sampler.model.apply_model(x_t, t_tensor, uc)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = sampler.model.apply_model(x_t, t_tensor, cond)
        
        # 预测 x_0
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # DDIM 去噪
        sqrt_alpha_prev = torch.sqrt(torch.tensor(alpha_prev, device=x_t.device, dtype=x_t.dtype))
        sqrt_one_minus_alpha_prev = torch.sqrt(torch.tensor(1.0 - alpha_prev, device=x_t.device, dtype=x_t.dtype))
        
        x_t = sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * noise_pred
    
    return x_t, start_idx  # 返回 start_idx 用于反向


@torch.no_grad()
def ddim_inversion_K_steps(sampler, x_start, cond, uc, cfg_scale, start_idx, total_steps):
    """
    正确的反向 inversion
    使用与前向完全相同的全局索引范围，只是倒序遍历
    """
    device = x_start.device
    batch_size = x_start.shape[0]
    
    x_t = x_start
    
    print(f"      Inversion: steps [{total_steps-1}, {start_idx}] (reversed)")
    print(f"      Timesteps: t={sampler.ddim_timesteps[-1]} → t={sampler.ddim_timesteps[start_idx]}")
    
    # ✅ 倒序遍历相同的全局索引范围
    for global_idx in reversed(range(start_idx, total_steps)):
        t_cur = sampler.ddim_timesteps[global_idx]
        t_tensor = torch.full((batch_size,), t_cur, device=device, dtype=torch.long)
        
        # ✅ 使用全局索引的 alpha 值
        alpha_t = sampler.ddim_alphas[global_idx]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = sampler.ddim_sqrt_one_minus_alphas[global_idx]
        
        # ✅ prev 的处理：加噪方向
        if global_idx > start_idx:
            # 不是第一步，前一个是 global_idx - 1
            alpha_prev = sampler.ddim_alphas[global_idx - 1]
        else:
            # 第一步（回到 x_T），使用起始时刻的 alpha
            alpha_prev = sampler.model.alphas_cumprod[sampler.ddim_timesteps[start_idx]]
        
        # 预测噪声（弱引导）
        if cfg_scale > 1.0 and uc is not None:
            noise_cond = sampler.model.apply_model(x_t, t_tensor, cond)
            noise_uncond = sampler.model.apply_model(x_t, t_tensor, uc)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = sampler.model.apply_model(x_t, t_tensor, cond)
        
        # 预测 x_0
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # DDIM inversion（加噪）
        alpha_prev_t = torch.tensor(alpha_prev, device=x_t.device, dtype=x_t.dtype)
        sqrt_alpha_prev = torch.sqrt(alpha_prev_t)
        sqrt_one_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev_t)
        
        x_t = sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * noise_pred
    
    return x_t


@torch.no_grad()
def build_golden_pairs(
    model,
    prompts: List[str],
    outdir: str,
    K_steps: int = 10,
    total_ddim_steps: int = 50,
    cfg_forward: float = 7.5,
    cfg_backward: float = 1.0,
    batch_size: int = 1,  
    height: int = 320,
    width: int = 512,
    frames: int = -1,
    seed: int = 20230211,
):
    torch.manual_seed(seed)
    os.makedirs(outdir, exist_ok=True)

    assert (height % 8 == 0) and (width % 8 == 0)
    h, w = height // 8, width // 8
    T = model.temporal_length if frames < 0 else frames
    C = model.channels
    
    device = next(model.parameters()).device
    
    print(f"Latent shape: [{C}, {T}, {h}, {w}]")
    print(f"K DDIM steps: {K_steps} out of {total_ddim_steps}")
    print(f"CFG: forward={cfg_forward}, backward={cfg_backward}")

    ddim_sampler = DDIMSampler(model)
    
    # ✅ 初始化 DDIM schedule（只调用一次）
    ddim_sampler.make_schedule(ddim_num_steps=total_ddim_steps, ddim_eta=0.0, verbose=False)
    
    n = len(prompts) # --- [MODIFIED] --- 总数先在这里获取
    
    # --- [NEW] ---
    # 断点续传逻辑
    start_idx = 0
    print(f"Scanning {outdir} for existing pairs to resume...")
    if os.path.exists(outdir):
        existing_files = [f for f in os.listdir(outdir) if f.endswith('.pt')]
        existing_indices = []
        for f in existing_files:
            try:
                # '000001.pt' -> '000001' -> 1
                idx_str = f.split('.')[0]
                if idx_str.isdigit():
                    existing_indices.append(int(idx_str))
            except:
                pass # 忽略不匹配的文件

        if existing_indices:
            max_idx = max(existing_indices)
            start_idx = max_idx + 1 # 从下一个索引开始

    if start_idx == 0:
        print("  No existing pairs found. Starting from index 0.")
    else:
        if start_idx >= n:
             print(f"  ✓ Found {start_idx} pairs, which matches or exceeds total prompts ({n}). Nothing to do.")
             # 将 start_idx 设为 n，使后续 range(n, n) 为空，不执行循环
             start_idx = n
        else:
            print(f"  ✓ Found max index {start_idx - 1}. Resuming from index {start_idx}.")
    # --- [END NEW] ---
    
    print(f"\nGenerating remaining {max(0, n - start_idx)} of {n} Golden pairs...")
    
    # --- [MODIFIED] ---
    # 从计算出的 start_idx 开始循环
    for idx in range(start_idx, n, batch_size):
    # --- [END MODIFIED] ---
        batch_prompts = prompts[idx:min(idx+batch_size, n)]
        bs = len(batch_prompts)
        
        print(f"  [{idx+1}/{n}] {batch_prompts[0][:50]}...")

        noise_shape = [bs, C, T, h, w]
        x_T = torch.randn(noise_shape, device=device)

        text_emb = model.get_learned_conditioning(batch_prompts)
        cond = {"c_crossattn": [text_emb]}
        
        uc_emb = model.get_learned_conditioning([""] * bs)
        uc = {"c_crossattn": [uc_emb]}
        
        # Forward K steps
        print(f"    [1/2] Forward {K_steps} steps (strong CFG={cfg_forward:.1f})")
        x_TminusK, fwd_start_idx = ddim_forward_K_steps( # 变量名 fwd_start_idx 避免与 start_idx 冲突
            sampler=ddim_sampler,
            x_T=x_T,
            cond=cond,
            uc=uc,
            cfg_scale=cfg_forward,
            K_steps=K_steps,
        )
        
        # Backward K steps
        print(f"    [2/2] Backward {K_steps} steps (weak CFG={cfg_backward:.1f})")
        x_T_golden = ddim_inversion_K_steps(
            sampler=ddim_sampler,
            x_start=x_TminusK,
            cond=cond,
            uc=uc,
            cfg_scale=cfg_backward,
            start_idx=fwd_start_idx, # 使用前向步骤返回的索引
            total_steps=total_ddim_steps,
        )

        for i in range(bs):
            file_idx = idx + i
            diff_norm = torch.norm(x_T[i] - x_T_golden[i]).item()
            
            torch.save(
                {
                    "x_T": x_T[i].half().cpu(),
                    "x_T_target": x_T_golden[i].half().cpu(),
                    "prompt": batch_prompts[i],
                    "meta": {
                        "K_steps": K_steps,
                        "total_ddim_steps": total_ddim_steps,
                        "cfg_forward": cfg_forward,
                        "cfg_backward": cfg_backward,
                        "start_idx": fwd_start_idx, # 保存前向步骤的索引
                        "method": "golden_noise_corrected",
                        "diff_norm": diff_norm,
                    },
                },
                os.path.join(outdir, f"{file_idx:06d}.pt"),
            )
        
        print(f"    ✓ Saved (diff: {diff_norm:.2f})")
        torch.cuda.empty_cache()

    print(f"\n✓ Done!")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211)
    parser.add_argument("--config", type=str, default="/root/autodl-tmp/VideoCrafter/configs/inference_t2v_512_v2.0.yaml")
    parser.add_argument("--ckpt_path", type=str, default="/root/autodl-tmp/VideoCrafter/checkpoints/base_512_v2/model.ckpt")
    parser.add_argument("--prompt_file", type=str, default="Work/prompts/prompts_for_golden.txt")
    parser.add_argument("--outdir", type=str, default="Work/outputs/prompts_for_golden")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=-1)
    parser.add_argument("--K_steps", type=int, default=10)
    parser.add_argument("--total_ddim_steps", type=int, default=50)
    parser.add_argument("--cfg_forward", type=float, default=7.5)
    parser.add_argument("--cfg_backward", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    
    print("=" * 60)
    print("Golden Noise (Corrected Version + Resume)") # --- [MODIFIED] ---
    print("=" * 60)
    
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config).cuda()
    
    assert os.path.exists(args.ckpt_path)
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    
    assert os.path.exists(args.prompt_file)
    prompts = read_prompts(args.prompt_file)
    
    build_golden_pairs(
        model=model,
        prompts=prompts,
        outdir=args.outdir,
        K_steps=args.K_steps,
        total_ddim_steps=args.total_ddim_steps,
        cfg_forward=args.cfg_forward,
        cfg_backward=args.cfg_backward,
        batch_size=args.bs,
        height=args.height,
        width=args.width,
        frames=args.frames,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()