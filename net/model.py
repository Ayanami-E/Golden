import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless draw
import matplotlib.pyplot as plt

# Import from the updated metrics module
from metrics import (
    compute_clipscore,
    compute_pickscore,
    compute_temporal_consistency,
    compute_flicker,
    compute_lpips,
    compute_psnr_ssim,
    compute_motion_variance,
)

############################################################
# Video loading
############################################################

def load_video_as_tensor(path: str, device: str = "cuda") -> torch.Tensor:
    """
    Returns video as float tensor [T,H,W,3] in [0,1] on specified device.
    Tries decord first; falls back to imageio.
    """
    frames_np = None

    # try decord for speed
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0))
        frames_np = vr.get_batch(list(range(len(vr)))).asnumpy()  # [T,H,W,3] uint8
    except Exception as e:
        print(f"Decord failed: {e}, trying imageio...")
        frames_np = None

    # fallback imageio
    if frames_np is None:
        import imageio.v2 as iio
        reader = iio.get_reader(path)
        frames_list = []
        for frame in reader:
            frames_list.append(frame)  # [H,W,3] uint8
        reader.close()
        frames_np = np.stack(frames_list, axis=0)

    video = torch.from_numpy(frames_np).float() / 255.0  # [T,H,W,3], float32 [0,1]
    
    # Move to GPU if available
    if device != "cpu" and torch.cuda.is_available():
        video = video.to(device)
    
    return video


############################################################
# Metric wrapper per video
############################################################

def get_metrics(
    video: torch.Tensor,
    prompt: str,
    device: str,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    pick_model_name: str = "yuvalkirstain/PickScore_v1",
):
    """
    Compute all metrics for a single video vs prompt on GPU.
    Returns a dict of floats (or None if not available).
    """

    # Ensure video is on the correct device
    if video.device.type != device and device != "cpu":
        video = video.to(device)

    # CLIPScore
    clip_sc = compute_clipscore(
        video=video,
        prompt=prompt,
        device=device,
        model_name=clip_model_name,
    )

    # PickScore
    pick_sc = compute_pickscore(
        video=video,
        prompt=prompt,
        device=device,
        model_name=pick_model_name,
    )

    # Temporal consistency
    temp_consist = compute_temporal_consistency(
        video=video,
        device=device,
        model_name=clip_model_name,
    )

    # Other metrics (some may move tensors to GPU internally)
    flick = compute_flicker(video)
    lp = compute_lpips(video, device=device)
    psnr_avg, ssim_avg = compute_psnr_ssim(video)
    mot_var = compute_motion_variance(video)

    return {
        "clipscore": clip_sc,
        "pickscore": pick_sc,
        "temporal_consistency": temp_consist,
        "flicker": flick,
        "lpips": lp,
        "psnr": psnr_avg,
        "ssim": ssim_avg,
        "motion_var": mot_var,
    }


############################################################
# Pretty-print per sample
############################################################

def print_single_report(stem, mb, mo):
    def fmt(x):
        return "None" if x is None else f"{x:.4f}"

    def delta(k):
        if mb[k] is None or mo[k] is None:
            return "None"
        return f"{(mo[k] - mb[k]):+.4f}"

    print(f"\n### Sample {stem} ###")
    print("{:<28} {:>12} {:>12} {:>12}".format(
        "Metric", "Baseline", "Ours", "Delta(O-B)"
    ))
    print("-" * 80)
    rows = [
        ("clipscore",            "CLIPScore (higher=better)"),
        ("pickscore",            "PickScore (higher=better)"),
        ("temporal_consistency", "Temporal Consistency (higher=better)"),
        ("flicker",              "Flicker MSE (lower=better)"),
        ("lpips",                "LPIPS (lower=better)"),
        ("psnr",                 "PSNR (higher=better)"),
        ("ssim",                 "SSIM (higher=better)"),
        ("motion_var",           "Motion Var (lower=better)"),
    ]
    for k, nice in rows:
        print("{:<28} {:>12} {:>12} {:>12}".format(
            nice,
            fmt(mb[k]),
            fmt(mo[k]),
            delta(k),
        ))
    print("-" * 80)


############################################################
# Summary / KEEP subset logic
############################################################

def summarize_subset(df, title, out_txt=None):
    """
    df: DataFrame with columns like clipscore_base, clipscore_ours, clipscore_delta, ...
    We'll report mean for each metric.
    """
    metrics = [
        ("clipscore", "CLIPScore (higher=better)"),
        ("pickscore", "PickScore (higher=better)"),
        ("temporal_consistency", "Temporal Consistency (higher=better)"),
        ("flicker", "Flicker MSE (lower=better)"),
        ("lpips", "LPIPS (lower=better)"),
        ("psnr", "PSNR (higher=better)"),
        ("ssim", "SSIM (higher=better)"),
        ("motion_var", "Motion Var (lower=better)"),
    ]

    lines = []
    header = f"\n========== {title} =========="
    lines.append(header)
    lines.append("{:<28} {:>12} {:>12} {:>12}".format(
        "Metric", "Baseline", "Ours", "Delta(O-B)"
    ))
    lines.append("-" + "-" * 79)

    for key, nice_name in metrics:
        b_col = f"{key}_base"
        o_col = f"{key}_ours"
        d_col = f"{key}_delta"

        b_series = df[b_col] if b_col in df else pd.Series(dtype=float)
        o_series = df[o_col] if o_col in df else pd.Series(dtype=float)
        d_series = df[d_col] if d_col in df else pd.Series(dtype=float)

        # convert to numeric, coerce non-numeric to NaN
        b_series = pd.to_numeric(b_series, errors="coerce")
        o_series = pd.to_numeric(o_series, errors="coerce")
        d_series = pd.to_numeric(d_series, errors="coerce")

        b_avg = b_series.mean()
        o_avg = o_series.mean()
        d_avg = d_series.mean()

        def fmt(x):
            if pd.isna(x):
                return "None"
            return f"{x:.4f}"

        lines.append("{:<28} {:>12} {:>12} {:>12}".format(
            nice_name,
            fmt(b_avg),
            fmt(o_avg),
            fmt(d_avg),
        ))

    lines.append("=" * 40 + "\n")

    text_block = "\n".join(lines)
    print(text_block)

    if out_txt is not None:
        with open(out_txt, "w") as f:
            f.write(text_block)


def is_keep(row):
    """
    Heuristic: keep "good and stable" samples.
    We mark True if:
      - semantic improved enough
      - AND stability/quality not obviously worse
    """

    # semantic improvement: clipscore or pickscore clearly better
    semantic_good = (
        (row.get("clipscore_delta", np.nan) is not None and
         row.get("clipscore_delta", -999) >= 0.01)
        or
        (row.get("pickscore_delta", np.nan) is not None and
         row.get("pickscore_delta", -999) >= 0.01)
    )

    # stability / quality constraints
    stable_enough = True

    flick_delta = row.get("flicker_delta", None)
    if flick_delta is not None and not pd.isna(flick_delta):
        if flick_delta > 0.002:
            stable_enough = False

    motion_delta = row.get("motion_var_delta", None)
    if motion_delta is not None and not pd.isna(motion_delta):
        if motion_delta > 3.0:
            stable_enough = False

    ssim_delta = row.get("ssim_delta", None)
    if ssim_delta is not None and not pd.isna(ssim_delta):
        if ssim_delta < -0.03:
            stable_enough = False

    return semantic_good and stable_enough


############################################################
# Plot helpers
############################################################

def plot_hist_delta(df, key, savepath):
    """
    Plot histogram of (ours - base) for a given metric key.
    """
    col = f"{key}_delta"
    if col not in df:
        return
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    if len(vals) == 0:
        return
    plt.figure()
    plt.hist(vals, bins=30)
    plt.title(f"{key} delta (ours - base)")
    plt.xlabel("delta")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_scatter(df, x_key, y_key, savepath):
    """
    Scatter of two deltas, e.g. clipscore_delta vs motion_var_delta.
    """
    x_col = f"{x_key}_delta"
    y_col = f"{y_key}_delta"
    if x_col not in df or y_col not in df:
        return

    x_vals = pd.to_numeric(df[x_col], errors="coerce").dropna().values
    y_vals = pd.to_numeric(df[y_col], errors="coerce").dropna().values

    # align lengths quickly (just min len)
    L = min(len(x_vals), len(y_vals))
    if L == 0:
        return
    x_vals = x_vals[:L]
    y_vals = y_vals[:L]

    plt.figure()
    plt.scatter(x_vals, y_vals, alpha=0.6)
    plt.axvline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel(f"{x_key} delta (ours - base)")
    plt.ylabel(f"{y_key} delta (ours - base)")
    plt.title(f"{x_key} vs {y_key} deltas")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


############################################################
# Main
############################################################

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=r"E:\let_work_work\ECCV\npnet_videos_v1\baseline_xT",
        help="Directory of BASELINE videos (pure Gaussian init, i.e. x_T/)."
    )

    parser.add_argument(
        "--ours_dir",
        type=str,
        default=r"E:\let_work_work\ECCV\npnet_videos_v1\npnet_x_star_T",
        help="Directory of OURS videos (semantic-initialized init, i.e. x_T_target/)."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Hugging Face CLIP model name"
    )

    parser.add_argument(
        "--pick_model_name",
        type=str,
        default="yuvalkirstain/PickScore_v1",
        help="Hugging Face PickScore model name"
    )

    parser.add_argument(
        "--logdir",
        type=str,
        default=r"E:\let_work_work\ECCV\npnet_videos\metrics_logs",
        help="Where to dump csv / plots / subset report."
    )

    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    # Force GPU if available
    device = args.device if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"⚠ Warning: GPU not available, using CPU")

    baseline_dir = args.baseline_dir.rstrip("/").rstrip("\\")
    ours_dir     = args.ours_dir.rstrip("/").rstrip("\\")

    # parent_dir should be the directory that also holds <stem>.txt
    parent_dir = os.path.dirname(baseline_dir)

    # collect videos
    base_videos = glob.glob(os.path.join(baseline_dir, "*.mp4"))
    ours_videos = glob.glob(os.path.join(ours_dir, "*.mp4"))

    def stem(p):
        return os.path.splitext(os.path.basename(p))[0]

    base_map = {stem(p): p for p in base_videos}
    ours_map = {stem(p): p for p in ours_videos}

    common = sorted(list(set(base_map.keys()) & set(ours_map.keys())))
    if not common:
        print("[Error] no matched filenames between baseline_dir and ours_dir")
        return

    print(f"Found {len(common)} matched pairs.")

    rows_for_csv = []

    # loop per paired sample
    for idx, st in enumerate(common, 1):
        print(f"\n[{idx}/{len(common)}] Processing: {st}")
        
        base_path = base_map[st]
        ours_path = ours_map[st]

        # read prompt from sibling txt
        prompt_path = os.path.join(parent_dir, f"{st}.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8", errors="ignore") as f:
                line1 = f.readline().strip()
            if line1.lower().startswith("prompt:"):
                prompt = line1.split(":", 1)[1].strip()
            else:
                prompt = line1
        else:
            prompt = ""
            print(f"  Warning: No prompt file found at {prompt_path}")

        try:
            # Load videos directly to GPU
            vb = load_video_as_tensor(base_path, device=device)
            vo = load_video_as_tensor(ours_path, device=device)

            # Compute metrics on GPU
            mb = get_metrics(
                vb, prompt, device,
                clip_model_name=args.clip_model_name,
                pick_model_name=args.pick_model_name,
            )
            mo = get_metrics(
                vo, prompt, device,
                clip_model_name=args.clip_model_name,
                pick_model_name=args.pick_model_name,
            )

            # debug print for this sample
            print_single_report(st, mb, mo)

            row = {
                "id": st,
                "prompt": prompt,
            }

            for key in [
                "clipscore", "pickscore", "temporal_consistency",
                "flicker", "lpips", "psnr", "ssim", "motion_var"
            ]:
                b_val = mb.get(key)
                o_val = mo.get(key)
                d_val = None
                if (b_val is not None) and (o_val is not None):
                    d_val = o_val - b_val

                row[f"{key}_base"] = b_val
                row[f"{key}_ours"] = o_val
                row[f"{key}_delta"] = d_val

            rows_for_csv.append(row)
            
            # Clear GPU cache periodically
            if device == "cuda" and idx % 10 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Error processing {st}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not rows_for_csv:
        print("[Error] No samples were successfully processed!")
        return

    # Build DataFrame with all samples
    df = pd.DataFrame(rows_for_csv)

    # make sure None -> NaN so that .mean() won't break
    df = df.map(lambda x: np.nan if x is None else x)

    csv_path = os.path.join(args.logdir, "metrics_per_sample.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved per-sample metrics] {csv_path}")

    # KEEP subset heuristic
    keep_mask = df.apply(is_keep, axis=1)
    df_keep = df[keep_mask].copy()
    df_drop = df[~keep_mask].copy()

    keep_csv = os.path.join(args.logdir, "metrics_keep_subset.csv")
    df_keep.to_csv(keep_csv, index=False)
    print(f"[Saved KEEP subset only] {keep_csv}")
    print(f"  Kept {len(df_keep)}/{len(df)} samples")

    # summarize both
    summarize_subset(
        df,
        title="Overall Averages (All Samples)",
        out_txt=os.path.join(args.logdir, "overall_all.txt")
    )
    summarize_subset(
        df_keep,
        title="Averages on KEEP Subset",
        out_txt=os.path.join(args.logdir, "overall_keep.txt")
    )

    # hist of CLIPScore deltas
    plot_hist_delta(
        df,
        "clipscore",
        savepath=os.path.join(args.logdir, "hist_clipscore_delta.png")
    )

    # scatter CLIPScore_delta vs MotionVar_delta
    plot_scatter(
        df,
        "clipscore",
        "motion_var",
        savepath=os.path.join(args.logdir, "scatter_clipscore_vs_motionvar.png")
    )

    print("\n[Done] All processing complete!")
    
    # Final GPU memory cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"✓ GPU memory cleared")


if __name__ == "__main__":
    main()