"""
NPNet-V1 训练脚本（Proposal Ver）
- 黄金 2 : 普通 8
- 余弦学习率 + warmup
- τ 余弦衰减
- Charbonnier + Temporal DCT 正则
- 全 float32
"""

import os, math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from modelV1 import NPNetV
from data import PrecomputedNoiseDataset, safe_collate
from lossV1 import NPNetVLoss


torch.set_default_dtype(torch.float32)


# -----------------------------
# lr schedule
# -----------------------------
def cosine_with_warmup(step, total, base, min_lr, warmup):
    if step < warmup:
        return base * float(step + 1) / float(max(1, warmup))
    prog = (step - warmup) / float(max(1, total - warmup))
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * prog))


def cosine_tau(step, total, t0, t1):
    prog = step / float(max(1, total))
    return t1 + 0.5 * (t0 - t1) * (1 + math.cos(math.pi * prog))


# -----------------------------
# main
# -----------------------------
def main():
    device = config.DEVICE

    # model
    npnet = NPNetV(
        channels=config.CHANNELS,
        T=config.TEMPORAL_DIM,
        H=config.HEIGHT,
        W=config.WIDTH,
        freq_decay=config.FREQ_DECAY
    ).to(device)

    criterion = NPNetVLoss(
        tau=config.TAU_START,
        temporal_low_freq_k=config.LOSS_TEMP_LOW_FREQ_K,
        charbonnier_eps=1e-6
    ).to(device)

    optimizer = optim.AdamW(npnet.parameters(), lr=config.LR, weight_decay=0.01)

    # Text encoder
    clip_path = "/root/autodl-tmp/VideoCrafter/local_clip_model"
    tokenizer = CLIPTokenizer.from_pretrained(clip_path)
    text_encoder = CLIPTextModel.from_pretrained(clip_path, torch_dtype=torch.float32).to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    # data
    weak_dataset = PrecomputedNoiseDataset(config.WEAK_PAIRS_DIR)
    golden_dataset = PrecomputedNoiseDataset(config.GOLDEN_PAIRS_DIR)

    weak_loader = DataLoader(
        weak_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
        collate_fn=safe_collate
    )
    golden_loader = DataLoader(
        golden_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
        collate_fn=safe_collate
    )

    steps_per_epoch = max(len(weak_loader), len(golden_loader))
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    warmup = max(100, int(0.05 * total_steps))
    min_lr = config.LR * 0.1

    npnet.train()
    global_step = 0

    losses = []
    main_losses = []
    temp_losses = []
    all_steps = []

    print("------ Training NPNet-V1 (Proposal Ver) ------")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{config.NUM_EPOCHS} =====")

        weak_iter = iter(weak_loader)
        gold_iter = iter(golden_loader)

        for _ in tqdm(range(steps_per_epoch)):
            # 80% weak, 20% golden
            if torch.rand(1).item() < config.WEAK_SUPERVISION_RATIO:
                try:
                    batch = next(weak_iter)
                except StopIteration:
                    weak_iter = iter(weak_loader)
                    batch = next(weak_iter)
            else:
                try:
                    batch = next(gold_iter)
                except StopIteration:
                    gold_iter = iter(golden_loader)
                    batch = next(gold_iter)

            if batch is None or batch[0] is None:
                continue

            x_T, x_T_target, texts = batch
            x_T = x_T.to(device)
            x_T_target = x_T_target.to(device)

            # text embedding
            with torch.no_grad():
                inputs = tokenizer(
                    texts, padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt"
                )
                out = text_encoder(inputs.input_ids.to(device))
                hidden = out.last_hidden_state.float()
                mask = inputs.attention_mask.to(device).unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                E_txt = pooled.float()

            # lr & tau
            lr_now = cosine_with_warmup(global_step, total_steps, config.LR, min_lr, warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            tau_now = cosine_tau(global_step, total_steps, config.TAU_START, config.TAU_END)

            # forward
            x_star = npnet(x_T, E_txt)
            loss, L_main, L_temp = criterion(x_star, x_T_target, tau_override=tau_now)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(npnet.parameters(), 1.0)
            optimizer.step()

            if global_step % config.LOG_STEP == 0:
                losses.append(loss.item())
                main_losses.append(L_main.item())
                temp_losses.append(L_temp.item())
                all_steps.append(global_step)

            global_step += 1

    # save loss curve (V1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.plot(all_steps, losses); plt.title("Total Loss V1")
    plt.subplot(1, 3, 2); plt.plot(all_steps, main_losses); plt.title("Main Loss V1")
    plt.subplot(1, 3, 3); plt.plot(all_steps, temp_losses); plt.title("Temp Loss V1")
    plt.tight_layout()
    plt.savefig("loss_curve_V1.png")

    # save checkpoint (V1)
    torch.save(npnet.state_dict(), "npnet_v1_final.pth")
    print("Saved npnet_v1_final.pth")


if __name__ == "__main__":
    main()
