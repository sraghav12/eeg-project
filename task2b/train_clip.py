"""Task 2B training: EEG -> CLIP alignment with full ablation support.

Design decisions optimized for best classification accuracy on this dataset:

  1. Align EEG to CAPTION embeddings as the primary contrastive target
     (handout's ask), but ALSO use CLIP IMAGE embeddings as the KD teacher.
     This gives strong supervision from both modalities.

  2. Category cross-entropy (text-prompt prototypes) with HEAVY weight (1.5).
     This directly optimizes the zero-shot classification eval and is by far
     the biggest lever we have for pushing accuracy up.

  3. Initialize encoder from your best Task 1 classifier checkpoint.
     Keep encoder trainable at low LR (lr_encoder=5e-5) — fully freezing
     typically costs 1-2 points.

  4. EMA on projection head + encoder for final eval.

  5. ViT-L/14 as the CLIP model (768-d, stronger teacher than B/32).

Ablations via CLI flags:
    --no-infonce      : disable contrastive loss
    --no-kd           : disable logit KD from image teacher
    --no-cosine       : disable cosine anchor
    --no-catce        : disable category CE (big drop expected)
    --debias-mode     : class | similarity | hybrid | none
    --text-strategy   : frozen | partial | lora | adapter
    --freeze-encoder  : projection-only training
    --target          : caption | image | both (contrastive target)

Usage (best single run):
    python train_clip.py \\
        --encoder-ckpt checkpoints/atm_sharmar_20260419-0319.pt \\
        --clip-cache clip_cache \\
        --clip-model openai/clip-vit-large-patch14 \\
        --run-name full_s42 --seed 42
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# From your existing code
from shared.atm.model import ATMConfig, ATMClassifier, ATMEncoder

# Task 2B modules
from shared.atm.task2b.dataset_clip import build_clip_datasets, clip_collate
from shared.atm.task2b.losses import (SymmetricInfoNCE, DebiasedInfoNCE, LogitKDLoss,
                    CosineAlignLoss, CategoryCELoss)
from shared.atm.task2b.model_clip import ProjectionHead, setup_clip_text_encoder, EMA


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return LambdaLR(optimizer, lr_lambda)


def _worker_init_fn(worker_id: int):
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed); random.seed(seed)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # --- Target / strategy toggles ---
    target: str = "caption"          # 'caption' | 'image' | 'both'
    text_strategy: str = "frozen"    # 'frozen' | 'partial' | 'lora' | 'adapter'
    freeze_encoder: bool = False

    # --- Loss toggles ---
    use_infonce: bool = True
    use_kd: bool = True
    use_cosine: bool = True
    use_category_ce: bool = True
    debias_mode: str = "class"       # 'class' | 'similarity' | 'hybrid' | 'none'

    # --- Loss weights ---
    w_infonce: float = 1.0
    w_kd: float = 1.0
    w_cosine: float = 0.3
    w_category_ce: float = 1.5       # heavier: directly optimizes eval
    w_image_aux: float = 0.5         # weight for image target when target='both'
    debias_alpha: float = 0.1

    # --- Temperatures ---
    tau_kd_teacher: float = 0.04
    tau_kd_student: float = 0.07
    tau_category: float = 0.07

    # --- Training ---
    batch_size: int = 256
    grad_accum: int = 1
    epochs: int = 50
    lr_proj: float = 5e-4
    lr_encoder: float = 5e-5
    lr_clip: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    label_smoothing: float = 0.1
    seed: int = 42

    # --- Partial / LoRA / adapter hyperparams ---
    partial_n_layers: int = 2
    lora_r: int = 8
    lora_alpha: int = 16
    adapter_bottleneck: int = 64


# ----------------------------------------------------------------------------
# Core training step
# ----------------------------------------------------------------------------
def pick_target(cfg: Config, img_z: torch.Tensor, cap_z: torch.Tensor):
    """Select the positive target for contrastive / KD / cosine losses.

    For 'both', use caption as primary and image as auxiliary (added below).
    Rows where cap_z is all-zero (missing caption) fall back to img_z.
    """
    missing_cap = (cap_z.norm(dim=-1) < 1e-6)
    if cfg.target == "image":
        return img_z
    if cfg.target == "caption":
        # Fallback for missing captions -> use image to avoid zero positive.
        tgt = cap_z.clone()
        if missing_cap.any():
            tgt[missing_cap] = img_z[missing_cap]
        return F.normalize(tgt, dim=-1)
    # 'both' -> primary caption (with fallback), img added as aux loss later.
    tgt = cap_z.clone()
    if missing_cap.any():
        tgt[missing_cap] = img_z[missing_cap]
    return F.normalize(tgt, dim=-1)


def forward_step(cfg, encoder, proj, clip_model,
                 cat_protos, losses, batch, device, captions=None,
                 cache_text_embs=None):
    """One training step: compute all active losses and their sum.

    Args:
        cache_text_embs: if CLIP text tower is FROZEN, we use precomputed
            caption embeddings from the batch (no CLIP forward needed).
            If text is trainable (partial/lora/adapter), caller must pass
            tokenized captions so we recompute on each step.
    """
    eeg, labels, sids, img_z_pre, cap_z_pre, _names = batch
    eeg, labels, sids = eeg.to(device), labels.to(device), sids.to(device)

    img_z = F.normalize(img_z_pre.to(device), dim=-1)
    cap_z_pre = cap_z_pre.to(device)
    # cap_z may be zeros for images without a caption — normalize would produce NaN
    cap_z = cap_z_pre.clone()
    valid = cap_z.norm(dim=-1) > 1e-6
    cap_z[valid] = F.normalize(cap_z[valid], dim=-1)

    # If we're training the text encoder, recompute caption embeddings live.
    # We ALWAYS re-encode here when clip_model is provided + has trainable params,
    # otherwise use the precomputed batch embeddings.
    if clip_model is not None and any(p.requires_grad for p in clip_model.parameters()):
        assert captions is not None, (
            "Text tower is trainable -> must pass tokenized captions"
        )
        tok = captions  # dict with input_ids, attention_mask on device
        text_feats = clip_model.get_text_features(**tok)
        cap_z = F.normalize(text_feats, dim=-1)

    tgt_z = pick_target(cfg, img_z, cap_z)

    # --- Student forward ---
    feat = encoder(eeg, sids)                 # (B, 1024)
    eeg_z = proj(feat)                         # (B, D) normalized

    # logit_scale lives on the encoder
    ls = encoder.logit_scale.exp().clamp(max=100.0)

    total = eeg_z.new_zeros(())
    logs = {}

    # ----- InfoNCE -----
    if cfg.use_infonce:
        if cfg.debias_mode == "none":
            l = losses["infonce"](eeg_z, tgt_z, ls)
        else:
            l = losses["debias"](eeg_z, tgt_z, ls, labels)
        total = total + cfg.w_infonce * l
        logs["infonce"] = l.detach()

        # If target='both', add an image-InfoNCE auxiliary term
        if cfg.target == "both":
            l2 = (losses["infonce"](eeg_z, img_z, ls) if cfg.debias_mode == "none"
                  else losses["debias"](eeg_z, img_z, ls, labels))
            total = total + cfg.w_image_aux * l2
            logs["infonce_img"] = l2.detach()

    # ----- Logit KD: image teacher over caption candidates -----
    if cfg.use_kd:
        # teacher: img -> captions ; student: eeg -> captions
        # candidates = tgt_z (caption batch). Rows with missing caption default to image.
        l = losses["kd"](eeg_z, img_z, tgt_z)
        total = total + cfg.w_kd * l
        logs["kd"] = l.detach()

    # ----- Cosine anchor -----
    if cfg.use_cosine:
        l = losses["cos"](eeg_z, tgt_z)
        total = total + cfg.w_cosine * l
        logs["cos"] = l.detach()

    # ----- Category CE (the accuracy lever) -----
    if cfg.use_category_ce:
        l = losses["catce"](eeg_z, cat_protos, labels)
        total = total + cfg.w_category_ce * l
        logs["catce"] = l.detach()

    logs["total"] = total.detach()
    return total, logs


# ----------------------------------------------------------------------------
# Evaluation: zero-shot classification via category prototypes
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_zeroshot(encoder, proj, loader, cat_protos, device,
                      return_preds: bool = False):
    encoder.eval(); proj.eval()
    correct, total = 0, 0
    per_subj: dict[int, list[int]] = {}
    all_preds, all_labels, all_sids = [], [], []
    for batch in loader:
        eeg, labels, sids, _img, _cap, _names = batch
        eeg, labels, sids = eeg.to(device), labels.to(device), sids.to(device)
        feat = encoder(eeg, sids)
        eeg_z = proj(feat)
        logits = eeg_z @ cat_protos.T
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item(); total += labels.size(0)
        for s, p, y in zip(sids.cpu().tolist(), pred.cpu().tolist(), labels.cpu().tolist()):
            d = per_subj.setdefault(s, [0, 0]); d[0] += int(p == y); d[1] += 1
        if return_preds:
            all_preds.append(pred.cpu()); all_labels.append(labels.cpu()); all_sids.append(sids.cpu())
    acc = correct / max(1, total)
    per = {s: c / max(1, t) for s, (c, t) in per_subj.items()}
    if return_preds:
        return acc, per, (torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_sids))
    return acc, per


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-ckpt", required=True,
                    help="Path to your best Task 1 ATMClassifier checkpoint.")
    ap.add_argument("--clip-cache", required=True,
                    help="Directory produced by clip_utils.py")
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out-dir", default="./clip_runs")
    ap.add_argument("--run-name", default=None)

    # Config overrides
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr-proj", type=float, default=5e-4)
    ap.add_argument("--lr-encoder", type=float, default=5e-5)
    ap.add_argument("--lr-clip", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)

    # Target
    ap.add_argument("--target", choices=["caption", "image", "both"], default="caption")

    # Strategies
    ap.add_argument("--text-strategy", choices=["frozen", "partial", "lora", "adapter"],
                    default="frozen")
    ap.add_argument("--freeze-encoder", action="store_true")

    # Loss toggles
    ap.add_argument("--no-infonce", action="store_true")
    ap.add_argument("--no-kd", action="store_true")
    ap.add_argument("--no-cosine", action="store_true")
    ap.add_argument("--no-catce", action="store_true")
    ap.add_argument("--debias-mode", choices=["class", "similarity", "hybrid", "none"],
                    default="class")

    # Loss weights (handy for ablations)
    ap.add_argument("--w-infonce", type=float, default=1.0)
    ap.add_argument("--w-kd", type=float, default=1.0)
    ap.add_argument("--w-cosine", type=float, default=0.3)
    ap.add_argument("--w-catce", type=float, default=1.5)

    args = ap.parse_args()

    cfg = Config(
        target=args.target,
        text_strategy=args.text_strategy,
        freeze_encoder=args.freeze_encoder,
        use_infonce=not args.no_infonce,
        use_kd=not args.no_kd,
        use_cosine=not args.no_cosine,
        use_category_ce=not args.no_catce,
        debias_mode=args.debias_mode,
        w_infonce=args.w_infonce,
        w_kd=args.w_kd,
        w_cosine=args.w_cosine,
        w_category_ce=args.w_catce,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr_proj=args.lr_proj,
        lr_encoder=args.lr_encoder,
        lr_clip=args.lr_clip,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = args.run_name or f"clip_{stamp}_s{cfg.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[run] {run_name} | device={device} | config={cfg}")

    # ---- Data ----
    train_ds, val_ds, test_ds, sub_to_idx, n_classes, cat_protos, meta = build_clip_datasets(
        args.clip_cache, augment_train=True,
    )
    cat_protos = F.normalize(cat_protos.to(device), dim=-1)
    clip_dim = meta["clip_dim"]
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"| classes={n_classes} | clip_dim={clip_dim}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=clip_collate, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=clip_collate, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=clip_collate, num_workers=args.num_workers,
                             pin_memory=True)

    # ---- Encoder: init from Task 1 classifier ckpt ----
    state = torch.load(args.encoder_ckpt, weights_only=False, map_location=device)
    enc_cfg = ATMConfig(**state["cfg"])
    clf = ATMClassifier(cfg=enc_cfg, num_classes=state["n_classes"])
    clf.load_state_dict(state["model_state"])
    encoder: ATMEncoder = clf.encoder.to(device)
    if cfg.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
    # Ensure logit_scale is trainable if it exists
    if hasattr(encoder, "logit_scale"):
        encoder.logit_scale.requires_grad = True

    # ---- Projection head ----
    proj = ProjectionHead(
        in_dim=enc_cfg.embed_dim, out_dim=clip_dim, hidden=max(1024, clip_dim),
        dropout=0.5,
    ).to(device)

    # ---- CLIP text encoder (optional trainable) ----
    clip_model = None
    clip_trainable: list[nn.Parameter] = []
    clip_processor = None
    if cfg.text_strategy != "frozen":
        clip_model, clip_trainable = setup_clip_text_encoder(
            args.clip_model, strategy=cfg.text_strategy,
            partial_n_layers=cfg.partial_n_layers,
            lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            adapter_bottleneck=cfg.adapter_bottleneck,
            device=str(device),
        )
        # We'll also need the processor to re-tokenise captions on each step.
        from transformers import CLIPProcessor
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
        # Load captions DataFrame so we can fetch caption text for each trial image
        import pandas as pd
        cap_df = pd.read_csv(os.path.join(args.clip_cache, "captions_df.csv"))
        name_to_caption = dict(zip(cap_df["image_name"].astype(str),
                                   cap_df["abstracted"].astype(str)))
    else:
        name_to_caption = None

    # ---- Losses ----
    losses = dict(
        infonce=SymmetricInfoNCE(),
        debias=DebiasedInfoNCE(mode=cfg.debias_mode, alpha=cfg.debias_alpha),
        kd=LogitKDLoss(tau_teacher=cfg.tau_kd_teacher, tau_student=cfg.tau_kd_student),
        cos=CosineAlignLoss(),
        catce=CategoryCELoss(tau=cfg.tau_category, label_smoothing=cfg.label_smoothing),
    )

    # ---- Optimiser ----
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    groups = [{"params": proj.parameters(), "lr": cfg.lr_proj,
               "weight_decay": cfg.weight_decay}]
    if enc_params:
        groups.append({"params": enc_params, "lr": cfg.lr_encoder,
                       "weight_decay": cfg.weight_decay})
    if clip_trainable:
        groups.append({"params": clip_trainable, "lr": cfg.lr_clip,
                       "weight_decay": cfg.weight_decay})
    optimizer = AdamW(groups)
    total_steps = max(1, cfg.epochs * len(train_loader) // cfg.grad_accum)
    scheduler = cosine_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    # ---- EMA ----
    ema_proj = EMA(proj, cfg.ema_decay)
    ema_enc = EMA(encoder, cfg.ema_decay) if enc_params else None

    # ---- Train ----
    best_val, best_epoch, history = 0.0, -1, []
    ckpt_path = os.path.join(args.out_dir, f"{run_name}.pt")
    meta_path = ckpt_path.replace(".pt", ".json")
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        encoder.train(); proj.train()
        if clip_model is not None:
            clip_model.train(any(p.requires_grad for p in clip_model.parameters()))

        running = {k: 0.0 for k in
                   ("total", "infonce", "infonce_img", "kd", "cos", "catce")}
        n_seen = 0
        t_ep = time.time()

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            # If training the text tower, tokenise the batch's captions here
            batch_captions = None
            if clip_model is not None and clip_trainable:
                _eeg, _labels, _sids, _img, _cap, names = batch
                texts = [name_to_caption.get(n, "") for n in names]
                tok = clip_processor.tokenizer(
                    texts, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                batch_captions = tok

            loss, logs = forward_step(
                cfg, encoder, proj, clip_model, cat_protos, losses,
                batch, device, captions=batch_captions,
            )
            (loss / cfg.grad_accum).backward()

            if (step + 1) % cfg.grad_accum == 0:
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]],
                        cfg.grad_clip,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema_proj.update(proj)
                if ema_enc is not None:
                    ema_enc.update(encoder)

            bs = batch[0].size(0); n_seen += bs
            for k in running:
                if k in logs:
                    running[k] += logs[k].item() * bs

        # ---- Validation with EMA weights ----
        ema_proj.apply_to(proj)
        if ema_enc is not None: ema_enc.apply_to(encoder)
        val_acc, val_per = evaluate_zeroshot(encoder, proj, val_loader, cat_protos, device)
        ema_proj.restore(proj)
        if ema_enc is not None: ema_enc.restore(encoder)

        ep_time = time.time() - t_ep
        msg = {k: v / max(1, n_seen) for k, v in running.items()}
        msg.update(dict(epoch=epoch, val_acc=val_acc, ep_time_s=ep_time,
                        lr=scheduler.get_last_lr()[0]))
        history.append(msg)
        print(f"[ep {epoch:>3}/{cfg.epochs}] "
              f"total={msg['total']:.3f} "
              f"catce={msg.get('catce', 0):.3f} "
              f"infonce={msg.get('infonce', 0):.3f} "
              f"kd={msg.get('kd', 0):.3f} "
              f"cos={msg.get('cos', 0):.3f} | "
              f"val_acc={val_acc:.4f} | {ep_time:.1f}s", flush=True)

        if val_acc > best_val:
            best_val = val_acc; best_epoch = epoch
            torch.save({
                "proj_state": proj.state_dict(),
                "encoder_state": encoder.state_dict(),
                "ema_proj": ema_proj.shadow,
                "ema_encoder": ema_enc.shadow if ema_enc is not None else None,
                "enc_cfg": state["cfg"],
                "n_classes": n_classes,
                "sub_to_idx": sub_to_idx,
                "clip_model": args.clip_model,
                "clip_dim": clip_dim,
                "config": asdict(cfg),
                "epoch": epoch, "val_acc": val_acc,
            }, ckpt_path)

    # ---- Final test with best EMA ----
    print(f"[best] epoch={best_epoch} val_acc={best_val:.4f}")
    ck = torch.load(ckpt_path, weights_only=False, map_location=device)
    proj.load_state_dict(ck["proj_state"])
    encoder.load_state_dict(ck["encoder_state"])
    with torch.no_grad():
        for n, p in proj.named_parameters():
            if n in ck["ema_proj"]: p.data.copy_(ck["ema_proj"][n])
        if ck.get("ema_encoder"):
            for n, p in encoder.named_parameters():
                if n in ck["ema_encoder"]: p.data.copy_(ck["ema_encoder"][n])

    test_acc, test_per, test_preds = evaluate_zeroshot(
        encoder, proj, test_loader, cat_protos, device, return_preds=True,
    )
    print(f"[test] acc={test_acc:.4f}")
    print(f"[test] per-subject: {test_per}")

    # Save predictions for later ensembling
    preds, labels, sids = test_preds
    torch.save({
        "preds": preds, "labels": labels, "sids": sids,
        "test_acc": test_acc, "test_per_subject": test_per,
    }, ckpt_path.replace(".pt", "_testpreds.pt"))

    with open(meta_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "config": asdict(cfg),
            "args": vars(args),
            "best_val_acc": best_val,
            "best_epoch": best_epoch,
            "test_acc": test_acc,
            "test_per_subject": test_per,
            "wall_time_s": time.time() - t_start,
            "history": history,
            "trainable_params": {
                "encoder": sum(p.numel() for p in enc_params),
                "proj": sum(p.numel() for p in proj.parameters()),
                "clip": sum(p.numel() for p in clip_trainable),
            },
        }, f, indent=2)
    print(f"[done] ckpt={ckpt_path} meta={meta_path}")


if __name__ == "__main__":
    main()
