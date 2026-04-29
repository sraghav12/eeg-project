"""Train the ENIGMA classifier on our multi-subject EEG dataset.

Hyperparameters follow the ATM brief (hybrid recipe) for an apples-to-apples
comparison:
    optimizer      : AdamW, lr=1e-4, weight_decay=0.01
    schedule       : linear warmup (500 steps) then cosine decay
    grad clip      : 1.0
    batch size     : 64
    epochs         : 40
    normalization  : per-subject per-channel z-score (via shared dataset.py)

Logs to wandb if WANDB_API_KEY is set, else stdout only.

Checkpointing: best-by-val-accuracy is saved to
    ./checkpoints/enigma_sharmar_<YYYYMMDD-HHMM>.pt
with a sidecar JSON recording hyperparameters, val/test accuracy and wall time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset import build_datasets, collate
from model import ENIGMAConfig, ENIGMAClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int):
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup -> cosine decay to 0."""

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for eeg, y, sid in loader:
        eeg, y, sid = eeg.to(device), y.to(device), sid.to(device)
        logits, _ = model(eeg, sid)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--backbone-width", type=int, default=80,
                    help="SpatioTemporalCNN channel width (paper default 40, we use 80).")
    ap.add_argument("--emb-size", type=int, default=8,
                    help="1x1 projection output channels (paper default 4, we use 8).")
    ap.add_argument("--embed-dim", type=int, default=1024,
                    help="CLIP embedding dimension (output of MLP projector).")
    ap.add_argument("--dropout-proj", type=float, default=0.5,
                    help="Dropout in the MLP projector residual block.")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="MixUp Beta(alpha, alpha). 0 disables; try 0.2.")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="CE label smoothing; try 0.1.")
    ap.add_argument("--per-subject-head", action="store_true",
                    help="Use one Linear(embed_dim, n_classes) per subject.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument(
        "--ckpt-dir",
        type=str,
        default="/ocean/projects/cis260086p/sharmar/enigma/checkpoints",
    )
    ap.add_argument("--wandb-project", type=str, default="eeg-enigma")
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = args.run_name or f"enigma_sharmar_{stamp}"

    # Data -------------------------------------------------------------------
    train_ds, val_ds, test_ds, sub_to_idx, n_classes = build_datasets(
        augment_train=not args.no_augment
    )
    num_subjects = len(sub_to_idx)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
          f" | subjects={num_subjects} | classes={n_classes}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        worker_init_fn=_worker_init_fn, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )

    # Model ------------------------------------------------------------------
    cfg = ENIGMAConfig(
        num_subjects=num_subjects,
        embed_dim=args.embed_dim,
        backbone_width=args.backbone_width,
        emb_size=args.emb_size,
        dropout_proj=args.dropout_proj,
        per_subject_head=args.per_subject_head,
    )
    model = ENIGMAClassifier(cfg=cfg, num_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_params/1e6:.2f}M  hidden_dim={model.encoder.hidden_dim}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = make_scheduler(optimizer, args.warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Logger -----------------------------------------------------------------
    try:
        import wandb  # type: ignore
        use_wandb = os.environ.get("WANDB_API_KEY") is not None
        if use_wandb:
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            wandb.watch(model, log="gradients", log_freq=200)
    except ImportError:
        use_wandb = False
        wandb = None

    # Train ------------------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{run_name}.pt")
    meta_path = ckpt_path.replace(".pt", ".json")

    best_val_acc = 0.0
    best_epoch = -1
    train_start = time.time()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for eeg, y, sid in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            y = y.to(device)
            sid = sid.to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.mixup > 0.0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(eeg.size(0), device=eeg.device)
                eeg_mix = lam * eeg + (1.0 - lam) * eeg[perm]
                y_a, y_b = y, y[perm]
                logits, _ = model(eeg_mix, sid)
                loss = lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
            else:
                logits, _ = model(eeg, sid)
                loss = loss_fn(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total += y.size(0)

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        val_loss, val_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[epoch {epoch:>3}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={lr_now:.2e} | {epoch_time:.1f}s",
            flush=True,
        )
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, lr=lr_now,
                            epoch_time_s=epoch_time))

        if use_wandb:
            wandb.log({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now,
                "epoch_time_s": epoch_time,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "sub_to_idx": sub_to_idx,
                "n_classes": n_classes,
                "epoch": epoch,
                "val_acc": val_acc,
            }, ckpt_path)

    total_time = time.time() - train_start

    # Final test on best checkpoint -----------------------------------------
    print(f"[ckpt] loading best (epoch {best_epoch}, val_acc={best_val_acc:.4f})")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")

    meta = {
        "run_name": run_name,
        "hyperparameters": vars(args),
        "n_classes": n_classes,
        "num_subjects": num_subjects,
        "sub_to_idx": sub_to_idx,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "total_train_time_s": total_time,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "history": history,
        "trainable_params": n_params,
        "hidden_dim": model.encoder.hidden_dim,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] ckpt: {ckpt_path}\n[done] meta: {meta_path}"
          f"\n[done] wall: {total_time/60:.1f} min")

    if use_wandb:
        wandb.log({"test_acc": test_acc, "test_loss": test_loss})
        wandb.finish()


if __name__ == "__main__":
    main()
