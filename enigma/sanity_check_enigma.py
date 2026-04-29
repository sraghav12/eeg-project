"""Overfit ENIGMA on 100 trials from one subject to confirm the model can learn.

If train accuracy fails to exceed 80% by the final epoch the model has a bug
and a full training run would waste GPU time. This is a hard assertion on exit.

Usage
-----
    python sanity_check_enigma.py                         # sub-02 (default)
    python sanity_check_enigma.py --subject sub-05
    python sanity_check_enigma.py --n-samples 100 --epochs 50 --lr 1e-3
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from dataset import build_datasets, collate
from model import ENIGMAConfig, ENIGMAClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=str, default="sub-02",
                    help="Subject key (e.g. sub-02). Must exist in train split.")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=150,
                    help="Memorizing 100 samples on a ~6M-param model takes "
                         "~80+ epochs; leaving headroom.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Higher than main run -- we are trying to overfit.")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="Minimum train accuracy by final epoch.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")

    train_ds, _val_ds, _test_ds, sub_to_idx, n_classes = build_datasets(
        augment_train=False
    )
    if args.subject not in sub_to_idx:
        sys.exit(f"[error] subject {args.subject} not found. "
                 f"Available: {sorted(sub_to_idx)}")

    sub_mask = (train_ds.df["subject"] == args.subject).values
    sub_indices = np.where(sub_mask)[0]
    if len(sub_indices) < args.n_samples:
        sys.exit(f"[error] only {len(sub_indices)} trials for {args.subject}, "
                 f"requested {args.n_samples}")
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(sub_indices, size=args.n_samples, replace=False).tolist()
    subset = Subset(train_ds, chosen)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)
    label_dist = np.bincount(train_ds.df.iloc[chosen]["label"].values, minlength=n_classes)
    print(f"[data] subject={args.subject} n={len(subset)} "
          f"classes_present={int((label_dist > 0).sum())}/{n_classes}")

    # Zero dropout so the model can memorize without regularisation fighting it.
    cfg = ENIGMAConfig(
        num_subjects=len(sub_to_idx),
        dropout_proj=0.0,
    )
    # Also zero the backbone dropout by patching tsencoder after construction.
    model = ENIGMAClassifier(cfg=cfg, num_classes=n_classes).to(device)
    for m in model.encoder.tsencoder.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_params/1e6:.2f}M  hidden_dim={model.encoder.hidden_dim}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    last_acc = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for eeg, y, sid in loader:
            eeg, y, sid = eeg.to(device), y.to(device), sid.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(eeg, sid)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        last_acc = correct / total
        print(f"[epoch {epoch:>3}/{args.epochs}] "
              f"loss={loss_sum/total:.4f} train_acc={last_acc:.4f}", flush=True)

    wall = time.time() - t0
    print(f"[done] {wall:.1f}s  final train_acc={last_acc:.4f}")

    assert last_acc >= args.threshold, (
        f"[FAIL] train acc {last_acc:.4f} < threshold {args.threshold}. "
        f"Model cannot overfit {args.n_samples} samples -- do NOT launch full training."
    )
    print(f"[PASS] train acc {last_acc:.4f} >= {args.threshold}")


if __name__ == "__main__":
    main()
