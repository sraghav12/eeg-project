"""Average softmax probabilities across several ATM checkpoints on the test set."""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import build_datasets, collate
from model import ATMConfig, ATMClassifier


@torch.no_grad()
def predict_proba(ckpt_path: str, test_loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ATMConfig(**state["cfg"])
    model = ATMClassifier(cfg=cfg, num_classes=state["n_classes"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    probs_all, y_all = [], []
    for eeg, y, sid in test_loader:
        eeg, sid = eeg.to(device), sid.to(device)
        logits, _ = model(eeg, sid)
        probs_all.append(F.softmax(logits, dim=-1).cpu())
        y_all.append(y)
    return torch.cat(probs_all), torch.cat(y_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="Paths to .pt files to ensemble.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_ds, _, _ = build_datasets(augment_train=False)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )

    print(f"[data] test={len(test_ds)}")
    avg_probs = None
    y_ref = None
    for i, ck in enumerate(args.checkpoints):
        probs, y = predict_proba(ck, test_loader, device)
        acc = (probs.argmax(dim=-1) == y).float().mean().item()
        print(f"[{i+1}/{len(args.checkpoints)}] {ck} | single test_acc={acc:.4f}")
        if avg_probs is None:
            avg_probs, y_ref = probs, y
        else:
            assert torch.equal(y, y_ref), "test label order diverged across ckpts"
            avg_probs = avg_probs + probs

    avg_probs = avg_probs / len(args.checkpoints)
    ens_acc = (avg_probs.argmax(dim=-1) == y_ref).float().mean().item()
    print(f"[ensemble] N={len(args.checkpoints)} test_acc={ens_acc:.4f}")


if __name__ == "__main__":
    main()
