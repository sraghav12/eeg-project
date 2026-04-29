"""Generate a Kaggle submission CSV from an ATM checkpoint.

Kaggle format (per 2026-04-23 announcement):
    Id,Category,CategoryName
    sub-02_ses-01_run-01_0,10,diningtable
    ...
26,000 rows total (all subjects/sessions/runs/trials -- NOT the val/test split).

Writes to sharmar/atm/submission_atm_<ckpt>.csv.

Usage
-----
    python make_submission.py                     # default: best-test ckpt
    python make_submission.py --ckpt path/to.pt
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import ArtifactPaths, EEGDataset, collate, load_artifacts
from model import ATMConfig, ATMClassifier


DEFAULT_CKPT = "/ocean/projects/cis260086p/sharmar/atm/checkpoints/atm_sharmar_20260419-0319.pt"
MASTER_CSV = "/ocean/projects/cis260086p/shared/eeg-project/artifacts/master_trials.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--master-csv", type=str, default=MASTER_CSV)
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Default: sharmar/atm/submission_atm_<ckpt>.csv",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")

    # Load checkpoint -------------------------------------------------------
    print(f"[ckpt] loading {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ATMConfig(**state["cfg"])
    n_classes = state["n_classes"]
    sub_to_idx = state["sub_to_idx"]
    print(
        f"[ckpt] epoch={state.get('epoch')} val_acc={state.get('val_acc'):.4f} "
        f"classes={n_classes} subjects={len(sub_to_idx)}"
    )

    model = ATMClassifier(cfg=cfg, num_classes=n_classes).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Build full-dataset (all 26k trials) with training-time preprocessing --
    art = load_artifacts()
    full_df = pd.read_csv(args.master_csv).reset_index(drop=True)
    print(f"[data] master_trials rows = {len(full_df)}")

    idx_to_cat = art["labels"]["idx_to_cat"]

    full_ds = EEGDataset(
        full_df,
        norm_stats=art["norm"]["per_subject"],
        clip_threshold=art["norm"]["clip_threshold"],
        bad_channels=art["bad"],
        sub_to_idx=sub_to_idx,
        augment=False,
    )
    loader = DataLoader(
        full_ds,
        batch_size=args.batch_size,
        shuffle=False,                     # preserve row order -> Id order
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Inference -------------------------------------------------------------
    preds = []
    correct = 0
    total = 0
    with torch.no_grad():
        for eeg, y, sid in loader:
            eeg, sid = eeg.to(device, non_blocking=True), sid.to(device)
            logits, _ = model(eeg, sid)
            p = logits.argmax(dim=1).cpu().numpy()
            preds.append(p)
            correct += int((p == y.numpy()).sum())
            total += y.size(0)
    preds = np.concatenate(preds)
    print(f"[full] acc on 26k (incl. train+val) = {correct/max(1,total):.4f} on {total} trials")

    # Report acc on the split the checkpoint was evaluated on, as a sanity
    # check: pick the test split rows out of the full predictions.
    test_key = set(
        zip(art["test_df"]["subject"], art["test_df"]["session"],
            art["test_df"]["run"], art["test_df"]["trial_idx"])
    )
    full_key = list(zip(full_df["subject"], full_df["session"],
                        full_df["run"], full_df["trial_idx"]))
    test_mask = np.array([k in test_key for k in full_key])
    test_correct = int((preds[test_mask] == full_df["label"].values[test_mask]).sum())
    test_total = int(test_mask.sum())
    print(f"[sanity] test-split acc (should match ckpt ~0.12): "
          f"{test_correct/max(1,test_total):.4f} on {test_total} trials")

    # Build Kaggle submission ----------------------------------------------
    ids = (
        full_df["subject"].astype(str)
        + "_" + full_df["session"].astype(str)
        + "_" + full_df["run"].astype(str)
        + "_" + full_df["trial_idx"].astype(int).astype(str)
    )
    sub_df = pd.DataFrame({
        "Id": ids,
        "Category": preds.astype(int),
        "CategoryName": [idx_to_cat[int(p)] for p in preds],
    })

    if args.out is None:
        stem = Path(args.ckpt).stem
        args.out = f"/ocean/projects/cis260086p/sharmar/atm/submission_atm_{stem}_kaggle.csv"
    sub_df.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out} ({len(sub_df)} rows)")
    print("[done] header:", ",".join(sub_df.columns))
    print(sub_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
