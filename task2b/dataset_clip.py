"""Task 2B dataset: EEG + precomputed CLIP targets (image & caption embeddings).

The existing train/val/test CSVs store (subject, trial_idx, npy_path, label) but
NOT image_name. We reconstruct image_name on the fly from the BIDS run-level
CSVs (*_image.csv) next to each *_1000Hz.npy, following handout Sec. 2.7.

Returned per sample (tuple):
    eeg        : float32 (122, 500)
    label      : int (category index in [0, 20))
    sub_id     : int (dense subject index in [0, n_subjects))
    img_emb    : float32 (D,)  CLIP image embedding for this trial's stimulus
    cap_emb    : float32 (D,)  CLIP caption embedding
    img_name   : str (for retrieval eval)

D = 768 for ViT-L/14 (default), 512 for ViT-B/32.

Usage:
    from dataset_clip import build_clip_datasets
    train_ds, val_ds, test_ds, sub_to_idx, n_classes, cat_protos, meta = (
        build_clip_datasets(clip_cache_dir="./clip_cache", augment_train=True)
    )
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from shared.atm.dataset import EEGDataset, load_artifacts, build_subject_index
from shared.atm.dataset import collate as _base_collate  # kept for reference


# ----------------------------------------------------------------------------
# Image-name resolution
# ----------------------------------------------------------------------------
def _resolve_image_csv(npy_path: str) -> str:
    """A run-level .npy has a sibling *_image.csv listing trial -> image."""
    # Handle both "_1000Hz.npy" and the original BIDS naming.
    for suffix in ("_1000Hz.npy", ".npy"):
        if npy_path.endswith(suffix):
            base = npy_path[: -len(suffix)]
            cand = base + "_image.csv"
            if os.path.exists(cand):
                return cand
    # Fallback: search same directory for *_image.csv
    d = os.path.dirname(npy_path)
    stem = os.path.basename(npy_path).split("_1000Hz")[0].split(".npy")[0]
    for f in os.listdir(d):
        if f.startswith(stem) and f.endswith("_image.csv"):
            return os.path.join(d, f)
    raise FileNotFoundError(f"No _image.csv found for {npy_path}")


def _attach_image_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'image_name' column to a split DataFrame.

    Uses a per-run cache: reads each _image.csv once, then indexes by trial_idx.
    Image names are the basename stem (no extension), e.g. '000002'.
    """
    df = df.copy()
    cache: dict[str, list[str]] = {}
    names = []
    for npy_path, trial_idx in zip(df["npy_path"].values, df["trial_idx"].values):
        if npy_path not in cache:
            image_csv = _resolve_image_csv(npy_path)
            img_df = pd.read_csv(image_csv)
            # FilePath is the BIDS convention per handout Sec. 2.7
            col = "FilePath" if "FilePath" in img_df.columns else img_df.columns[0]
            cache[npy_path] = [
                os.path.splitext(os.path.basename(str(p)))[0]
                for p in img_df[col].values
            ]
        trial_list = cache[npy_path]
        if trial_idx >= len(trial_list):
            raise IndexError(
                f"trial_idx={trial_idx} out of range for {npy_path} "
                f"(has {len(trial_list)} trials)"
            )
        names.append(trial_list[int(trial_idx)])
    df["image_name"] = names
    return df


# ----------------------------------------------------------------------------
# Dataset wrapper
# ----------------------------------------------------------------------------
class EEGCLIPDataset(Dataset):
    """Wraps the base EEGDataset and appends CLIP targets per trial."""

    def __init__(
        self,
        base: EEGDataset,
        image_embs: dict,     # {image_name: (D,) float32 tensor}
        caption_embs: dict,   # {image_name: (D,) float32 tensor}
        clip_dim: int,
    ):
        self.base = base
        self.image_embs = image_embs
        self.caption_embs = caption_embs
        self.clip_dim = clip_dim

        # base.df must have been enriched with image_name already.
        assert "image_name" in base.df.columns, (
            "Base DataFrame missing 'image_name'. Use _attach_image_names first."
        )
        self._names = base.df["image_name"].astype(str).values

        # Stats: how many trials have targets?
        n_img = sum(n in image_embs for n in self._names)
        n_cap = sum(n in caption_embs for n in self._names)
        print(
            f"[EEGCLIPDataset] n={len(base)} | img_emb hits={n_img}/{len(base)} "
            f"| cap_emb hits={n_cap}/{len(base)}"
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        eeg, label, sid = self.base[idx]
        name = self._names[idx]
        img_emb = self.image_embs.get(name)
        cap_emb = self.caption_embs.get(name)
        if img_emb is None:
            img_emb = torch.zeros(self.clip_dim, dtype=torch.float32)
        if cap_emb is None:
            cap_emb = torch.zeros(self.clip_dim, dtype=torch.float32)
        return eeg, int(label), int(sid), img_emb, cap_emb, name


def clip_collate(batch):
    eegs, labels, sids, imgs, caps, names = zip(*batch)
    return (
        torch.stack(eegs),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(sids, dtype=torch.long),
        torch.stack(imgs),
        torch.stack(caps),
        list(names),
    )


# ----------------------------------------------------------------------------
# Top-level builder
# ----------------------------------------------------------------------------
def build_clip_datasets(
    clip_cache_dir: str,
    augment_train: bool = True,
    image_embs_file: str = "image_embs.pt",
    caption_embs_file: str = "caption_embs.pt",
    category_embs_file: str = "category_embs.pt",
    category_names_file: str = "category_names.json",
):
    """Build train/val/test EEGCLIPDatasets + return category prototypes + meta.

    Returns:
        (train_ds, val_ds, test_ds, sub_to_idx, n_classes,
         cat_protos: (20, D) tensor,
         meta: dict)
    """
    import json
    from shared.atm.dataset import build_datasets as _build_base

    # Base datasets (your existing loader)
    train_ds, val_ds, test_ds, sub_to_idx, n_classes = _build_base(
        augment_train=augment_train
    )

    # Enrich each df with image_name (done once per split).
    for ds in (train_ds, val_ds, test_ds):
        if "image_name" not in ds.df.columns:
            ds.df = _attach_image_names(ds.df)

    # Load precomputed CLIP embeddings
    img_path = os.path.join(clip_cache_dir, image_embs_file)
    cap_path = os.path.join(clip_cache_dir, caption_embs_file)
    cat_path = os.path.join(clip_cache_dir, category_embs_file)
    names_path = os.path.join(clip_cache_dir, category_names_file)
    if not (os.path.exists(img_path) and os.path.exists(cap_path)):
        raise FileNotFoundError(
            f"CLIP cache missing. Run clip_utils.py to build it first.\n"
            f"  expected: {img_path}\n  expected: {cap_path}"
        )

    image_embs = torch.load(img_path, weights_only=False)
    caption_embs = torch.load(cap_path, weights_only=False)
    cat_protos = torch.load(cat_path, weights_only=False) if os.path.exists(cat_path) else None
    with open(names_path) as f:
        category_names = json.load(f) if os.path.exists(names_path) else None

    # Infer CLIP dim from any image embedding
    sample_emb = next(iter(image_embs.values()))
    clip_dim = int(sample_emb.numel())
    print(f"[clip_cache] dim={clip_dim} | n_images={len(image_embs)} | "
          f"n_captions={len(caption_embs)} | categories={len(category_names) if category_names else '?'}")

    train_wrap = EEGCLIPDataset(train_ds, image_embs, caption_embs, clip_dim)
    val_wrap = EEGCLIPDataset(val_ds, image_embs, caption_embs, clip_dim)
    test_wrap = EEGCLIPDataset(test_ds, image_embs, caption_embs, clip_dim)

    meta = dict(
        clip_dim=clip_dim,
        category_names=category_names,
        n_classes=n_classes,
    )
    return train_wrap, val_wrap, test_wrap, sub_to_idx, n_classes, cat_protos, meta


if __name__ == "__main__":
    # Smoke test — requires clip_cache already built
    import sys
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "./clip_cache"
    train_ds, val_ds, test_ds, sti, nc, cp, meta = build_clip_datasets(cache_dir)
    eeg, y, sid, img, cap, name = train_ds[0]
    print(f"sample: eeg={tuple(eeg.shape)} label={y} sub={sid} "
          f"img_emb={tuple(img.shape)} cap_emb={tuple(cap.shape)} name={name!r}")
    print(f"cat_protos: {tuple(cp.shape) if cp is not None else None}")
