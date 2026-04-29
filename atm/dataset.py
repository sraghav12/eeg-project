"""Private dataset loader for the ATM training pipeline.

Reads the team's preprocessed artifacts in
    /ocean/projects/cis260086p/shared/eeg-project/artifacts/
read-only and reproduces the preprocessing used by the EEGNet baseline
(clip -> per-subject per-channel z-score -> bad-channel masking). This is
intentionally kept under sharmar/ so that no file in shared/ is touched.

Raw .npy layout on disk (1 file per run):
    shape (100 trials, 500 timepoints, 122 channels), float32
    i.e. 100 trials per run at 1000 Hz, 0-500 ms post-stimulus.

Returned per sample:
    eeg        : torch.float32 (122, 500)  -- channels first
    label      : int                       -- category index in [0, 20)
    subject_id : int                       -- subject index in [0, n_subjects)
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SHARED_ARTIFACTS = "/ocean/projects/cis260086p/shared/eeg-project/artifacts"


@dataclass
class ArtifactPaths:
    root: str = SHARED_ARTIFACTS

    @property
    def train_csv(self) -> str:
        return os.path.join(self.root, "train.csv")

    @property
    def val_csv(self) -> str:
        return os.path.join(self.root, "val.csv")

    @property
    def test_csv(self) -> str:
        return os.path.join(self.root, "test.csv")

    @property
    def norm_stats(self) -> str:
        return os.path.join(self.root, "norm_stats.pkl")

    @property
    def bad_channels(self) -> str:
        return os.path.join(self.root, "bad_channels.pkl")

    @property
    def label_mappings(self) -> str:
        return os.path.join(self.root, "label_mappings.pkl")


def load_artifacts(paths: Optional[ArtifactPaths] = None):
    p = paths or ArtifactPaths()
    with open(p.norm_stats, "rb") as f:
        norm = pickle.load(f)
    with open(p.bad_channels, "rb") as f:
        bad = pickle.load(f)
    with open(p.label_mappings, "rb") as f:
        lbl = pickle.load(f)
    train_df = pd.read_csv(p.train_csv)
    val_df = pd.read_csv(p.val_csv)
    test_df = pd.read_csv(p.test_csv)
    return dict(
        norm=norm,
        bad=bad,
        labels=lbl,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


def build_subject_index(*dfs: pd.DataFrame) -> dict:
    """Union of subjects across splits, sorted -> contiguous 0..N-1 index."""
    subjects = sorted(set().union(*[set(df["subject"].unique()) for df in dfs]))
    return {s: i for i, s in enumerate(subjects)}


class EEGDataset(Dataset):
    """Row -> trial dataset. Mirrors EEGNet baseline preprocessing.

    Args:
        df             : split DataFrame with columns subject/trial_idx/npy_path/label.
        norm_stats     : dict[subject] -> {'mean': (122,), 'std': (122,)}.
        clip_threshold : absolute-value clip before normalization.
        bad_channels   : dict[subject] -> list[int]; flagged channels are zeroed
                         after normalization (matching EEGNet baseline).
        sub_to_idx     : {subject_str: int}. Must cover every subject in df.
        max_cache      : number of run-level .npy files to LRU-cache.
        augment        : apply EEGNet-style stochastic augmentations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        norm_stats: dict,
        clip_threshold: float,
        bad_channels: dict,
        sub_to_idx: dict,
        max_cache: int = 50,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.norm_stats = norm_stats
        self.clip_threshold = clip_threshold
        self.bad_channels = bad_channels or {}
        self.sub_to_idx = sub_to_idx
        self.max_cache = max_cache
        self.augment = augment
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []

    def __len__(self) -> int:
        return len(self.df)

    def _load_npy(self, path: str) -> np.ndarray:
        if path not in self._cache:
            if self.max_cache and len(self._cache) >= self.max_cache:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]
            self._cache[path] = np.load(path)
            self._cache_order.append(path)
        return self._cache[path]

    def _augment_inplace(self, eeg: np.ndarray) -> np.ndarray:
        """Stochastic augmentations on (122, 500) float32."""
        if np.random.rand() < 0.5:
            eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * 0.1
        if np.random.rand() < 0.3:
            shift = np.random.randint(-10, 11)
            if shift > 0:
                eeg[:, shift:] = eeg[:, :-shift]
                eeg[:, :shift] = 0.0
            elif shift < 0:
                eeg[:, :shift] = eeg[:, -shift:]
                eeg[:, shift:] = 0.0
        if np.random.rand() < 0.3:
            n_drop = max(1, int(0.05 * eeg.shape[0]))
            drop_idx = np.random.choice(eeg.shape[0], n_drop, replace=False)
            eeg[drop_idx, :] = 0.0
        if np.random.rand() < 0.5:
            eeg = eeg * np.float32(np.random.uniform(0.9, 1.1))
        if np.random.rand() < 0.5:
            T = eeg.shape[1]
            for _ in range(np.random.randint(1, 3)):
                w = np.random.randint(10, 51)
                s = np.random.randint(0, max(1, T - w))
                eeg[:, s:s + w] = 0.0
        if np.random.rand() < 0.5:
            spec = np.fft.rfft(eeg, axis=-1)
            F = spec.shape[-1]
            for _ in range(np.random.randint(1, 3)):
                w = np.random.randint(3, 21)
                s = np.random.randint(0, max(1, F - w))
                spec[:, s:s + w] = 0.0
            eeg = np.fft.irfft(spec, n=eeg.shape[1], axis=-1).astype(np.float32)
        return eeg

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        arr = self._load_npy(row["npy_path"])                       # (100, 500, 122)
        eeg = arr[int(row["trial_idx"])].astype(np.float32).T        # (122, 500)

        if self.clip_threshold is not None:
            np.clip(eeg, -self.clip_threshold, self.clip_threshold, out=eeg)

        sub = row["subject"]
        stats = self.norm_stats[sub]
        eeg = (eeg - stats["mean"][:, None]) / stats["std"][:, None]

        if sub in self.bad_channels and len(self.bad_channels[sub]) > 0:
            eeg[self.bad_channels[sub], :] = 0.0

        if self.augment:
            eeg = self._augment_inplace(eeg)

        return (
            torch.from_numpy(eeg),
            int(row["label"]),
            self.sub_to_idx[sub],
        )


def collate(batch):
    eegs, labels, sub_ids = zip(*batch)
    return (
        torch.stack(eegs),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(sub_ids, dtype=torch.long),
    )


def build_datasets(augment_train: bool = True):
    """Convenience: returns (train_ds, val_ds, test_ds, sub_to_idx, n_classes)."""
    art = load_artifacts()
    sub_to_idx = build_subject_index(art["train_df"], art["val_df"], art["test_df"])

    norm_per_subject = art["norm"]["per_subject"]
    clip_thr = art["norm"]["clip_threshold"]
    bad = art["bad"]

    common = dict(
        norm_stats=norm_per_subject,
        clip_threshold=clip_thr,
        bad_channels=bad,
        sub_to_idx=sub_to_idx,
    )
    train_ds = EEGDataset(art["train_df"], augment=augment_train, **common)
    val_ds = EEGDataset(art["val_df"], augment=False, **common)
    test_ds = EEGDataset(art["test_df"], augment=False, **common)

    cat_to_idx = art["labels"].get("cat_to_idx") if isinstance(art["labels"], dict) else None
    n_classes = len(cat_to_idx) if cat_to_idx else int(art["train_df"]["label"].max() + 1)
    return train_ds, val_ds, test_ds, sub_to_idx, n_classes


if __name__ == "__main__":
    tr, va, te, sub_to_idx, n_classes = build_datasets(augment_train=False)
    print(f"train {len(tr):>6} | val {len(va):>6} | test {len(te):>6}")
    print(f"subjects ({len(sub_to_idx)}):", sub_to_idx)
    print(f"n_classes: {n_classes}")
    x, y, s = tr[0]
    print("sample eeg", tuple(x.shape), x.dtype, "label", y, "subject_id", s)
    print("eeg stats after normalization: mean={:.3f} std={:.3f} min={:.3f} max={:.3f}".format(
        x.mean().item(), x.std().item(), x.min().item(), x.max().item()))
