# ATM classifier (sharmar/)

Adaptation of Li et al., "Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion" (NeurIPS 2024, arXiv:2403.07721), for the 11-685 team project.

Reference code (MIT-licensed): https://github.com/ncclab-sustech/EEG_Image_decode — mirrored under `../atm_reference/` (read-only).

## File map

| File | Role |
|---|---|
| [model.py](model.py) | Self-contained ATM encoder + classifier. `build_atm_classifier(num_subjects, num_classes, embed_dim)` factory. No braindecode / reformer_pytorch / einops deps. |
| [dataset.py](dataset.py) | Private dataset reading `shared/eeg-project/artifacts/*.csv` + `norm_stats.pkl` + `bad_channels.pkl`. Reproduces EEGNet baseline preprocessing (clip → subject-channel z-score → bad-channel mask → optional augmentation). |
| [train_atm.py](train_atm.py) | Full training loop (40 epochs, bs=64, AdamW lr=1e-4, wd=0.01, warmup 500 + cosine, grad-clip 1.0). Saves best-by-val-acc ckpt + sidecar JSON. |
| [sanity_check_atm.py](sanity_check_atm.py) | Overfit test on 100 trials from one subject. Asserts train acc ≥ 0.8 by epoch 50. |
| `checkpoints/` | Output directory for ckpts. |

## Adaptation notes

| Dimension | Paper | Ours |
|---|---|---|
| Channels | 63 | 122 |
| Timepoints | 250 | 500 |
| Classes | 1654 concepts / retrieval | 20 / classification |
| Subjects | 10 | 13 |
| Sampling rate | 250 Hz (pre-downsampled) | 1000 Hz (no downsampling — preserves 500 timepoints) |
| Transformer d_model | 250 | 500 (tied to seq_len, same design as paper) |
| Transformer d_ff | 256 | 512 (kept paper's ~1× d_model ratio) |
| Loss | CLIP contrastive (img+text) | Cross-entropy on 20 logits |
| Head | no classifier; projects to CLIP space | single shared `Linear(1024 → 20)` on top of encoder |

## How per-subject handling works

The reference model prepends one learned `SubjectEmbedding` token to the channel-token sequence before transformer attention ([Embed.py:109-121](../atm_reference/models/subject_layers/Embed.py#L109-L121) → used at [ATMS_retrieval.py:176](../atm_reference/Retrieval/ATMS_retrieval.py#L176)). We keep this intact (13 subject slots) and **use a single shared classifier head** on top of the encoder. The subject token gives the transformer a per-subject bias without paying for 13 separate heads — this is the user-selected head strategy for the project.

## Hyperparameter provenance

| Parameter | Value | Source |
|---|---|---|
| optimizer | AdamW | paper ([ATMS_retrieval.py:548](../atm_reference/Retrieval/ATMS_retrieval.py#L548)) |
| lr | 1e-4 | brief (paper used 3e-4) |
| weight decay | 0.01 | brief (paper had none) |
| warmup steps | 500 | brief |
| schedule | linear warmup → cosine decay | brief |
| grad clip | 1.0 | brief (paper had none) |
| batch size | 64 | paper ([ATMS_retrieval.py:526](../atm_reference/Retrieval/ATMS_retrieval.py#L526)) |
| epochs | 40 | paper ([ATMS_retrieval.py:525](../atm_reference/Retrieval/ATMS_retrieval.py#L525)) |
| dropout (transformer) | 0.25 | paper ([ATMS_retrieval.py:53](../atm_reference/Retrieval/ATMS_retrieval.py#L53)) |
| dropout (conv+proj) | 0.5 | paper ([ATMS_retrieval.py:109](../atm_reference/Retrieval/ATMS_retrieval.py#L109)) |
| n_heads | 4 | paper ([ATMS_retrieval.py:55](../atm_reference/Retrieval/ATMS_retrieval.py#L55)) |
| e_layers | 1 | paper ([ATMS_retrieval.py:56](../atm_reference/Retrieval/ATMS_retrieval.py#L56)) |
| embed_dim | 1024 | paper (CLIP ViT-H-14 dim) |

## Assumptions made (flag if incorrect)

1. **CLIP variant for embed_dim.** Paper uses open-clip ViT-H-14 with 1024-d features ([eegdatasets_leaveone.py:18-21](../atm_reference/Retrieval/eegdatasets_leaveone.py#L18-L21)). Kept 1024-d so the encoder slots directly into Shrirang's Task 2B without resizing.
2. **Transformer d_model == seq_len.** Paper's Config has `d_model=250, seq_len=250` ([ATMS_retrieval.py:47,50](../atm_reference/Retrieval/ATMS_retrieval.py#L47-L50)). We preserve the tie (d_model=seq_len=500). If this is too big, drop to 250 with a Linear(500→250) projection — keeps transformer param count low but loses the clean design.
3. **d_ff scaled from 256 → 512.** Paper's d_ff is slightly above d_model; we preserve the ratio rather than copy the literal 256 (which would make d_ff < d_model for us).
4. **Subject ID space = 13.** Subjects in the CSVs are non-contiguous (`sub-02, sub-03, sub-05, sub-09, sub-14, sub-15, sub-17, sub-19, sub-20, sub-23, sub-24, sub-28, sub-29`). `dataset.build_subject_index` maps them to dense `0..12`. If a new subject appears in a future CSV, they must be re-indexed.
5. **No trial averaging.** Paper train data has 4 repetitions per image that they stack; our dataset has 1 trial per image presentation. No change needed.
6. **Preprocessing matches the EEGNet baseline.** Clip ±800, per-subject per-channel z-score, zero out bad channels. The paper applies no per-channel normalization at runtime (their data arrives pre-normalized), so this is our adaptation.

## Commands

Sanity check (should take ~5-15 min on 1 GPU):
```bash
cd /ocean/projects/cis260086p/sharmar/atm
python sanity_check_atm.py                         # default: sub-02, 100 trials, 50 epochs
python sanity_check_atm.py --subject sub-05        # different subject
```

Full training (40 epochs, bs=64):
```bash
cd /ocean/projects/cis260086p/sharmar/atm
python train_atm.py                                # defaults match the brief
python train_atm.py --batch-size 128 --epochs 30   # override if you want brief-spec exactly
WANDB_API_KEY=xxx python train_atm.py              # enables wandb logging
```

## Dependencies

Everything needed is already in `shared/eeg-project/environment.yml` *or* doesn't need to be added:

| Package | Status |
|---|---|
| torch ≥ 2.5 | already in environment.yml |
| numpy, pandas | already in environment.yml |
| wandb (optional) | already in environment.yml |
| einops | **removed** — was the only dep that could have been new; avoided via a 4-line `_Rearrange` in `model.py` |
| braindecode | **not needed** — reference repo imports it but the ATMS model itself doesn't use it |
| reformer_pytorch | **not needed** — only ReformerLayer uses it and we don't |
| open_clip / clip | **not needed** for classification — would only be for Task 2B |

No new dependencies required.

## Things to verify manually before launching full training

1. **GPU node.** Sanity check + full training require CUDA. User's `eeg_idl` conda env on login shell reports `cuda=False`. Submit via SLURM (`interact -p GPU-shared -N 1 --gpus=v100:1 -t 2:00:00` or similar) before `python train_atm.py`.
2. **Subject-index coverage.** Running `python dataset.py` should print 13 subjects. Confirmed at build time: `{'sub-02': 0, ..., 'sub-29': 12}`.
3. **Checkpoint dir.** `mkdir -p /ocean/projects/cis260086p/sharmar/atm/checkpoints` (already done).
4. **Expected numbers.** EEGNet baseline = 9.46% test. ATM's per-subject THINGS-EEG v200 top-1 is ~50% (paper Table 1) — but that is on pretrained CLIP targets across 1654 concepts, not 20-way classification, so direct transfer is not expected. A reasonable first-run target: **beat 9.46% test top-1** (chance = 5%).
5. **wandb.** `wandb login` first if you want the run logged; otherwise `train_atm.py` falls back to stdout silently.
6. **Sanity check must pass.** If it fails, do NOT launch full training; check:
   - Label remap: are labels in `[0, 20)` for the 100 chosen trials?
   - Normalization: `dataset.py __main__` prints `min≈-4, max≈3` after z-score. If values blow up, there's a data issue.
   - Shapes: `model.py __main__` must print `logits (4, 20) emb (4, 1024)`.

## Expected epoch time

Rough GPU-time estimate for one epoch at bs=64, ~15,600 train trials → 243 steps/epoch:
- On one V100/A100, similar EEG transformers in the paper run at ~5-10 min/epoch for bs=64, 250 timepoints. Our input has 2× timepoints and ~2× channels, so estimate **~10-20 min/epoch**, i.e. **7-13 h for 40 epochs**. Verify by timing the first epoch before letting the full 40 run.
- Data loading is the likely bottleneck (.npy files from `/ocean/projects/cis250019p/...`). Consider preloading or upping `--num-workers`.

## Attribution

Portions of `model.py` (iTransformer, DataEmbedding, SubjectEmbedding, FullAttention, AttentionLayer, EncoderLayer, Encoder, PatchEmbedding, Proj_eeg) were adapted from the reference repo under MIT License:
```
MIT License
Copyright (c) 2023 DongyangLi
(permission/disclaimer as in atm_reference/LICENSE)
```
Specific line citations appear inline as module-level comments.
