# ENIGMA classifier (sharmar/enigma/)

Adapted from Kneeland et al., *ENIGMA* (NeurIPS 2025 workshop, arXiv:2602.10361).
Reference repo (Apache-licensed): https://github.com/Alljoined/ENIGMA.

Built as an architecture comparison against [`../atm/`](../atm/) (ATM paper,
NeurIPS 2024). Both use the same dataset, preprocessing, split, augmentation,
optimizer, schedule, and head strategy so differences isolate the encoder.

## File map

| File | Role |
|---|---|
| [model.py](model.py) | Self-contained ENIGMA encoder + classifier. Per-subject Linear → SpatioTemporalCNN → MLP projector → Linear head. `build_enigma_classifier(...)` factory. |
| [dataset.py](dataset.py) | Loads ATM's `dataset.py` by absolute path under the name `atm_dataset` (using `importlib`) and re-exports the public symbols — identical preprocessing to ATM. |
| [train_enigma.py](train_enigma.py) | Training loop mirroring `atm/train_atm.py`. 40 epochs, bs=64, AdamW lr=1e-4 wd=0.01, warmup 500 + cosine, grad-clip 1.0. Best-by-val-acc checkpoint + sidecar JSON. |
| [sanity_check_enigma.py](sanity_check_enigma.py) | Overfit 100 trials from one subject with zero dropout. Asserts train acc ≥ 0.8. |
| `checkpoints/` | Output dir for `enigma_sharmar_<timestamp>.pt` + `.json`. |

## Architecture

```
input (B, 122, 500)
  │
  ├─ subject_wise_linear[sid]   : Linear(500, 500)      per-subject adapter
  │
  ├─ SpatioTemporalCNN (shared)
  │     Conv2d(1 → W=80, (1,5))
  │     AvgPool2d((1,17), stride (1,5))
  │     BN + ELU
  │     Conv2d(W → W, (122, 1))                         spatial conv
  │     BN + ELU + Dropout(0.5)
  │     Conv2d(W → emb=8, 1×1)                          1×1 projection
  │     flatten  → hidden_dim = 768
  │
  ├─ MLPProjector
  │     Linear(768 → 1024)
  │     Residual(GELU + Linear + Dropout(0.5))
  │     LayerNorm
  │
  └─ Linear(1024 → 20)                                   shared classifier
```

**Widths scaled up from paper defaults** (W=40→80, emb=4→8). Paper tuned for
63 ch / 250 tp / 10 subjects; we have 122 ch / 500 tp / 13 subjects, so the
shared backbone gets more capacity to match the bigger per-subject adapter.

## Adaptations from reference (`enigma_reference/source/models.py`)

| # | Change | Reason |
|---|---|---|
| 1 | Subject routing: `nn.ModuleList` by int id instead of `nn.ModuleDict` by string | Matches the `(eeg, label, subject_id)` contract of our EEGDataset |
| 2 | `einops.Rearrange` replaced with `permute + view` | Avoid adding einops dep |
| 3 | Added shared `Linear(embed_dim, num_classes)` classifier head | Reference is retrieval-only (MSE + CLIP contrastive); we need 20-way CE |
| 4 | Widths: W=40→80, emb=4→8 | Larger input (122×500 vs 63×250) and more subjects (13 vs 10) |
| 5 | Hidden dim probed lazily | Avoid hardcoding the flatten dim |
| 6 | No "latent alignment layer" | Reference repo does not implement it either, despite the paper describing it |

## Training configuration

| Parameter | Value | Source |
|---|---|---|
| optimizer | AdamW | paper + brief |
| lr | 1e-4 | brief |
| weight decay | 0.01 | brief |
| warmup steps | 500 | brief |
| schedule | linear warmup → cosine decay | brief |
| grad clip | 1.0 | brief |
| batch size | 64 | paper default |
| epochs | 40 | paper default |
| dropout (proj residual) | 0.5 | paper default |
| dropout (conv) | 0.5 | paper default |
| embed_dim | 1024 | CLIP ViT-H-14 dim |
| num_subjects | 13 | our setup |
| augmentation | additive noise / time shift / channel drop / amplitude scale / time mask / freq mask | shared dataset (`atm/dataset.py`) |
| normalization | per-subject per-channel z-score → bad-channel zero | shared dataset |

## Results (run `enigma_sharmar_20260424-0158`)

| Metric | Value |
|---|---|
| Best val accuracy | **0.0960** (epoch 7) |
| Test accuracy | **0.0935** |
| Final train accuracy | 0.7784 (epoch 40) |
| Train ↔ val gap | **~68 pp** (heavy overfit) |
| Trainable params | 5.90 M |
| Hidden dim after CNN | 768 |
| Wall time | 15.7 min (40 epochs, ~23 s/epoch on GPU) |
| Sanity-check overfit | 100% train acc on 100 trials (passed by epoch 17) |

**Observation:** val loss starts rising immediately after epoch 7 while train
accuracy keeps climbing → classic overfit. Best checkpoint fires very early.
The model learns the train set but doesn't generalise under the current
regularisation regime.

### Possible follow-ups to close the generalisation gap

- Raise `dropout_proj` from 0.5 → 0.7 or add weight decay only on the
  subject-wise Linear.
- Enable `--mixup 0.2 --label-smoothing 0.1` (both wired into
  `train_enigma.py`).
- Try `--per-subject-head` (ablation).
- Shrink backbone back toward paper defaults (`--backbone-width 40 --emb-size 4`)
  now that we know the wider version overfits.
- Early-stop at epoch 7-10 to avoid wasting GPU time on the post-peak tail.

## Head-to-head comparison table

Fill alongside results from sibling architectures (same preprocessing + split):

| Model | Params | Best val acc | Test acc | Epoch of best | Wall time | Key design |
|---|---|---|---|---|---|---|
| EEGNet baseline | — | — | 0.0946 (paper) | — | — | Depthwise + separable conv |
| ATM (`atm/`) | ~6.5 M | *fill from `atm/checkpoints/*.json`* | *fill* | *fill* | *fill* | iTransformer (channel-as-token) + ShallowNet conv + MLP proj |
| **ENIGMA (`enigma/`)** | **5.90 M** | **0.0960** | **0.0935** | **7** | **15.7 min** | Per-subject Linear + shared SpatioTemporalCNN + MLP proj |

Chance = 5 % (20 classes). EEGNet baseline from the shared project = 9.46 %.

## Commands

```bash
# GPU env: conda activate eeg_idl  (pytorch matching CUDA 12.6)
cd /ocean/projects/cis260086p/sharmar/enigma

# Sanity check (~2 min on GPU) — must pass before full run
python sanity_check_enigma.py

# Full training (~16 min on GPU)
python train_enigma.py

# Regularisation variants
python train_enigma.py --mixup 0.2 --label-smoothing 0.1
python train_enigma.py --dropout-proj 0.7
python train_enigma.py --per-subject-head
python train_enigma.py --backbone-width 40 --emb-size 4    # paper-default capacity

# With wandb logging
WANDB_API_KEY=xxx python train_enigma.py
```

## Attribution

Portions of `model.py` (SpatioTemporalCNN, ResidualAdd, MLPProjector, ENIGMA
encoder) were adapted from the reference repo under the Apache-2.0 License;
specific line citations appear inline in `model.py`. `dataset.py` re-exports
ATM's dataset module (MIT License, © 2023 DongyangLi).
