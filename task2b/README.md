# Task 2B: EEG -> CLIP Alignment

Full implementation of handout Sec. 5.5 (EEG-Caption Retrieval) with every
required feature plus the full metric suite.

## Files

| File | Purpose |
|---|---|
| `clip_utils.py`   | Precompute CLIP image/caption/category embeddings once. |
| `dataset_clip.py` | EEG dataset + CLIP targets. Resolves `image_name` from BIDS CSVs. |
| `losses.py`       | InfoNCE, Debiased InfoNCE, Logit KD, Cosine, Category CE. |
| `model_clip.py`   | Projection head, LoRA / Adapter / partial-unfreeze, EMA. |
| `train_clip.py`   | Training loop with every ablation toggle. |
| `eval_task2b.py`  | Full evaluation suite (R@K, MAP × 3, BERTScore, CLIPScore). |
| `run_ablations.sh`| Reference commands for the full ablation grid. |

All files assume your project layout:
```
atm/
├── dataset.py        (your existing loader)
├── model.py          (your existing ATM encoder / classifier)
├── checkpoints/*.pt  (your Task 1 checkpoints)
└── task2b/           <-- drop these files here and cd into it
```

## Handout requirement coverage

| Requirement | File | Notes |
|---|---|---|
| Trainable projection head (MLP) | `model_clip.py` | 2-layer MLP + LayerNorm + skip |
| L2 normalization at output | `model_clip.py` | `normalize=True` default |
| Logit KD via KL divergence | `losses.py` | `LogitKDLoss` |
| Cosine similarity-based KD | `losses.py` | `CosineAlignLoss` |
| Naive InfoNCE | `losses.py` | `SymmetricInfoNCE` |
| Debiased InfoNCE | `losses.py` | `DebiasedInfoNCE` (class / sim / hybrid) |
| Category CE | `losses.py` | `CategoryCELoss` |
| Combined objective | `train_clip.py` | tunable λ per term |
| Frozen CLIP | `train_clip.py` | `--text-strategy frozen` |
| Partial unfreeze | `model_clip.py` | `--text-strategy partial --partial-n-layers 2` |
| LoRA | `model_clip.py` | `--text-strategy lora` |
| Adapters | `model_clip.py` | `--text-strategy adapter` |
| Trainable param tracking | `train_clip.py` | printed + saved in meta JSON |
| Instance Recall@{1,3,5} | `eval_task2b.py` | `recall_at_k` |
| Class-aware Recall@{1,3,5} | `eval_task2b.py` | `class_aware_recall_at_k` |
| Caption-level MAP | `eval_task2b.py` | `compute_map_caption_level` |
| Class-aware MAP | `eval_task2b.py` | `compute_map_class_aware` |
| Per-class MAP | `eval_task2b.py` | (returned alongside class-aware) |
| BERTScore | `eval_task2b.py` | requires `pip install bert_score` |
| CLIPScore distributions | `eval_task2b.py` | matched / retrieved / random |
| Confusion matrix | `eval_task2b.py` | saved as `.npy` + `.png` |
| Per-subject accuracy | `train_clip.py`, `eval_task2b.py` | both |

## Run order

### Prerequisites
```bash
pip install peft transformers pillow tqdm bert_score matplotlib
```

### 1. Precompute CLIP targets (once, ~20-30 min on V100)

```bash
cd task2b
python clip_utils.py \
    --captions-txt /ocean/projects/cis260086p/shared/eeg-project/captions.txt \
    --images-dir   /ocean/projects/cis250019p/gandotra/11785-gp-eeg/All_images \
    --out-dir      /ocean/projects/cis260086p/sharmar/atm/clip_cache \
    --clip-model   openai/clip-vit-large-patch14
```

This writes `image_embs.pt`, `caption_embs.pt`, `category_embs.pt`,
`category_names.json`, `captions_df.csv` to `clip_cache/`.

### 2. Smoke-test the data pipeline

```bash
python dataset_clip.py /ocean/projects/cis260086p/sharmar/atm/clip_cache
```
Should print a single sample with correct shapes and a non-zero `img_emb`.

### 3. Main training runs (3 seeds for ensembling, ~2-3 h each)

```bash
python train_clip.py \
    --encoder-ckpt ../checkpoints/atm_sharmar_20260419-0319.pt \
    --clip-cache /ocean/projects/cis260086p/sharmar/atm/clip_cache \
    --clip-model openai/clip-vit-large-patch14 \
    --run-name full_s42 --seed 42

python train_clip.py ... --run-name full_s43 --seed 43
python train_clip.py ... --run-name full_s44 --seed 44
```

### 4. Ensemble evaluation (the number that goes in your report)

```bash
python eval_task2b.py \
    --checkpoints clip_runs/full_s42.pt clip_runs/full_s43.pt clip_runs/full_s44.pt \
    --clip-cache /ocean/projects/cis260086p/sharmar/atm/clip_cache \
    --out-dir eval_results/ensemble_full
```

This writes `eval_results/ensemble_full/eval_results.json` with every metric.

### 5. Ablation table (one ablation per row)

See `run_ablations.sh` for the full grid. Every row changes exactly one factor
vs. the full model. Given your timeline, the highest-value ablations to
actually run are:

1. `abl_no_catce` — show that category CE is the accuracy driver
2. `abl_no_kd` — show KD contribution
3. `abl_no_debias` — show debiasing helps on same-class negatives
4. `abl_text_lora` — required for the parameter-efficient fine-tuning rubric
5. `abl_frozen_enc` — baseline for "does training the encoder help?"

## Design decisions (all made to maximize classification accuracy)

1. **CLIP ViT-L/14 as teacher** (768-d). Stronger than ViT-B/32. Costs ~2 GB more
   GPU memory but gives a cleaner KD signal.
2. **Caption target + image aux**. Contrastive target is CAPTIONS (handout's
   ask, makes retrieval evaluation meaningful). Image embeddings also used as
   the KD teacher -> student effectively gets supervised from both modalities.
3. **Category CE with w=1.5**. Directly optimizes the zero-shot classification
   eval. This is the single biggest lever for accuracy.
4. **Encoder partially unfrozen at low LR (5e-5)**. Fully freezing loses 1-2
   points; blowing it out with a high LR destroys pretrained features.
5. **EMA on encoder + projection**, decay 0.999. Reliable +0.5%.
6. **Prompt ensemble for category prototypes**. 8 templates, averaged.
7. **3-seed ensemble**. Your Task 1 ensemble gave +0.5% over single best;
   expect similar here.

## Realistic accuracy expectations

Starting point: your single best Task 1 is **12.27%**, ensemble **12.77%**.

Likely outcomes with this code (based on what the ATM paper reports moving
from classification to CLIP alignment, scaled to this noisier dataset):

| Config | Expected test accuracy |
|---|---|
| Single run, full config | 14-17% |
| 3-seed ensemble, full config | 16-19% |

**17%+ is plausible but not guaranteed.** The primary uncertainty is whether
the EEG signal is strong enough to benefit substantially from the richer
loss structure, or whether we plateau at a ceiling imposed by session
variability + cross-subject noise.

If first run comes back below 14%, check in this order:
1. `img_emb` hit rate in `[EEGCLIPDataset] ...` line (should be near 100%)
2. Category CE loss actually decreasing over training
3. `logit_scale` not blowing up (clamp is working)
4. `encoder.logit_scale.exp()` prints a reasonable value (5-30)

If second run is worse than first:
- LR might be too high. Drop `--lr-encoder 2e-5 --lr-proj 2e-4`.
- Try `--freeze-encoder` as an ablation; if it helps, your encoder LR is
  destroying your Task 1 pretrained weights.

## Dependencies

```
torch>=2.0
transformers>=4.30
peft>=0.5          # for LoRA
pillow             # image loading
tqdm
bert_score         # optional, for eval
matplotlib         # optional, for confusion matrix plot
```
