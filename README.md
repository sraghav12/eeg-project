# EEG → Image Classification & CLIP Alignment

11-685 Group Project (S26). 20-class image-category decoding from scalp EEG, plus EEG↔image/caption retrieval and CLIP-aligned representation learning on top of a multi-subject EEG dataset.

Dataset: 20 ImageNet-style categories, 13 subjects, 122 channels, 500 timepoints @ 1 kHz. 26,000 trials total (15,600 train / 5,200 val / 5,200 test). Per-trial paired captions and category labels are released with the data; CLIP image embeddings are precomputed by [task2b/clip_utils.py](task2b/clip_utils.py).

Chance accuracy = 5%. EEGNet baseline = 9.46% test top-1.

## Repository layout

| Path | Stage | Contents |
|---|---|---|
| [eeg_data_pipeline.ipynb](eeg_data_pipeline.ipynb) | Pre-midterm | Raw → preprocessed pipeline. Bandpass, ICA, bad-channel detection, per-subject z-score, train/val/test split. Outputs the CSVs + pickles in [artifacts/](artifacts/) consumed by every downstream model. |
| [baseline_mlp.ipynb](baseline_mlp.ipynb) | Pre-midterm | MLP baseline. |
| [cnn_transformer.ipynb](cnn_transformer.ipynb) | Pre-midterm | Conv front-end + transformer encoder. |
| [EEGNET.ipynb](EEGNET.ipynb) | Pre-midterm | EEGNet (Lawhern et al.) — depthwise + separable conv. The 9.46% baseline. |
| [task2a_clip_retrieval.py](task2a_clip_retrieval.py) / [.ipynb](task2a_clip_retrieval.ipynb) | Pre-midterm | Task 2A: zero-shot EEG → caption retrieval using a frozen CLIP teacher. Outputs in [task2a_results/](task2a_results/). |
| [atm/](atm/) | Post-midterm | **ATM** encoder + classifier. Self-contained adaptation of *Visual Decoding and Reconstruction via EEG Embeddings* (Li et al., NeurIPS 2024) for our 13-subject / 20-class setup. iTransformer (channel-as-token) + ShallowNet conv + MLP projection + shared classifier head. |
| [enigma/](enigma/) | Post-midterm | **ENIGMA** encoder + classifier. Adaptation of Kneeland et al. (NeurIPS 2025 workshop). Per-subject `Linear(500,500)` adapter → shared SpatioTemporalCNN → MLP projector → linear head. Built as an architecture comparison vs ATM (same data, split, optimizer, schedule, augmentation). |
| [task2b/](task2b/) | Post-midterm | **EEG → CLIP alignment** (handout Sec. 5.5). Trainable projection head on top of the ATM encoder, contrastive against CLIP caption embeddings, with KD against image embeddings, category CE auxiliary loss, and full ablation grid over text-tower strategies (frozen / partial / LoRA / adapter). Full eval suite: instance + class-aware R@K, caption-level + class-aware MAP, BERTScore, CLIPScore, confusion matrix, per-subject accuracy. |
| [eval_results/](eval_results/) | Post-midterm | Output of [task2b/eval_task2b.py](task2b/eval_task2b.py): `eval_results.json`, `confusion_matrix.png`, `confusion_matrix.npy`. (`similarity_matrix.npy` and `ensemble_predictions.pt` are gitignored — too large.) |
| [artifacts/](artifacts/) | Pre-midterm | Pipeline outputs: `master_trials.csv`, `train.csv`/`val.csv`/`test.csv`, `norm_stats.pkl`, `bad_channels.pkl`, `label_mappings.pkl`, EDA plots. Consumed by every model. |
| [results/](results/) | Pre-midterm | Per-model artifacts (loss curves + confusion matrices + Kaggle CSVs) for the four pre-midterm models. |
| [task2a_results/](task2a_results/) | Pre-midterm | Task 2A retrieval evaluation outputs (R@K, MAP, BERTScore, CLIPScore plots + CSVs). |
| [environment.yml](environment.yml) | — | Conda env (`eeg_idl`). PyTorch ≥2.5 w/ CUDA 12.6, transformers, peft, bert_score, etc. |
| [config.py](config.py) | — | Shared paths (data root, artifact dir). |
| [captions.txt](captions.txt) | — | Per-trial captions (BIDS-resolved). |
| [midterm_report.tex](midterm_report.tex) / [final_report_notes.md](final_report_notes.md) | — | Report sources / notes. |

## Results

### Task 1: 20-way classification (test top-1)

| Model | Test acc | Best val | Best epoch | Trainable params | Wall time | Notes |
|---|---|---|---|---|---|---|
| Baseline MLP            | 6.28%   | —       | —  | —     | —      | from [baseline_mlp.ipynb](baseline_mlp.ipynb) |
| CNN-Transformer         | 8.77%   | —       | —  | —     | —      | from [cnn_transformer.ipynb](cnn_transformer.ipynb) |
| EEGNet                  | **9.46%** | —     | —  | —     | —      | reference baseline ([EEGNET.ipynb](EEGNET.ipynb)) |
| ENIGMA ([enigma/](enigma/)) | 9.35%  | 9.60%  | 7  | 5.90 M | 15.7 min | wider backbone (W=80, emb=8); overfits early |
| **ATM** ([atm/](atm/))  | **12.27%** | 11.83% | 27 | 6.82 M | 20.6 min | best single model — `atm_sharmar_20260419-0319` |
| ATM ensemble (3-seed)   | **12.77%** | —     | —  | —     | —      | average of 3 seeds, [atm/ensemble_eval.py](atm/ensemble_eval.py) |

### Task 2A: zero-shot retrieval (frozen CLIP teacher)

See [task2a_results/task2a_results.csv](task2a_results/task2a_results.csv). Highlights: R@1, R@5, MAP, BERTScore, CLIPScore distributions reported per category.

### Task 2B: EEG → CLIP alignment (single-seed `full_s42`, [eval_results/full_s42/eval_results.json](eval_results/full_s42/eval_results.json))

| Metric | Value |
|---|---|
| Test classification acc | 11.25% |
| Class-aware R@1 / R@3 / R@5 | 11.17% / 15.73% / 18.13% |
| Caption-level MAP | 0.0047 |
| Class-aware MAP | 0.1405 |
| Mean BERTScore | 0.872 (100% above 0.7) |
| CLIPScore (matched / retrieved / random) | 0.221 / 0.341 / 0.196 |

The 3-seed ensemble run was not completed before submission.

## Reproducing

### Environment
```bash
conda env create -f environment.yml
conda activate eeg_idl
```

### Pipeline
```bash
# 1. Build artifacts/ (one time)
jupyter nbconvert --to notebook --execute eeg_data_pipeline.ipynb

# 2. Pre-midterm baselines (optional, for comparison)
jupyter nbconvert --to notebook --execute baseline_mlp.ipynb
jupyter nbconvert --to notebook --execute cnn_transformer.ipynb
jupyter nbconvert --to notebook --execute EEGNET.ipynb

# 3. Task 1 — ATM (best single model)
cd atm
python sanity_check_atm.py     # must pass before full training
python train_atm.py            # 40 ep, ~20 min on V100
python ensemble_eval.py        # ensemble across multiple checkpoints

# 4. Task 1 — ENIGMA (architecture comparison)
cd ../enigma
python sanity_check_enigma.py
python train_enigma.py

# 5. Task 2B — CLIP alignment on top of ATM encoder
cd ../task2b
python clip_utils.py --captions-txt ../captions.txt \
                     --images-dir <path-to-images> \
                     --out-dir <clip-cache-dir>
python train_clip.py --encoder-ckpt ../atm/checkpoints/atm_sharmar_20260419-0319.pt \
                     --clip-cache <clip-cache-dir> \
                     --run-name full_s42 --seed 42
python eval_task2b.py --checkpoints clip_runs/full_s42.pt \
                      --clip-cache <clip-cache-dir> \
                      --out-dir ../eval_results/full_s42
```

### GPU
Bridges-2 (PSC). `interact -p GPU-shared -N 1 --gpus=v100:1 -t 4:00:00`.

## Per-folder READMEs

Each post-midterm directory has a self-contained README documenting model architecture, hyperparameter provenance, adaptations from the reference paper/repo, ablation results, and run commands:
- [atm/README.md](atm/README.md) — ATM (NeurIPS 2024)
- [enigma/README.md](enigma/README.md) — ENIGMA (NeurIPS 2025 workshop)
- [task2b/README.md](task2b/README.md) — CLIP alignment, full handout coverage table, ablation grid

## Excluded from this repository

- **HW4P2** (`4p2/`) — separate homework, not part of the project.
- **Reference repos** (`atm_reference/`, `enigma_reference/`) — third-party clones with their own `.git` (kept locally as read-only references; both MIT/Apache licensed). See attribution sections in [atm/README.md](atm/README.md) and [enigma/README.md](enigma/README.md).
- **Model checkpoints** (`*.pt`) — gitignored. Only the JSON sidecars (full hyperparameters + per-epoch history + best metrics) are committed under each model's `checkpoints/` directory.
- **Large eval artifacts** — `similarity_matrix.npy` (~84 MB), `ensemble_predictions.pt`, `clip_cache/`, `wandb/`.

## Attribution

- ATM model code adapted from https://github.com/ncclab-sustech/EEG_Image_decode (MIT, © DongyangLi 2023). See [atm/model.py](atm/model.py) inline citations and [atm/README.md](atm/README.md).
- ENIGMA model code adapted from https://github.com/Alljoined/ENIGMA (Apache-2.0). See [enigma/model.py](enigma/model.py) inline citations and [enigma/README.md](enigma/README.md).
- CLIP teacher: `openai/clip-vit-large-patch14` via HuggingFace `transformers`.
