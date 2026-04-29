# Decoding What You See From Your Scalp

**11-685 (S26) Group Project — single-trial EEG → image-category decoding & EEG → CLIP alignment**

What can a 122-channel scalp recording tell us about what someone is *currently looking at*? In this project we attack that question two ways: (1) classify the trial into one of 20 ImageNet-style categories, and (2) align the EEG into CLIP space so any caption or image can be retrieved by neural activity alone. Chance is **5%**. EEGNet — the de-facto strong baseline — gets **9.46%**. Our best system reaches **12.67%** (a 2-seed ATM ensemble, +3.21 pt over EEGNet, **2.53× chance**), and the CLIP-aligned variant produces caption embeddings whose CLIPScore against the actually-shown image is reliably above the random-pair baseline.

The interesting result isn't the headline number. It's *why* a 6.57M-parameter transformer beats EEGNet's 36K, why a 5.90M-parameter CNN with the **same downstream layers** as that transformer flatlines, why a 1.25M-parameter CNN-Transformer hybrid never escaped chance, and why CLIP retrieval at the *category* level works almost perfectly even though single-trial classification stays under 13%. Read on.

---

## 1. The data

| Property | Value |
|---|---|
| Subjects | 13 (sub-02, 03, 05, 09, 14, 15, 17, 19, 20, 23, 24, 28, 29) |
| Channels | 122 scalp electrodes |
| Sampling rate | 1000 Hz, **not** downsampled (preserves 500 timepoints / trial vs paper's 250) |
| Trials | 26,000 = 5 sessions × 4 runs × 100 trials |
| Classes | 20 ImageNet categories |
| Split | ses-1–3 train (15,600) / ses-4 val (5,200) / ses-5 test (5,200) — **strict session split**, no temporal leakage |
| Companion data | per-trial captions and a precomputed CLIP image embedding for every stimulus |

Preprocessing — built once in [eeg_data_pipeline.ipynb](eeg_data_pipeline.ipynb) and consumed by every model — is intentionally simple and identical across architectures so that downstream comparisons are clean: clip ±800 μV → per-subject per-channel z-score (stats from train only) → bad-channel zero-mask. After step 2, each trial sits at mean ≈ 0, std ≈ 1, range roughly [−4, 3].

---

## 2. Pre-midterm: what doesn't work, and why that matters

| Model | Params | Test acc | What we learned |
|---|---|---|---|
| Baseline MLP                    | 757 K   | 8.73%  | Even a flat MLP on the raw signal beats two of the next three models. EEG has *some* trivially decodable category content. |
| **CNN-Transformer**             | 1.25 M  | **5.00%** | Stuck at chance the entire 40 epochs. A useful negative result — capacity without the right inductive bias *cannot* learn this dataset. |
| **EEGNet** (Lawhern et al.)     | 36 K    | **9.46%** | Tiny depthwise + separable convolutions. The strongest pre-midterm model. The bar everything else has to clear. |
| Task 2A: zero-shot retrieval (frozen CLIP) | — | class-aware R@1 ≈ 97% | The **category-level** signal is essentially solvable when you pool information across many trials of the same class. So the difficulty is single-trial separability, not "is the signal there." |

These three results frame the rest of the project. EEGNet is a real ceiling (not chance); CNN-Transformer at 1.25M says raw capacity isn't enough; Task 2A retrieval says the category-level information *exists*, so a stronger encoder ought to be able to extract more of it per trial.

Code: [baseline_mlp.ipynb](baseline_mlp.ipynb), [cnn_transformer.ipynb](cnn_transformer.ipynb), [EEGNET.ipynb](EEGNET.ipynb), [task2a_clip_retrieval.py](task2a_clip_retrieval.py). Outputs: [results/](results/), [task2a_results/](task2a_results/).

---

## 3. Post-midterm: a controlled architecture comparison

The post-midterm half builds **two** stronger encoders that share the same data, split, augmentation, optimizer, schedule, head strategy, and early-stopping criterion. The only thing that changes between them is the encoder itself. That makes the comparison interpretable.

### 3.1 [`atm/`](atm/) — Adaptive Thinking Mapper (Li et al., NeurIPS 2024)

Three-stage encoder, end-to-end:

```
input (B, 122, 500)
  │
  ├── Stage A: iTransformer + per-subject token
  │     • Linear(500 → 500) per channel, sinusoidal positional embedding
  │     • learned subject-token s_i ∈ ℝ^500 PREPENDED to the channel sequence  ← key idea
  │     • 1 transformer encoder (4 heads, d_model=500, d_ff=512, drop 0.25)
  │     • drop subject token, back to (122, 500)
  │
  ├── Stage B: ShallowNet conv
  │     • Conv2d(1→40, 1×25)  →  AvgPool(1×51, s=5)  →  BN + ELU
  │     • Conv2d(40→40, 122×1) → BN + ELU + Drop(0.5)         ← spatial conv across all channels
  │     • Conv2d(40→40, 1×1)  → flatten to 3440-d
  │
  └── Stage C: residual MLP projector
        • Linear(3440 → 1024) → Residual(GELU + Linear + Drop) → LayerNorm
        • → 1024-d embedding (CLIP ViT-H-14 dim — reused by Task 2B unchanged)

  head: Linear(1024 → 20)  [or per-subject variant, +0.25M]
```

The subject token is the load-bearing idea: the transformer learns one bias vector per subject and shares **all** other parameters across subjects. A per-subject *bias* without a per-subject *function*. We kept the entire architecture (with shape adaptations to 122 ch / 500 tp) and replaced only the CLIP contrastive loss with cross-entropy + a 1024 → 20 linear head.

Notable engineering: **no new dependencies**. The reference repo pulls in `braindecode`, `einops`, and `reformer_pytorch`; we re-implemented the necessary pieces inline (`_Rearrange`, sinusoidal pos enc, attention) so the project's `environment.yml` stays clean. See [atm/model.py](atm/model.py).

### 3.2 [`enigma/`](enigma/) — ENIGMA (Kneeland et al., NeurIPS 2025 workshop)

Same Stage B and Stage C as ATM. The single thing that changes is **how subject identity enters the model:**

```
input (B, 122, 500)
  │
  ├── per-subject Linear(500, 500) along the time axis     ← 250K params PER subject × 13 = 3.25M
  │   (i.e. a separate FUNCTION per subject, not a bias)
  │
  ├── shared SpatioTemporalCNN (W=80, emb=8)
  ├── shared MLP projector
  └── shared Linear(1024 → 20)
```

That's the entire isolated variable: **transformer subject token** (parameters shared across subjects, conditioning is a learned bias) **vs per-subject affine adapter** (separate function per subject, 55% of all trainable parameters spent on subject identity).

### 3.3 The ablation that matters most

The ATM ablation is six rows, each adding exactly one factor over the previous:

| # | Δ vs previous | Val % | Test % | Peak ep | Train @40 | Insight |
|---|---|---:|---:|---:|---:|---|
| 1 | Paper defaults, CE only, shared head | 11.29 | 11.58 | 18 | ~44% | Already +2.12 pt over EEGNet. Heavy overfit; val loss ↑ from epoch 8. |
| 2 | + MixUp 0.2, drop_proj 0.5→0.6, wd 0.01→0.05 | 11.63 | 12.00 | 18 | ~21% | Closes train–val gap from 33 pp to 10 pp. +0.42 test. |
| 3 | + Per-subject linear head (+0.25M) | 11.94 | 11.71 | 32 | ~24% | Val ↑, test ↓ — within seed-level noise. |
| 4 | + Time mask + FFT freq mask + label smoothing 0.1 | 12.27 | 12.04 | 31 | ~21% | **Best single-model config.** |
| 5 | Config #4, seed 43 | 11.83 | **12.27** | 27 | ~21% | Seed variance ±0.2 pt — real. |
| 6 | Config #4, seed 44 (DataLoader seeding bug) | 11.81 | 12.08 | 33 | ~21% | Correlated draws despite different `seed`. |
| **7** | **Ensemble {seed 42, seed 43}** | — | **12.67** | — | — | **+0.40** over best single. **Headline number.** |
| 8 | Ensemble {42, 43, 44} | — | 12.56 | — | — | Adding the correlated seed *regresses* the ensemble. Diversity matters. |

Per-checkpoint history is committed under [atm/checkpoints/*.json](atm/checkpoints/) — every JSON has the full per-epoch train/val curve, hyperparameter snapshot, and timing.

### 3.4 The dramatic failure: ENIGMA overfits like a textbook example

```
                      ENIGMA training curve (40 epochs, no early stopping)

  train acc:  5%  →  22%  →  47%  →  71%  →  78%
   val  acc:  5%  →  10%  →   9%  →   9%  →   9%
   val loss: 3.07 → 3.14 → 3.69 → 4.28 → 4.35   ← rises monotonically from epoch 8
                          ↑
                    best val (epoch 7)
```

Final train–val gap: **~68 percentage points**. Sanity check (100-trial overfit, dropout = 0): hits 100% train accuracy by epoch 17 — the architecture *can* fit. The failure is **generalization**, not optimization. And because ENIGMA shares Stage B + Stage C with ATM, the isolated cause is the per-subject affine adapter **memorizing subject-specific patterns** that don't transfer to held-out sessions. The transformer subject token, which only contributes a learned bias rather than a separate per-subject function, doesn't have the capacity to do this kind of memorization.

Full ENIGMA epoch history: [enigma/checkpoints/enigma_sharmar_20260424-0158.json](enigma/checkpoints/enigma_sharmar_20260424-0158.json).

### 3.5 What this all says, distilled

| Lesson | Evidence |
|---|---|
| **Inductive bias beats raw capacity at this data scale.** | EEGNet (36K) > CNN-Transformer (1.25M); ATM (6.57M) > ENIGMA (5.90M). |
| **Subject conditioning matters more than parameter count.** | ATM and ENIGMA share Stage B+C. The transformer-token vs affine-adapter swap alone accounts for the +3 pt gap. |
| **Regularization closes the train–val gap but doesn't raise the ceiling.** | 33 pp → 10 pp gap = only +0.42 pt test. Bottleneck is representational, not optimization. |
| **Ensembles are free — *if* the seeds are diverse.** | Two well-shuffled seeds: +0.40 pt. Adding a correlated seed (DataLoader bug): −0.11 pt. |
| **The category signal exists.** | Frozen-CLIP zero-shot retrieval (Task 2A) hits class-aware R@1 ≈ 97%. |
| **Single-trial separability is the genuinely hard part.** | And our 12.67% is ~3.5× chance, but still well below the population-level retrieval ceiling. |

---

## 4. Task 2B — building a CLIP-aligned EEG embedding on top of ATM

[`task2b/`](task2b/) takes the 1024-d ATM encoder embedding and fine-tunes a projection head against CLIP, with a multi-objective loss and a full ablation grid over how the text tower is updated.

**Why this works at all:** the ATM encoder was *designed* to project to CLIP's 1024-d space (same `embed_dim` as ViT-H-14). So the encoder slots into Task 2B without resizing. The Task 2B model is the same encoder + a trainable projection head + an optional update strategy on the CLIP text tower.

**Loss (combined objective):**
- **Symmetric InfoNCE** between EEG embedding and CLIP **caption** embedding — this is the contrastive target the handout asks for.
- **Logit-KD KL-divergence** between EEG-CLIP and image-CLIP similarity matrices — gives a richer image-side signal beyond the caption.
- **Cosine alignment** to the CLIP image embedding — auxiliary KD term.
- **Category cross-entropy** with weight 1.5 — directly optimizes the zero-shot classification eval. The single biggest lever for accuracy.
- **Debiased InfoNCE** (class-aware) — prevents same-category negatives from being pushed apart.
- All terms have tunable λ; full ablation grid in [task2b/run_ablations.sh](task2b/run_ablations.sh).

**Text-tower update strategies** (one of):
- `frozen` — text encoder fixed.
- `partial` — last *N* layers unfrozen at low LR.
- `lora` — LoRA adapters on the text encoder (via `peft`).
- `adapter` — bottleneck adapters in each text-encoder block.

**Training discipline:** EMA on encoder + projection (decay 0.999, +~0.5%), prompt-ensemble for category prototypes (8 templates averaged), 3-seed ensembling planned.

**Evaluation suite** ([task2b/eval_task2b.py](task2b/eval_task2b.py)) — every metric the handout requires, plus a few:

- Instance Recall@{1, 3, 5} *and* class-aware Recall@{1, 3, 5}.
- Caption-level MAP *and* class-aware MAP *and* per-class MAP.
- BERTScore distribution against ground-truth captions.
- CLIPScore distributions: matched, retrieved-by-EEG, random pair.
- 20×20 confusion matrix, top-K confusions, per-subject accuracy.

**Single-seed result** (`full_s42`, [eval_results/full_s42/eval_results.json](eval_results/full_s42/eval_results.json)):

| Metric | Value |
|---|---|
| Test classification accuracy | 11.25% |
| Class-aware Recall@1 / @3 / @5 | 11.17% / 15.73% / 18.13% |
| Caption-level MAP | 0.0047 |
| Class-aware MAP | 0.1405 |
| Mean BERTScore | 0.872 (100% above 0.7) |
| CLIPScore matched | 0.221 (mean) |
| CLIPScore retrieved-by-EEG | 0.341 |
| CLIPScore random pair | 0.196 |

The CLIPScore numbers are the most telling: captions retrieved by EEG (0.341) score *higher* against the actual image than the matched ground-truth captions (0.221). Reading: the model latches onto categorically-relevant retrieval, not exact-caption retrieval — which is consistent with the 11% instance R@1 vs ~11% class-aware R@1.

The 3-seed ensemble run is designed but was not completed before submission.

---

## 5. Repository map

| Path | Stage | What's there |
|---|---|---|
| [eeg_data_pipeline.ipynb](eeg_data_pipeline.ipynb) | Pre-midterm | Raw → preprocessed pipeline, EDA plots, train/val/test split. Builds [artifacts/](artifacts/). |
| [baseline_mlp.ipynb](baseline_mlp.ipynb) | Pre-midterm | MLP baseline (8.73%). |
| [cnn_transformer.ipynb](cnn_transformer.ipynb) | Pre-midterm | The 5% failure case. |
| [EEGNET.ipynb](EEGNET.ipynb) | Pre-midterm | EEGNet baseline (9.46%). |
| [task2a_clip_retrieval.py](task2a_clip_retrieval.py) / [.ipynb](task2a_clip_retrieval.ipynb) | Pre-midterm | Zero-shot frozen-CLIP retrieval. Outputs in [task2a_results/](task2a_results/). |
| [atm/](atm/) | Post-midterm | ATM encoder + classifier. Best single 12.27%, ensemble 12.67%. Self-contained — no new deps. |
| [enigma/](enigma/) | Post-midterm | ENIGMA encoder + classifier. 9.35%. Architectural foil for ATM. |
| [task2b/](task2b/) | Post-midterm | EEG → CLIP alignment with full ablation grid + full eval suite. |
| [eval_results/](eval_results/) | Post-midterm | task2b CLIP-eval outputs (`eval_results.json`, confusion matrix). |
| [artifacts/](artifacts/) | — | Pipeline outputs consumed by every model: `train.csv`/`val.csv`/`test.csv`, `norm_stats.pkl`, `bad_channels.pkl`, `label_mappings.pkl`, EDA plots. |
| [results/](results/) | — | Per-model artifacts (loss curves + confusion matrices + Kaggle submissions). |
| [environment.yml](environment.yml) | — | Conda env (`eeg_idl`). PyTorch ≥ 2.5 / CUDA 12.6, transformers, peft, bert_score. |
| [config.py](config.py) | — | Shared paths. |
| [captions.txt](captions.txt) | — | Per-trial captions. |
| [final_report_notes.md](final_report_notes.md) | — | Deep-dive working notes for the final report (all curves, ablations, references). |
| [midterm_report.tex](midterm_report.tex) | — | LaTeX source for the midterm report. |

Each post-midterm directory has its own README with provenance: [atm/README.md](atm/README.md), [enigma/README.md](enigma/README.md), [task2b/README.md](task2b/README.md).

---

## 6. Bottom-line scoreboard

| Model | Params | Val | Test | vs EEGNet | vs chance | Source |
|---|---:|---:|---:|---:|---:|---|
| Baseline MLP                    | 757 K   | 8.50%  | 8.73%  | −0.73 pt | 1.7× | [baseline_mlp.ipynb](baseline_mlp.ipynb) |
| CNN-Transformer (failure)       | 1.25 M  | 5.00%  | 5.00%  | −4.46 pt | 1.0× | [cnn_transformer.ipynb](cnn_transformer.ipynb) |
| **EEGNet** (baseline)           | 36 K    | 8.50%  | **9.46%**  | — | 1.9× | [EEGNET.ipynb](EEGNET.ipynb) |
| ENIGMA (CE, shared head)        | 5.90 M  | 9.60%  | 9.35%  | −0.11 pt | 1.9× | [enigma/](enigma/) |
| ATM, paper defaults             | 6.57 M  | 11.29% | 11.58% | +2.12 pt | 2.3× | [atm/](atm/) |
| ATM + MixUp + WD                | 6.57 M  | 11.63% | 12.00% | +2.54 pt | 2.4× | [atm/](atm/) |
| ATM best single (config #4, seed 43) | 6.82 M | 11.83% | **12.27%** | +2.81 pt | 2.5× | [atm/checkpoints/atm_sharmar_20260419-0319.json](atm/checkpoints/atm_sharmar_20260419-0319.json) |
| **ATM 2-seed ensemble {42, 43}** | 2 × 6.82 M | — | **12.67%** | **+3.21 pt** | **2.53×** | [atm/ensemble_eval.py](atm/ensemble_eval.py) |
| Task 2B (CLIP-aligned, single seed) | 6.82 M + proj | 11.15% | 11.25% | +1.79 pt | 2.3× | [eval_results/full_s42/](eval_results/full_s42/) |

---

## 7. Reproducing

```bash
# Environment (PSC Bridges-2)
conda env create -f environment.yml
conda activate eeg_idl

# 1. Build artifacts (one time, runs the full preprocessing pipeline)
jupyter nbconvert --to notebook --execute eeg_data_pipeline.ipynb

# 2. Pre-midterm baselines (optional — for comparison)
jupyter nbconvert --to notebook --execute baseline_mlp.ipynb
jupyter nbconvert --to notebook --execute cnn_transformer.ipynb
jupyter nbconvert --to notebook --execute EEGNET.ipynb

# 3. Task 1 — ATM (best single model)
cd atm
python sanity_check_atm.py        # 100-trial overfit; must pass before full run
python train_atm.py               # 40 epochs, ~20 min on V100
python ensemble_eval.py           # 2-seed softmax average

# 4. Task 1 — ENIGMA (architecture foil)
cd ../enigma
python sanity_check_enigma.py
python train_enigma.py            # ~16 min on V100; expect overfit by epoch 8

# 5. Task 2B — CLIP alignment on top of ATM
cd ../task2b
python clip_utils.py --captions-txt ../captions.txt \
                     --images-dir <stimuli-dir> --out-dir <clip-cache-dir>
python train_clip.py --encoder-ckpt ../atm/checkpoints/atm_sharmar_20260419-0319.pt \
                     --clip-cache <clip-cache-dir> --run-name full_s42 --seed 42
python eval_task2b.py --checkpoints clip_runs/full_s42.pt \
                      --clip-cache <clip-cache-dir> \
                      --out-dir ../eval_results/full_s42
```

Hardware: 1× V100, PSC Bridges-2 GPU-shared partition. ATM full run ≈ 20 min/40 ep; ENIGMA ≈ 16 min/40 ep; full Task 2B run ≈ 2–3 h.

---

## 8. What's *not* in this repo (and why)

- **HW4P2 (`4p2/`)** — separate homework, not part of the project.
- **`atm_reference/`, `enigma_reference/`** — third-party reference clones with their own `.git`. Kept locally as read-only documentation; full attribution in [atm/README.md](atm/README.md) and [enigma/README.md](enigma/README.md). Both are MIT/Apache licensed.
- **Model checkpoints (`*.pt`)** — gitignored. Only the JSON sidecars (full hyperparameters + per-epoch history + best metrics) are committed under each model's `checkpoints/` directory. That's enough to *reproduce* every plot in this README from scratch.
- **Large eval artifacts** — `similarity_matrix.npy` (~84 MB), `ensemble_predictions.pt`, `clip_cache/`, `wandb/` runs.

---

## 9. What's still on the table

The interesting follow-ups, in order of likely return:

1. **CLIP auxiliary loss for ATM in Task 1.** ATM's *original* loss was contrastive against CLIP, not CE — adding a CLIP-aux term to the Task 1 objective probably breaks the 12.7% ceiling. We didn't get there.
2. **Shrink ENIGMA's per-subject adapter** — low-rank (LoRA-style) or `I + A_s` residual instead of full `Linear(500, 500)`. Likely fixes the generalization gap by removing the memorization capacity.
3. **Early-stop ENIGMA at epoch 7–10.** Saves 13 min/run, costs nothing.
4. **Deeper ATM transformer** (`e_layers = 2 or 3`) once CLIP supervision is in place.
5. **Leave-one-subject-out evaluation** — true cross-subject generalization. We never ran this.
6. **Distill ATM → EEGNet** via soft labels — quantifies how much of the ATM signal is recoverable at 36K parameters. Cheap, illuminating.
7. **3-seed ensemble for Task 2B** — wired up in `train_clip.py`, not run.

---

## 10. Attribution

- ATM model code adapted from <https://github.com/ncclab-sustech/EEG_Image_decode> (MIT, © DongyangLi 2023). Specific line citations in [atm/model.py](atm/model.py) and [atm/README.md](atm/README.md).
- ENIGMA model code adapted from <https://github.com/Alljoined/ENIGMA> (Apache-2.0). Citations in [enigma/model.py](enigma/model.py) and [enigma/README.md](enigma/README.md).
- CLIP teacher: `openai/clip-vit-large-patch14` via HuggingFace `transformers`.

**Papers we built on:**
- Li et al., *Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion*. NeurIPS 2024 (arXiv:2403.07721).
- Kneeland et al., ENIGMA. NeurIPS 2025 workshop. <https://github.com/Alljoined/ENIGMA>.
- Liu et al., *iTransformer*. ICLR 2024 — the channel-as-token formulation underneath ATM and ENIGMA.
- Schirrmeister et al., *Deep learning with convolutional neural networks for EEG decoding and visualization*. Human Brain Mapping 38(11), 2017 — ShallowNet, the convolutional stage common to all three encoders.
- Lawhern et al., *EEGNet*. J. Neural Eng. 15(5), 2018 — the baseline we had to beat.
- Radford et al., *CLIP*. ICML 2021.
