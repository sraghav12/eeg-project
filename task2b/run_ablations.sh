#!/usr/bin/env bash
# Full Task 2B ablation grid.
#
# Run individual rows for the report's ablation table. Each changes ONE factor
# from the baseline to satisfy handout Sec. 6 ("proper ablation").
#
# Baseline (`full_s42`) = all losses on, caption target, frozen CLIP, debias=class.
#
# Expected rough wall time per row on a single V100: ~2-3 h (50 epochs @ bs=256).
# Run the most informative rows first; cut the rest if time is tight.
#
# TODO: edit these paths to match your setup.
ENC_CKPT="/ocean/projects/cis260086p/sharmar/atm/checkpoints/atm_sharmar_20260419-0319.pt"
CLIP_CACHE="/ocean/projects/cis260086p/sharmar/atm/clip_cache"
CLIP_MODEL="openai/clip-vit-large-patch14"
OUT_DIR="/ocean/projects/cis260086p/sharmar/atm/clip_runs"

COMMON="--encoder-ckpt $ENC_CKPT --clip-cache $CLIP_CACHE --clip-model $CLIP_MODEL --out-dir $OUT_DIR"

# =============================================================================
# 1. MAIN RUNS (for ensembling). Different seeds, all other settings identical.
# =============================================================================
python train_clip.py $COMMON --run-name full_s42 --seed 42
python train_clip.py $COMMON --run-name full_s43 --seed 43
python train_clip.py $COMMON --run-name full_s44 --seed 44

# Ensemble eval
python eval_task2b.py \
    --checkpoints $OUT_DIR/full_s42.pt $OUT_DIR/full_s43.pt $OUT_DIR/full_s44.pt \
    --clip-cache $CLIP_CACHE \
    --out-dir ./eval_results/ensemble_full

# =============================================================================
# 2. LOSS ABLATIONS (single seed; change one loss term at a time)
# =============================================================================
python train_clip.py $COMMON --seed 42 --run-name abl_no_infonce --no-infonce
python train_clip.py $COMMON --seed 42 --run-name abl_no_kd       --no-kd
python train_clip.py $COMMON --seed 42 --run-name abl_no_cosine   --no-cosine
python train_clip.py $COMMON --seed 42 --run-name abl_no_catce    --no-catce   # expected big drop
python train_clip.py $COMMON --seed 42 --run-name abl_no_debias   --debias-mode none

# =============================================================================
# 3. CONTRASTIVE DEBIASING VARIANTS (handout Sec. 5.5.4)
# =============================================================================
python train_clip.py $COMMON --seed 42 --run-name abl_debias_similarity --debias-mode similarity
python train_clip.py $COMMON --seed 42 --run-name abl_debias_hybrid     --debias-mode hybrid

# =============================================================================
# 4. CLIP TEXT-ENCODER STRATEGY (handout Sec. 5.5.7)
# =============================================================================
# a. Frozen CLIP  -> baseline (run above as full_s42)
# b. Partial unfreezing (last 2 layers of text encoder + text_projection)
python train_clip.py $COMMON --seed 42 --run-name abl_text_partial --text-strategy partial
# c. LoRA
python train_clip.py $COMMON --seed 42 --run-name abl_text_lora    --text-strategy lora
# d. Adapters
python train_clip.py $COMMON --seed 42 --run-name abl_text_adapter --text-strategy adapter

# =============================================================================
# 5. TARGET SELECTION (what the contrastive loss aligns EEG to)
# =============================================================================
python train_clip.py $COMMON --seed 42 --run-name abl_target_image --target image
python train_clip.py $COMMON --seed 42 --run-name abl_target_both  --target both

# =============================================================================
# 6. ENCODER FREEZING
# =============================================================================
python train_clip.py $COMMON --seed 42 --run-name abl_frozen_enc --freeze-encoder

# =============================================================================
# 7. PER-CHECKPOINT EVAL (with full metrics suite)
# =============================================================================
# Run eval on each ablation checkpoint to fill the ablation table.
# These generate the retrieval metrics too (R@K, MAP, BERTScore, CLIPScore).
for name in full_s42 \
            abl_no_infonce abl_no_kd abl_no_cosine abl_no_catce abl_no_debias \
            abl_debias_similarity abl_debias_hybrid \
            abl_text_partial abl_text_lora abl_text_adapter \
            abl_target_image abl_target_both abl_frozen_enc; do
    python eval_task2b.py \
        --checkpoints $OUT_DIR/${name}.pt \
        --clip-cache $CLIP_CACHE \
        --out-dir ./eval_results/${name} \
        --skip-bertscore  # drop this flag on final run for BERTScore
done
