"""
Task 2A: Image-Caption Retrieval with Pretrained CLIP
=====================================================
11-685 Guided Project — Midterm Checkpoint

This script performs zero-shot image↔caption retrieval using a pretrained CLIP model
and computes ALL metrics required by Section 5.4.3:
  1. Instance-level Recall@1, @3, @5 (both directions)
  2. Class-aware Recall@1, @3, @5
  3. BERTScore (F1) for retrieved captions
  4. CLIPScore distributions (matched vs mismatched)
  5. Mean Average Precision (MAP): caption-level, class-aware, per-class

Dataset: PSC path /ocean/projects/cis250019p/gandotra/11785-gp-eeg
"""

import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PSC
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from bert_score import score as bert_score_fn

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Dataset paths on PSC ──
DATA_ROOT = "/ocean/projects/cis250019p/gandotra/11785-gp-eeg"
CAPTIONS_FILE = os.path.join(DATA_ROOT, "captions.txt")
IMG_FOLDER = os.path.join(DATA_ROOT, "All_images")      # adjust if your images dir differs
# If images are in "images/" instead, uncomment below:
# IMG_FOLDER = os.path.join(DATA_ROOT, "images")

# ── CLIP model ──
# Options: "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16",
#          "openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14-336"
MODEL_NAME = "openai/clip-vit-base-patch32"

# ── Sampling (set to 1.0 for full dataset, 0.1 for quick dev runs) ──
SAMPLE_FRAC = 0.1

# ── Output directory for plots ──
OUT_DIR = "task2a_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 2. LOAD DATA
# ──────────────────────────────────────────────

print("\n[1/7] Loading captions and sampling data ...")
df = pd.read_csv(CAPTIONS_FILE, sep="\t")
print(f"  Full dataset: {len(df)} rows")
print(f"  Columns: {list(df.columns)}")
print(f"  Categories: {df['category'].nunique()} unique")

if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=SEED).reset_index(drop=True)
    print(f"  Sampled {SAMPLE_FRAC*100:.0f}%: {len(df)} rows")


# ──────────────────────────────────────────────
# 3. LOAD CLIP MODEL
# ──────────────────────────────────────────────

print(f"\n[2/7] Loading CLIP model: {MODEL_NAME} ...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("  Model loaded.")


# ──────────────────────────────────────────────
# 4. ENCODE TEXT + IMAGES
# ──────────────────────────────────────────────

def encode_texts(texts, batch_size=64):
    """Encode captions → normalized embeddings."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt",
                           padding=True, truncation=True).to(device)
        with torch.no_grad():
            embs = model.get_text_features(**inputs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


def encode_images(image_names, batch_size=32):
    """Encode images → normalized embeddings (skip missing files)."""
    embs_list, valid_idx = [], []
    for idx, name in tqdm(enumerate(image_names), total=len(image_names),
                          desc="  Encoding images"):
        # try common extensions
        found_path = None
        for ext in [".jpg", ".jpeg", ".JPEG", ".JPG", ".png"]:
            p = os.path.join(IMG_FOLDER, name + ext)
            if os.path.exists(p):
                found_path = p
                break
        if found_path is None:
            continue

        img = Image.open(found_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embs_list.append(emb.cpu())
        valid_idx.append(idx)

    if not embs_list:
        raise RuntimeError("No valid images found! Check IMG_FOLDER path.")
    return torch.cat(embs_list, dim=0), valid_idx


print("\n[3/7] Encoding captions ...")
text_embs = encode_texts(df["abstracted"].tolist())
print(f"  Text embeddings shape: {text_embs.shape}")

print("\n[4/7] Encoding images ...")
image_embs, valid_idx = encode_images(df["image_name"].tolist())
print(f"  Image embeddings shape: {image_embs.shape}")
print(f"  Valid images: {len(valid_idx)} / {len(df)}")

# Keep only the valid subset (images that were found on disk)
text_embs = text_embs[valid_idx]
df_valid = df.iloc[valid_idx].reset_index(drop=True)
N = len(df_valid)

# Move to device for similarity computation
image_embs = image_embs.to(device)
text_embs = text_embs.to(device)

# ── Similarity matrix: [N_images x N_captions] ──
sim_matrix = image_embs @ text_embs.T  # (N, N)
print(f"  Similarity matrix shape: {sim_matrix.shape}")


# ──────────────────────────────────────────────
# 5. RETRIEVAL METRICS
# ──────────────────────────────────────────────

# ── 5a. Instance-level Recall@K ──

def recall_at_k(similarity, k):
    """Recall@K: fraction where the ground-truth index i is in the top-k of row i."""
    n = similarity.shape[0]
    topk = similarity.topk(k, dim=1).indices  # (N, k)
    correct = sum(1 for i in range(n) if i in topk[i].tolist())
    return correct / n


# ── 5b. Class-aware Recall@K ──

def class_recall_at_k(similarity, classes, k):
    """Recall@K where any caption from the same class counts as correct."""
    n = similarity.shape[0]
    topk = similarity.topk(k, dim=1).indices
    correct = 0
    for i in range(n):
        if any(classes[j] == classes[i] for j in topk[i].tolist()):
            correct += 1
    return correct / n


# ── 5c. Mean Average Precision ──

def average_precision(sim_row, gt_relevant_set):
    """AP for a single query given a set of relevant indices."""
    ranked = sim_row.argsort(descending=True).tolist()
    hits, running_sum = 0, 0.0
    for rank, idx in enumerate(ranked, start=1):
        if idx in gt_relevant_set:
            hits += 1
            running_sum += hits / rank
    return running_sum / len(gt_relevant_set) if gt_relevant_set else 0.0


def compute_map_caption_level(similarity):
    """Caption-level MAP: only exact ground-truth caption is relevant."""
    n = similarity.shape[0]
    aps = []
    for i in range(n):
        ap = average_precision(similarity[i], {i})
        aps.append(ap)
    return np.mean(aps)


def compute_map_class_aware(similarity, classes):
    """Class-aware MAP: any caption from the same class is relevant."""
    n = similarity.shape[0]
    # precompute class→index mapping
    class_indices = {}
    for idx, c in enumerate(classes):
        class_indices.setdefault(c, set()).add(idx)
    aps = []
    for i in range(n):
        relevant = class_indices[classes[i]]
        ap = average_precision(similarity[i], relevant)
        aps.append(ap)
    return np.mean(aps)


def compute_map_per_class(similarity, classes):
    """Per-class MAP (class-aware relevant set, averaged within each class)."""
    class_indices = {}
    for idx, c in enumerate(classes):
        class_indices.setdefault(c, set()).add(idx)
    per_class_map = {}
    for cls, indices in sorted(class_indices.items()):
        aps = []
        for i in indices:
            ap = average_precision(similarity[i], class_indices[cls])
            aps.append(ap)
        per_class_map[cls] = np.mean(aps)
    return per_class_map


print("\n[5/7] Computing retrieval metrics ...")
classes = df_valid["category"].tolist()

print("\n── Image → Caption Retrieval ──")
for k in [1, 3, 5]:
    inst = recall_at_k(sim_matrix, k)
    cls  = class_recall_at_k(sim_matrix, classes, k)
    print(f"  R@{k}  Instance: {inst:.4f}   Class-aware: {cls:.4f}")

print("\n── Caption → Image Retrieval ──")
for k in [1, 3, 5]:
    inst = recall_at_k(sim_matrix.T, k)
    cls  = class_recall_at_k(sim_matrix.T, classes, k)
    print(f"  R@{k}  Instance: {inst:.4f}   Class-aware: {cls:.4f}")

# MAP
map_caption = compute_map_caption_level(sim_matrix)
map_class   = compute_map_class_aware(sim_matrix, classes)
map_per_cls = compute_map_per_class(sim_matrix, classes)

print(f"\n── Mean Average Precision (Image→Caption) ──")
print(f"  Caption-level MAP: {map_caption:.4f}")
print(f"  Class-aware MAP:   {map_class:.4f}")
print(f"  Per-class MAP:")
for cls, val in map_per_cls.items():
    print(f"    {cls:20s}: {val:.4f}")


# ──────────────────────────────────────────────
# 6. SEMANTIC METRICS: BERTScore + CLIPScore
# ──────────────────────────────────────────────

print("\n[6/7] Computing BERTScore & CLIPScore ...")

# For each image, get the top-1 retrieved caption
best_caption_idx = sim_matrix.argmax(dim=1).cpu().tolist()
true_captions      = df_valid["abstracted"].tolist()
retrieved_captions = [true_captions[j] for j in best_caption_idx]

# ── BERTScore ──
print("  Running BERTScore (this may take a minute) ...")
P, R, F1 = bert_score_fn(retrieved_captions, true_captions, lang="en", verbose=False)
bert_f1 = F1.numpy()

print(f"  BERTScore F1 — Mean: {bert_f1.mean():.4f}  Std: {bert_f1.std():.4f}")
high_sem = (bert_f1 > 0.7).sum()
print(f"  BERTScore F1 > 0.7:  {high_sem}/{N} ({100*high_sem/N:.1f}%)")

# ── CLIPScore distributions ──
sim_np = sim_matrix.cpu().numpy()

# Matched (diagonal): similarity of image_i with its true caption_i
clip_score_matched = np.array([sim_np[i, i] for i in range(N)])

# Best retrieved
clip_score_retrieved = np.array([sim_np[i, best_caption_idx[i]] for i in range(N)])

# Mismatched / random
clip_score_random = np.array([
    sim_np[i, random.choice([j for j in range(N) if j != i])]
    for i in range(N)
])


# ──────────────────────────────────────────────
# 7. PLOTS
# ──────────────────────────────────────────────

print("\n[7/7] Generating plots ...")

# ── Plot 1: CLIPScore — Matched vs Retrieved ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(clip_score_matched, bins=30, alpha=0.6, label="True (matched)", color="steelblue")
ax.hist(clip_score_retrieved, bins=30, alpha=0.6, label="Retrieved (top-1)", color="orange")
ax.set_xlabel("CLIPScore (cosine similarity)")
ax.set_ylabel("Frequency")
ax.set_title("CLIPScore Distribution: Matched vs Retrieved Captions")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "clipscore_matched_vs_retrieved.png"), dpi=150)
plt.close(fig)

# ── Plot 2: CLIPScore — Matched vs Retrieved vs Random ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(clip_score_matched, bins=30, alpha=0.6, label="Matched", color="steelblue")
ax.hist(clip_score_retrieved, bins=30, alpha=0.6, label="Retrieved", color="orange")
ax.hist(clip_score_random, bins=30, alpha=0.6, label="Random (mismatched)", color="green")
ax.set_xlabel("CLIPScore (cosine similarity)")
ax.set_ylabel("Frequency")
ax.set_title("CLIPScore Distribution: Matched vs Retrieved vs Random")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "clipscore_three_way.png"), dpi=150)
plt.close(fig)

# ── Plot 3: BERTScore F1 histogram ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(bert_f1, bins=30, alpha=0.7, color="mediumpurple")
ax.axvline(0.7, color="red", linestyle="--", label="F1 = 0.7 threshold")
ax.set_xlabel("BERTScore F1")
ax.set_ylabel("Frequency")
ax.set_title("BERTScore F1 of Retrieved Captions")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bertscore_histogram.png"), dpi=150)
plt.close(fig)

# ── Plot 4: Per-class MAP bar chart ──
fig, ax = plt.subplots(figsize=(10, 5))
sorted_cls = sorted(map_per_cls.items(), key=lambda x: x[1], reverse=True)
ax.bar([c for c, _ in sorted_cls], [v for _, v in sorted_cls], color="teal")
ax.set_xlabel("Category")
ax.set_ylabel("Class-aware MAP")
ax.set_title("Per-Class MAP (Image → Caption)")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "per_class_map.png"), dpi=150)
plt.close(fig)


# ──────────────────────────────────────────────
# 8. QUALITATIVE EXAMPLES
# ──────────────────────────────────────────────

print("\n── Qualitative Retrieval Examples ──")
for i in range(min(5, N)):
    true_img = df_valid.loc[i, "image_name"]
    true_cap = df_valid.loc[i, "abstracted"]
    true_cls = df_valid.loc[i, "category"]

    ret_idx = best_caption_idx[i]
    ret_cap = true_captions[ret_idx]
    ret_cls = df_valid.loc[ret_idx, "category"]

    # best image for this caption (caption→image direction)
    best_img_idx = sim_matrix[:, i].argmax().item()
    ret_img = df_valid.loc[best_img_idx, "image_name"]

    print(f"\n  Example {i+1}")
    print(f"  {'─'*50}")
    print(f"  True Image:        {true_img}  [{true_cls}]")
    print(f"  True Caption:      {true_cap}")
    print(f"  Retrieved Caption: {ret_cap}  [{ret_cls}]")
    print(f"  Retrieved Image:   {ret_img}")


# ──────────────────────────────────────────────
# 9. SAVE SUMMARY CSV
# ──────────────────────────────────────────────

results_df = pd.DataFrame({
    "image_name":        df_valid["image_name"],
    "category":          df_valid["category"],
    "true_caption":      true_captions,
    "retrieved_caption":  retrieved_captions,
    "CLIPScore_matched":  clip_score_matched,
    "CLIPScore_retrieved": clip_score_retrieved,
    "CLIPScore_random":   clip_score_random,
    "BERTScore_F1":       bert_f1,
})
results_df.to_csv(os.path.join(OUT_DIR, "task2a_results.csv"), index=False)

print(f"\nAll plots and results saved to: {OUT_DIR}/")
print("Done!")
