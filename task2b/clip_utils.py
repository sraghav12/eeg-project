"""Precompute CLIP targets once and cache to disk.

Given captions.txt (tab-separated: dataset, category, image_name, abstracted)
and the image directory, writes to --out-dir:
    image_embs.pt        : dict[image_stem -> (D,) float32] — CLIP image features
    caption_embs.pt      : dict[image_stem -> (D,) float32] — CLIP text features of captions
    category_embs.pt     : (20, D) float32 — prompt-ensembled category prototypes
    category_names.json  : list[str] of 20 category names in label-index order
    captions_df.csv      : the captions DataFrame actually used (for reproducibility)

Run once per CLIP model you want to experiment with. Choose the strongest teacher
that fits in memory — ViT-L/14 (768-d) gives substantially better signal than
ViT-B/32 (512-d) for the ATM encoder alignment.

Usage (on Bridges-2 GPU):
    python clip_utils.py \\
        --captions-txt /ocean/projects/cis260086p/shared/eeg-project/captions.txt \\
        --images-dir   /ocean/projects/cis250019p/gandotra/11785-gp-eeg/All_images \\
        --out-dir      /ocean/projects/cis260086p/sharmar/atm/clip_cache \\
        --clip-model   openai/clip-vit-large-patch14

CLIP dims:
    openai/clip-vit-base-patch32 -> 512
    openai/clip-vit-base-patch16 -> 512
    openai/clip-vit-large-patch14 -> 768        <-- recommended default
    openai/clip-vit-large-patch14-336 -> 768
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# Prompt ensemble — subset of OpenAI's 80 templates that cover object categories.
# Ensembling lifts zero-shot accuracy by 1–2% essentially for free.
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "a photograph of a {}.",
    "an image of a {}.",
    "a close-up photo of a {}.",
    "a good photo of a {}.",
    "a photo of one {}.",
    "a bright photo of a {}.",
]


def load_captions_df(path: str) -> pd.DataFrame:
    """captions.txt is tab-separated: dataset, category, image_name, abstracted."""
    df = pd.read_csv(path, sep="\t")
    assert {"image_name", "category", "abstracted"}.issubset(df.columns), (
        f"captions.txt must have columns: dataset, category, image_name, abstracted. "
        f"Got: {list(df.columns)}"
    )
    # Normalise image_name to the stem (no extension).
    df["image_name"] = df["image_name"].apply(
        lambda s: os.path.splitext(os.path.basename(str(s)))[0]
    )
    df["abstracted"] = df["abstracted"].astype(str)
    return df


@torch.no_grad()
def encode_texts(clip, processor, texts: list[str], device, batch_size: int = 256):
    clip.eval()
    out = []
    tok = processor.tokenizer
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inp = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        feats = clip.get_text_features(**inp)
        feats = F.normalize(feats, dim=-1)
        out.append(feats.cpu().float())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_images(clip, processor, image_paths: list[Path], device, batch_size: int = 32):
    clip.eval()
    embs: dict[str, torch.Tensor] = {}
    for i in tqdm(range(0, len(image_paths), batch_size), desc="images"):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        keep_paths = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                keep_paths.append(p)
            except Exception as e:
                print(f"[warn] skipping {p}: {e}")
        if not imgs:
            continue
        inp = processor.image_processor(imgs, return_tensors="pt").to(device)
        feats = clip.get_image_features(**inp)
        feats = F.normalize(feats, dim=-1).cpu().float()
        for p, f in zip(keep_paths, feats):
            embs[p.stem] = f
    return embs


@torch.no_grad()
def build_category_prototypes(clip, processor, category_names: list[str], device):
    """Encode each of PROMPT_TEMPLATES for each category, average, renormalise."""
    protos = []
    for cat in category_names:
        word = cat.replace("_", " ")
        prompts = [t.format(word) for t in PROMPT_TEMPLATES]
        emb = encode_texts(clip, processor, prompts, device)  # (T, D)
        proto = emb.mean(dim=0)
        proto = F.normalize(proto, dim=0)
        protos.append(proto)
    return torch.stack(protos, dim=0)  # (C, D)


def resolve_image_paths(image_dir: Path, image_names: list[str]) -> list[Path]:
    """Look up each stem in image_dir with common extensions. Silent on misses."""
    found = []
    missing = 0
    for name in image_names:
        hit = None
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
            p = image_dir / f"{name}{ext}"
            if p.exists():
                hit = p; break
        if hit is None:
            missing += 1
        else:
            found.append(hit)
    if missing:
        print(f"[warn] {missing}/{len(image_names)} image names not found in {image_dir}")
    return found


def get_category_names(captions_df: pd.DataFrame) -> list[str]:
    """20 category names, sorted. Your label artifact may also define this."""
    # Sorted alphabetical ordering — stable across runs.
    return sorted(captions_df["category"].unique().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-txt", required=True,
                    help="Path to captions.txt (tab-separated).")
    ap.add_argument("--images-dir", required=True,
                    help="Directory containing stimulus .jpg files.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--image-batch-size", type=int, default=32)
    ap.add_argument("--text-batch-size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} | clip={args.clip_model}")

    clip = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    D = clip.config.projection_dim
    print(f"[model] projection dim = {D}")

    # ---- Captions DataFrame ----
    cap_df = load_captions_df(args.captions_txt)
    print(f"[captions] {len(cap_df)} rows | {cap_df['category'].nunique()} categories")
    cap_df.to_csv(os.path.join(args.out_dir, "captions_df.csv"), index=False)

    # ---- Image embeddings ----
    # Some image_names may not have an image file (unlikely for this dataset, but
    # resolve_image_paths silently skips missing ones).
    img_paths = resolve_image_paths(Path(args.images_dir), cap_df["image_name"].tolist())
    print(f"[images] resolved {len(img_paths)} files on disk")
    image_embs = encode_images(clip, processor, img_paths, device, args.image_batch_size)
    torch.save(image_embs, os.path.join(args.out_dir, "image_embs.pt"))
    print(f"[images] saved {len(image_embs)} image embeddings")

    # ---- Caption embeddings ----
    # Key by image_name so EEGCLIPDataset can look up by trial's image.
    print(f"[captions] encoding...")
    cap_feats = encode_texts(clip, processor, cap_df["abstracted"].tolist(),
                             device, args.text_batch_size)
    caption_embs = {
        name: feat for name, feat in zip(cap_df["image_name"].tolist(), cap_feats)
    }
    torch.save(caption_embs, os.path.join(args.out_dir, "caption_embs.pt"))
    print(f"[captions] saved {len(caption_embs)} caption embeddings")

    # ---- Category prototypes (prompt-ensembled) ----
    category_names = get_category_names(cap_df)
    print(f"[categories] {category_names}")
    cat_protos = build_category_prototypes(clip, processor, category_names, device)
    torch.save(cat_protos, os.path.join(args.out_dir, "category_embs.pt"))
    with open(os.path.join(args.out_dir, "category_names.json"), "w") as f:
        json.dump(category_names, f, indent=2)
    print(f"[categories] saved {cat_protos.shape} prototypes")

    # ---- Manifest ----
    manifest = {
        "clip_model": args.clip_model,
        "clip_dim": D,
        "n_images": len(image_embs),
        "n_captions": len(caption_embs),
        "n_categories": len(category_names),
        "category_names": category_names,
        "prompt_templates": PROMPT_TEMPLATES,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote cache to {args.out_dir}")


if __name__ == "__main__":
    main()
