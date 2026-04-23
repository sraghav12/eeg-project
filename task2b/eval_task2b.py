"""Task 2B full evaluation suite.

Computes every metric required by handout Sec. 5.4.3 and Sec. 5.6:

  Classification (zero-shot via category prototypes):
    - overall accuracy
    - per-subject accuracy
    - confusion matrix

  Retrieval (EEG -> caption):
    - Instance-level Recall@{1,3,5}
    - Class-aware Recall@{1,3,5}
    - Caption-level MAP
    - Class-aware MAP
    - Per-class MAP
    - BERTScore distribution (F1 mean/std, % > 0.7)
    - CLIPScore distribution (matched vs retrieved vs random)

  Ensembling (optional, multi-checkpoint):
    - average predictions/embeddings across seeds
    - final ensembled accuracy + retrieval metrics

Usage (single checkpoint):
    python eval_task2b.py \\
        --checkpoints clip_runs/full_s42.pt \\
        --clip-cache clip_cache \\
        --out-dir eval_results

Usage (ensemble):
    python eval_task2b.py \\
        --checkpoints clip_runs/full_s42.pt clip_runs/full_s43.pt clip_runs/full_s44.pt \\
        --clip-cache clip_cache \\
        --out-dir eval_results/ensemble
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Project modules
from model import ATMConfig, ATMClassifier
from dataset_clip import build_clip_datasets, clip_collate
from model_clip import ProjectionHead


# ----------------------------------------------------------------------------
# Load a checkpoint -> (encoder, proj) with EMA weights applied
# ----------------------------------------------------------------------------
def load_checkpoint(ckpt_path: str, device):
    ck = torch.load(ckpt_path, weights_only=False, map_location=device)
    enc_cfg = ATMConfig(**ck["enc_cfg"])
    clf = ATMClassifier(cfg=enc_cfg, num_classes=ck["n_classes"])
    encoder = clf.encoder.to(device).eval()
    clip_dim = ck["clip_dim"]
    proj = ProjectionHead(
        in_dim=enc_cfg.embed_dim, out_dim=clip_dim,
        hidden=max(1024, clip_dim), dropout=0.5,
    ).to(device).eval()
    encoder.load_state_dict(ck["encoder_state"])
    proj.load_state_dict(ck["proj_state"])
    # Apply EMA weights
    with torch.no_grad():
        for n, p in proj.named_parameters():
            if n in ck["ema_proj"]: p.data.copy_(ck["ema_proj"][n].to(device))
        if ck.get("ema_encoder"):
            for n, p in encoder.named_parameters():
                if n in ck["ema_encoder"]: p.data.copy_(ck["ema_encoder"][n].to(device))
    return encoder, proj, ck


# ----------------------------------------------------------------------------
# Inference over the test set
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_eeg_embeddings(encoder, proj, loader, device):
    """Return (eeg_embs: (N,D), labels: (N,), sids: (N,), names: list[str])."""
    embs, labels, sids, names = [], [], [], []
    for eeg, y, sid, _img, _cap, nm in loader:
        eeg, sid = eeg.to(device), sid.to(device)
        feat = encoder(eeg, sid)
        z = proj(feat)
        embs.append(z.cpu()); labels.append(y); sids.append(sid.cpu()); names.extend(nm)
    return torch.cat(embs), torch.cat(labels), torch.cat(sids), names


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def recall_at_k(sim: torch.Tensor, gt_idx: torch.Tensor, ks=(1, 3, 5)) -> dict:
    """Instance-level: the single ground-truth index must be in top-K."""
    topk = sim.topk(max(ks), dim=-1).indices
    out = {}
    for k in ks:
        hits = (topk[:, :k] == gt_idx.unsqueeze(-1)).any(dim=-1).float().mean().item()
        out[f"R@{k}"] = hits
    return out


def class_aware_recall_at_k(sim: torch.Tensor, query_classes: torch.Tensor,
                            cand_classes: torch.Tensor, ks=(1, 3, 5)) -> dict:
    """Class-aware: any retrieved candidate from the same class counts."""
    topk = sim.topk(max(ks), dim=-1).indices  # (Q, K)
    out = {}
    for k in ks:
        # Fetch class of top-k candidates for each query
        top_classes = cand_classes[topk[:, :k]]  # (Q, K)
        hits = (top_classes == query_classes.unsqueeze(-1)).any(dim=-1).float().mean().item()
        out[f"R@{k}_class"] = hits
    return out


def average_precision(sim_row: np.ndarray, relevant: set[int]) -> float:
    """AP for a single query: rank all candidates, sum precision@hits."""
    if not relevant:
        return 0.0
    order = np.argsort(-sim_row)  # descending
    hits = 0; running = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(idx) in relevant:
            hits += 1
            running += hits / rank
    return running / len(relevant)


def compute_map_caption_level(sim: np.ndarray, gt_idx: np.ndarray) -> float:
    """MAP where only the exact ground-truth caption is relevant."""
    aps = [average_precision(sim[i], {int(gt_idx[i])}) for i in range(sim.shape[0])]
    return float(np.mean(aps))


def compute_map_class_aware(sim: np.ndarray, query_classes: np.ndarray,
                            cand_classes: np.ndarray) -> tuple[float, dict]:
    """MAP where any caption of the same class counts + per-class breakdown."""
    class_to_cand_indices: dict[int, set[int]] = {}
    for idx, c in enumerate(cand_classes):
        class_to_cand_indices.setdefault(int(c), set()).add(idx)

    overall_aps = []
    per_class_aps: dict[int, list[float]] = {}
    for i in range(sim.shape[0]):
        c = int(query_classes[i])
        relevant = class_to_cand_indices.get(c, set())
        ap = average_precision(sim[i], relevant)
        overall_aps.append(ap)
        per_class_aps.setdefault(c, []).append(ap)

    per_class_map = {int(c): float(np.mean(aps)) for c, aps in per_class_aps.items()}
    return float(np.mean(overall_aps)), per_class_map


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for y, p in zip(labels, preds):
        cm[int(y), int(p)] += 1
    return cm


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="One or more .pt files from train_clip.py. "
                         "Multiple -> ensemble.")
    ap.add_argument("--clip-cache", required=True)
    ap.add_argument("--out-dir", default="./eval_results")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--skip-bertscore", action="store_true",
                    help="Skip BERTScore (slow; requires bert_score installed).")
    ap.add_argument("--retrieval-pool", choices=["test", "all"], default="test",
                    help="'test' = retrieve among test-set captions only; "
                         "'all' = use all captions in the dataset.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} | n_checkpoints={len(args.checkpoints)}")

    # ---- Data ----
    train_ds, val_ds, test_ds, sub_to_idx, n_classes, cat_protos, meta = build_clip_datasets(
        args.clip_cache, augment_train=False,
    )
    cat_protos = F.normalize(cat_protos.to(device), dim=-1)
    category_names = meta["category_names"]

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=clip_collate, num_workers=4, pin_memory=True)

    # ---- Inference per checkpoint, average embeddings + probabilities ----
    all_embs = None
    all_probs = None
    y_ref = None
    sids_ref = None
    names_ref = None
    per_ckpt_acc = []
    for ck_path in args.checkpoints:
        print(f"\n[load] {ck_path}")
        encoder, proj, ck = load_checkpoint(ck_path, device)
        embs, labels, sids, names = compute_eeg_embeddings(encoder, proj, test_loader, device)

        logits = (embs.to(device) @ cat_protos.T).cpu()
        probs = F.softmax(logits, dim=-1)
        acc = (logits.argmax(-1) == labels).float().mean().item()
        per_ckpt_acc.append(dict(ckpt=ck_path, test_acc=acc,
                                 epoch=ck.get("epoch"), val_acc=ck.get("val_acc")))
        print(f"  test_acc = {acc:.4f}")

        if all_embs is None:
            all_embs = embs.clone()
            all_probs = probs.clone()
            y_ref, sids_ref, names_ref = labels.clone(), sids.clone(), list(names)
        else:
            assert torch.equal(labels, y_ref), "test order diverged across ckpts"
            all_embs = all_embs + embs
            all_probs = all_probs + probs

    # Ensemble average
    N = len(args.checkpoints)
    all_embs = F.normalize(all_embs / N, dim=-1)      # re-normalize after averaging
    all_probs = all_probs / N
    ens_preds = all_probs.argmax(-1)
    ens_acc = (ens_preds == y_ref).float().mean().item()

    # Per-subject accuracy
    per_subj: dict[int, list[int]] = {}
    for s, p, y in zip(sids_ref.tolist(), ens_preds.tolist(), y_ref.tolist()):
        d = per_subj.setdefault(s, [0, 0]); d[0] += int(p == y); d[1] += 1
    per_subj_acc = {s: c / max(1, t) for s, (c, t) in per_subj.items()}

    print(f"\n[ensemble] N={N} test_acc={ens_acc:.4f}")
    print(f"[ensemble] per-subject: {per_subj_acc}")

    # Confusion matrix
    cm = confusion_matrix(ens_preds.numpy(), y_ref.numpy(), n_classes)
    # Most-confused pairs
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    off_diag = cm_norm.copy(); np.fill_diagonal(off_diag, 0)
    top_confusions = []
    for _ in range(10):
        i, j = np.unravel_index(off_diag.argmax(), off_diag.shape)
        if off_diag[i, j] == 0: break
        top_confusions.append(dict(
            true=category_names[i] if category_names else int(i),
            pred=category_names[j] if category_names else int(j),
            rate=float(off_diag[i, j]),
        ))
        off_diag[i, j] = 0

    # ---- Retrieval setup: build the candidate caption pool ----
    caption_embs = torch.load(os.path.join(args.clip_cache, "caption_embs.pt"),
                              weights_only=False)
    cap_df = pd.read_csv(os.path.join(args.clip_cache, "captions_df.csv"))
    name_to_caption_text = dict(zip(cap_df["image_name"].astype(str),
                                    cap_df["abstracted"].astype(str)))
    name_to_category = dict(zip(cap_df["image_name"].astype(str),
                                cap_df["category"].astype(str)))
    cat_to_idx = {c: i for i, c in enumerate(category_names)}

    # ---- Retrieval pool ----
    if args.retrieval_pool == "test":
        pool_names = [n for n in dict.fromkeys(names_ref) if n in caption_embs]
    else:
        pool_names = list(caption_embs.keys())
    pool_names = sorted(set(pool_names))
    pool_z = F.normalize(torch.stack([caption_embs[n] for n in pool_names]), dim=-1)
    pool_classes = torch.tensor([cat_to_idx[name_to_category[n]] for n in pool_names],
                                dtype=torch.long)
    print(f"[retrieval] pool size = {len(pool_names)}")

    # Query-side: keep only test trials whose image has a caption
    pool_idx = {n: i for i, n in enumerate(pool_names)}
    test_has_cap = torch.tensor([n in caption_embs and n in pool_idx for n in names_ref])
    q_embs = all_embs[test_has_cap]
    q_labels = y_ref[test_has_cap]
    q_names = [n for n, keep in zip(names_ref, test_has_cap.tolist()) if keep]
    q_gt_idx = torch.tensor([pool_idx[n] for n in q_names], dtype=torch.long)
    q_classes = torch.tensor([cat_to_idx[name_to_category[n]] for n in q_names],
                             dtype=torch.long)
    print(f"[retrieval] queries = {len(q_names)}")

    sim = (q_embs @ pool_z.T)  # (Q, P)

    # Instance-level Recall@K
    inst_recall = recall_at_k(sim, q_gt_idx, ks=(1, 3, 5))
    # Class-aware Recall@K
    cls_recall = class_aware_recall_at_k(sim, q_classes, pool_classes, ks=(1, 3, 5))

    # MAP variants
    sim_np = sim.numpy()
    gt_np = q_gt_idx.numpy()
    qcls_np = q_classes.numpy()
    pcls_np = pool_classes.numpy()
    map_caption = compute_map_caption_level(sim_np, gt_np)
    map_class, per_class_map_by_idx = compute_map_class_aware(sim_np, qcls_np, pcls_np)
    per_class_map = {category_names[c]: v for c, v in per_class_map_by_idx.items()}

    print(f"\n[retrieval] Instance Recall: {inst_recall}")
    print(f"[retrieval] Class-aware Recall: {cls_recall}")
    print(f"[retrieval] MAP caption-level: {map_caption:.4f}")
    print(f"[retrieval] MAP class-aware:   {map_class:.4f}")
    print(f"[retrieval] Per-class MAP:")
    for c, v in sorted(per_class_map.items(), key=lambda x: -x[1]):
        print(f"    {c:20s} {v:.4f}")

    # ---- CLIPScore distributions ----
    # "Matched" = similarity of EEG embedding with ITS OWN ground-truth caption emb
    # "Retrieved" = similarity of EEG embedding with the top-1 retrieved caption emb
    # "Random"    = similarity with a random non-matching caption emb
    topk1 = sim.topk(1, dim=-1).indices.squeeze(-1)
    matched_scores = sim[torch.arange(sim.size(0)), q_gt_idx].numpy()
    retrieved_scores = sim[torch.arange(sim.size(0)), topk1].numpy()
    rng = np.random.default_rng(0)
    random_idx = (np.arange(sim.size(0)) + rng.integers(1, sim.size(1), sim.size(0))) % sim.size(1)
    # Ensure random != gt
    for i in range(len(random_idx)):
        while random_idx[i] == gt_np[i]:
            random_idx[i] = rng.integers(0, sim.size(1))
    random_scores = sim_np[np.arange(sim.size(0)), random_idx]
    clipscore_stats = dict(
        matched_mean=float(matched_scores.mean()), matched_std=float(matched_scores.std()),
        retrieved_mean=float(retrieved_scores.mean()), retrieved_std=float(retrieved_scores.std()),
        random_mean=float(random_scores.mean()), random_std=float(random_scores.std()),
    )
    print(f"\n[CLIPScore] matched={clipscore_stats['matched_mean']:.4f} "
          f"retrieved={clipscore_stats['retrieved_mean']:.4f} "
          f"random={clipscore_stats['random_mean']:.4f}")

    # ---- BERTScore ----
    bert_stats = None
    if not args.skip_bertscore:
        try:
            from bert_score import score as bert_score_fn
            retrieved_caps = [name_to_caption_text[pool_names[int(i)]] for i in topk1]
            true_caps = [name_to_caption_text[n] for n in q_names]
            P, R, F1 = bert_score_fn(retrieved_caps, true_caps, lang="en",
                                      verbose=False, device=str(device))
            F1 = F1.numpy()
            bert_stats = dict(
                mean=float(F1.mean()), std=float(F1.std()),
                min=float(F1.min()), max=float(F1.max()),
                pct_above_0_7=float((F1 > 0.7).mean()),
            )
            print(f"[BERTScore] mean={bert_stats['mean']:.4f} "
                  f">0.7={bert_stats['pct_above_0_7']*100:.1f}%")
        except ImportError:
            print("[BERTScore] bert_score not installed; skipping. "
                  "Install with: pip install bert_score")

    # ---- Save everything ----
    out = dict(
        checkpoints=args.checkpoints,
        n_checkpoints=N,
        per_ckpt=per_ckpt_acc,
        ensemble_acc=ens_acc,
        per_subject_acc=per_subj_acc,
        top_confusions=top_confusions,
        retrieval=dict(
            pool_size=len(pool_names),
            n_queries=len(q_names),
            instance_recall=inst_recall,
            class_aware_recall=cls_recall,
            map_caption_level=map_caption,
            map_class_aware=map_class,
            per_class_map=per_class_map,
            clipscore=clipscore_stats,
            bertscore=bert_stats,
        ),
    )
    json_path = os.path.join(args.out_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    np.save(os.path.join(args.out_dir, "confusion_matrix.npy"), cm)
    np.save(os.path.join(args.out_dir, "similarity_matrix.npy"), sim_np)
    torch.save(dict(preds=ens_preds, labels=y_ref, sids=sids_ref, names=names_ref),
               os.path.join(args.out_dir, "ensemble_predictions.pt"))

    # Plot artifacts (optional — saved as PNG so you can drop into the report)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
        if category_names:
            ax.set_xticklabels(category_names, rotation=90, fontsize=8)
            ax.set_yticklabels(category_names, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (acc={ens_acc:.4f})")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[warn] could not save confusion matrix plot: {e}")

    print(f"\n[done] wrote results to {args.out_dir}")
    print(f"    overall test_acc: {ens_acc:.4f}")


if __name__ == "__main__":
    main()
