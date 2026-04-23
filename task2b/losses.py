"""Task 2B loss functions.

All operate on L2-normalized embeddings. Four families:
    1. InfoNCE-style contrastive  (aligns EEG <-> caption embeddings)
    2. Knowledge distillation     (EEG student mimics CLIP image teacher)
    3. Cosine anchor              (similarity-based KD, stabilizer)
    4. Category cross-entropy     (zero-shot classification objective)

Handout references: Sec. 5.5.3 (KD), 5.5.4 (debiased contrastive),
5.5.5 (category CE), 5.5.6 (combined objective).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1. Contrastive (InfoNCE family)
# ----------------------------------------------------------------------------
class SymmetricInfoNCE(nn.Module):
    """Standard CLIP-style symmetric InfoNCE.

    Computes:
        Z_ij = logit_scale * <eeg_i, tgt_j>
        L = 0.5 * (CE(Z, diag) + CE(Z^T, diag))

    Uses the ATM encoder's learnable logit_scale parameter.
    """
    def forward(self, eeg_z: torch.Tensor, tgt_z: torch.Tensor,
                logit_scale: torch.Tensor) -> torch.Tensor:
        # logit_scale is the exp'd scalar; clamp upstream.
        logits = logit_scale * (eeg_z @ tgt_z.T)
        targets = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, targets)
                      + F.cross_entropy(logits.T, targets))


class DebiasedInfoNCE(nn.Module):
    """Weighted InfoNCE per handout Sec. 5.5.4.

    Down-weights same-class negatives (class mode), captions with high
    embedding similarity to the anchor (similarity mode), or both (hybrid).
    This combats the 'false negative' problem — same-class captions evoke
    similar EEG responses and shouldn't be pushed apart as strict negatives.

    Modes:
        'class'      : w_ij = alpha if labels[i] == labels[j] else 1
        'similarity' : w_ij = clamp(1 - beta * cos(tgt_i, tgt_j), 0, 1)
        'hybrid'     : product of both
        'none'       : equivalent to SymmetricInfoNCE
    Positives (i == j) always have w_ii = 1.
    """
    def __init__(self, mode: str = "class", alpha: float = 0.1, beta: float = 1.0):
        super().__init__()
        assert mode in ("class", "similarity", "hybrid", "none")
        self.mode = mode
        self.alpha = alpha
        self.beta = beta

    def _build_weights(self, labels, tgt_z):
        B = tgt_z.size(0)
        device = tgt_z.device
        eye = torch.eye(B, device=device, dtype=torch.bool)
        w = torch.ones(B, B, device=device)

        if self.mode in ("class", "hybrid"):
            assert labels is not None, "class/hybrid mode requires labels"
            same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
            w = torch.where(same, torch.full_like(w, self.alpha), w)

        if self.mode in ("similarity", "hybrid"):
            sim = (tgt_z @ tgt_z.T).clamp(min=0.0)  # only positive sim penalised
            sim_w = (1.0 - self.beta * sim).clamp(0.0, 1.0)
            sim_w = sim_w.masked_fill(eye, 1.0)
            w = w * sim_w

        return w.masked_fill(eye, 1.0).detach()

    def forward(self, eeg_z, tgt_z, logit_scale, labels=None):
        if self.mode == "none":
            return SymmetricInfoNCE()(eeg_z, tgt_z, logit_scale)

        logits = logit_scale * (eeg_z @ tgt_z.T)
        w = self._build_weights(labels, tgt_z)

        def _directional(logits_2d, w_2d):
            # Numerically stable weighted log-sum-exp:
            #   log sum_j w_ij exp(z_ij) = m + log sum_j w_ij exp(z_ij - m)
            m = logits_2d.max(dim=-1, keepdim=True).values
            denom = (w_2d * torch.exp(logits_2d - m)).sum(-1).clamp(min=1e-12)
            log_denom = torch.log(denom) + m.squeeze(-1)
            return -(logits_2d.diagonal() - log_denom).mean()

        return 0.5 * (_directional(logits, w) + _directional(logits.T, w.T))


# ----------------------------------------------------------------------------
# 2. Knowledge distillation
# ----------------------------------------------------------------------------
class LogitKDLoss(nn.Module):
    """Logit-based KD (handout Sec. 5.5.3).

    Teacher: CLIP image encoder sees the stimulus -> distribution over captions.
    Student: EEG encoder -> distribution over same captions.
    Objective: KL(teacher || student).

    The teacher's distribution is richer than one-hot ground truth — it
    encodes CLIP's own ranking over the candidate pool, which the student
    (EEG) mimics. Scale by tau_s^2 (Hinton convention) so gradient magnitude
    stays comparable across tau choices.
    """
    def __init__(self, tau_teacher: float = 0.04, tau_student: float = 0.07):
        super().__init__()
        self.tau_t = tau_teacher
        self.tau_s = tau_student

    def forward(self, eeg_z, img_z, txt_z) -> torch.Tensor:
        # candidates = batch text embeddings
        t_logits = (img_z @ txt_z.T) / self.tau_t
        s_logits = (eeg_z @ txt_z.T) / self.tau_s
        s_logp = F.log_softmax(s_logits, dim=-1)
        t_prob = F.softmax(t_logits, dim=-1)
        return F.kl_div(s_logp, t_prob, reduction="batchmean") * (self.tau_s ** 2)


class CosineAlignLoss(nn.Module):
    """Similarity-based KD anchor: 1 - cos(eeg, target). Stabilizes training."""
    def forward(self, eeg_z, tgt_z) -> torch.Tensor:
        return (1.0 - (eeg_z * tgt_z).sum(-1)).mean()


# ----------------------------------------------------------------------------
# 3. Category cross-entropy
# ----------------------------------------------------------------------------
class CategoryCELoss(nn.Module):
    """Cross-entropy against 20 category text-prompt prototypes.

    This is the PRIMARY classification-accuracy lever: it directly optimizes
    the zero-shot eval metric (argmax cos(eeg_z, category_protos)).

    category_protos: (C, D) precomputed, L2-normalized prompt-ensembled text embs.
    """
    def __init__(self, tau: float = 0.07, label_smoothing: float = 0.1):
        super().__init__()
        self.tau = tau
        self.label_smoothing = label_smoothing

    def forward(self, eeg_z, category_protos, labels) -> torch.Tensor:
        logits = (eeg_z @ category_protos.T) / self.tau
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
