"""ENIGMA classifier (sharmar/enigma/).

Adapted from Kneeland et al., ENIGMA (NeurIPS 2025 workshop, arXiv:2602.10361).
Reference repo (Apache-licensed): https://github.com/Alljoined/ENIGMA
Mirrored read-only under ../enigma_reference/.

Architecture (matches the reference repo's source/models.py, lines 106-231):
    per-subject Linear(seq, seq)        -- subject-specific adapter
    Spatio-Temporal CNN                 -- shared conv backbone
        Conv2d(1 -> W, (1,5))
        AvgPool2d((1,17), (1,5))
        BN + ELU
        Conv2d(W -> W, (C, 1))          -- spatial conv across all channels
        BN + ELU + Dropout(0.5)
        Conv2d(W -> emb, 1x1)           -- projection
        rearrange + flatten -> hidden_dim
    MLP_Projector                       -- projects to CLIP dim
        Linear(hidden -> embed_dim)
        Residual(GELU + Linear + Dropout)
        LayerNorm
    Linear(embed_dim, num_classes)      -- OUR addition, classification head

Adaptations from the reference code:
    1. Subject routing uses int indices (nn.ModuleList) instead of string keys
       (nn.ModuleDict) to match our EEGDataset contract.
    2. `Rearrange` replaced by permute+view (no einops dependency).
    3. Added a shared `Linear(embed_dim, num_classes)` classifier head. The
       repo is retrieval-only (MSE+contrastive against CLIP features); we
       reuse the encoder for Task 2B CLIP retrieval.
    4. Widths scaled up from paper defaults (W=40, emb=4) to W=80, emb=8. The
       paper's defaults were tuned for 63ch/250tp; we have 122ch/500tp and
       13 subjects, so we give the shared backbone more capacity to use the
       larger per-subject adapter. See README.md for param-count table.
    5. No "latent alignment layer" -- the reference repo does not implement
       it despite the paper description. We do not add it here; flag as a
       potential ablation if time permits.

Citations in-code reference ../enigma_reference/source/models.py line numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ENIGMAConfig:
    num_channels: int = 122
    seq_len: int = 500
    num_subjects: int = 13
    # Backbone width. Paper default = 40.
    backbone_width: int = 80
    # 1x1 projection output channels. Paper default = 4.
    emb_size: int = 8
    # CLIP embedding dim. Paper uses 1024 (ViT-H-14), same as our ATM.
    embed_dim: int = 1024
    # Dropout in the MLP projector residual. Paper default = 0.5.
    dropout_proj: float = 0.5
    # Whether classifier is shared (paper philosophy) or per-subject (ablation).
    per_subject_head: bool = False


# ---------------------------------------------------------------------------
# Building blocks (adapted from enigma_reference/source/models.py)
# ---------------------------------------------------------------------------


class SpatioTemporalCNN(nn.Module):
    """Shared spatio-temporal backbone.

    Adapted from reference:models.py:106-139 (Spatio_Temporal_CNN).
    Paper defaults widened: width 40 -> 80, emb_size 4 -> 8.
    """

    def __init__(
        self,
        num_channels: int,
        width: int = 80,
        emb_size: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=(1, 5), stride=(1, 1)),
            nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 5)),
            nn.BatchNorm2d(width),
            nn.ELU(),
            nn.Conv2d(width, width, kernel_size=(num_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(width),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(width, emb_size, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.unsqueeze(1)                 # (B, 1, C, T)
        x = self.tsconv(x)                 # (B, W, 1, T')
        x = self.projection(x)             # (B, emb, 1, T')
        # einops 'b e h w -> b (h w) e' then flatten -> 'b (h w e)'
        B, E, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W * E)
        return x


class ResidualAdd(nn.Module):
    """reference:models.py:142-153"""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class MLPProjector(nn.Sequential):
    """reference:models.py:167-181 (MLP_Projector).

    Linear -> Residual(GELU + Linear + Dropout) -> LayerNorm
    """

    def __init__(self, embedding_dim: int, proj_dim: int = 1024, dropout: float = 0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(dropout),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


# ---------------------------------------------------------------------------
# ENIGMA encoder + classifier
# ---------------------------------------------------------------------------


class ENIGMAEncoder(nn.Module):
    """Per-subject Linear + shared CNN backbone + MLP projector.

    Adapted from reference:models.py:184-231 (ENIGMA). Key changes:
      - Subject routing: nn.ModuleList indexed by int id (ours) instead of
        nn.ModuleDict keyed by string (reference). This matches the
        ``(eeg, label, subject_id)`` contract of our EEGDataset.
      - Shape probe done lazily in __init__ so we don't hard-code hidden_dim.
    """

    def __init__(self, cfg: ENIGMAConfig | None = None):
        super().__init__()
        self.cfg = cfg or ENIGMAConfig()
        c = self.cfg

        # Per-subject Linear(seq, seq). Paper: ModuleDict by subject string.
        # Ours: ModuleList by integer id.
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(c.seq_len, c.seq_len) for _ in range(c.num_subjects)]
        )

        self.tsencoder = SpatioTemporalCNN(
            num_channels=c.num_channels,
            width=c.backbone_width,
            emb_size=c.emb_size,
        )

        # Compute flatten_dim by probing with a dummy tensor (matches reference
        # behavior -- they hard-coded 184 for 63ch/250tp).
        with torch.no_grad():
            dummy = torch.zeros(1, c.num_channels, c.seq_len)
            hidden_dim = self.tsencoder(dummy).shape[1]
        self.hidden_dim = hidden_dim

        self.mlp_proj = MLPProjector(
            embedding_dim=hidden_dim,
            proj_dim=c.embed_dim,
            dropout=c.dropout_proj,
        )

        # Kept so Task 2B can reuse encoder with CLIP contrastive loss.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), subject_ids: (B,) long
        # Route each trial through its subject's Linear layer. Reference
        # iterates unique subjects; we do the same with int masks.
        out = torch.empty_like(x)
        unique_ids = subject_ids.unique().tolist()
        for sid in unique_ids:
            mask = (subject_ids == sid)
            out[mask] = self.subject_wise_linear[int(sid)](x[mask])
        z = self.tsencoder(out)            # (B, hidden_dim)
        emb = self.mlp_proj(z)             # (B, embed_dim)
        return emb


class PerSubjectHead(nn.Module):
    """Batched per-subject linear head -- same pattern as atm/model.py:333-346."""

    def __init__(self, num_subjects: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_subjects, embed_dim, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_subjects, num_classes))
        bound = 1.0 / (embed_dim ** 0.5)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, emb: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        W = self.weight[subject_ids]        # (B, D, C)
        b = self.bias[subject_ids]          # (B, C)
        return torch.bmm(emb.unsqueeze(1), W).squeeze(1) + b


class ENIGMAClassifier(nn.Module):
    """ENIGMA encoder + classification head.

    Default head is shared Linear(embed_dim, num_classes) -- the paper argues
    the per-subject adapter already absorbs subject variance, so the head
    doesn't need to be subject-aware. Use cfg.per_subject_head=True for the
    ablation.
    """

    def __init__(self, cfg: ENIGMAConfig | None = None, num_classes: int = 20):
        super().__init__()
        self.cfg = cfg or ENIGMAConfig()
        self.num_classes = num_classes
        self.encoder = ENIGMAEncoder(self.cfg)

        if self.cfg.per_subject_head:
            self.classifier = PerSubjectHead(
                self.cfg.num_subjects, self.cfg.embed_dim, num_classes
            )
        else:
            self.classifier = nn.Linear(self.cfg.embed_dim, num_classes)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor):
        emb = self.encoder(x, subject_ids)
        if self.cfg.per_subject_head:
            logits = self.classifier(emb, subject_ids)
        else:
            logits = self.classifier(emb)
        return logits, emb


def build_enigma_classifier(
    num_subjects: int = 13,
    num_classes: int = 20,
    embed_dim: int = 1024,
    backbone_width: int = 80,
    emb_size: int = 8,
    per_subject_head: bool = False,
) -> ENIGMAClassifier:
    """Factory matching the signature requested in the Phase 2 brief."""
    cfg = ENIGMAConfig(
        num_subjects=num_subjects,
        embed_dim=embed_dim,
        backbone_width=backbone_width,
        emb_size=emb_size,
        per_subject_head=per_subject_head,
    )
    return ENIGMAClassifier(cfg=cfg, num_classes=num_classes)


if __name__ == "__main__":
    model = build_enigma_classifier()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_subj = sum(p.numel() for p in model.encoder.subject_wise_linear.parameters())
    n_enc = sum(p.numel() for p in model.encoder.tsencoder.parameters())
    n_mlp = sum(p.numel() for p in model.encoder.mlp_proj.parameters())
    n_head = sum(p.numel() for p in model.classifier.parameters())
    print(f"subject_wise_linear: {n_subj:>12,}")
    print(f"tsencoder:           {n_enc:>12,}")
    print(f"mlp_proj:            {n_mlp:>12,}")
    print(f"classifier:          {n_head:>12,}")
    print(f"TOTAL (trainable):   {n_params:>12,}   ({n_params/1e6:.2f}M)")
    print(f"hidden_dim:          {model.encoder.hidden_dim}")

    x = torch.randn(4, 122, 500)
    sid = torch.randint(0, 13, (4,))
    logits, emb = model(x, sid)
    print(f"logits {tuple(logits.shape)}  emb {tuple(emb.shape)}")
