"""ATM (Adaptive Thinking Mapper) encoder + classifier, adapted for the
11-685 team project (122 channels, 500 timepoints, 20 categories, 13 subjects).

Reference
---------
Li et al., "Visual Decoding and Reconstruction via EEG Embeddings with Guided
Diffusion", NeurIPS 2024 (arXiv:2403.07721).
Reference implementation: https://github.com/ncclab-sustech/EEG_Image_decode
    Retrieval/ATMS_retrieval.py (MIT License, (c) 2023 DongyangLi).

The ATM encoder is a 3-stage module:
    Stage A -- iTransformer: channel-as-token self-attention with a learned
        subject-embedding token prepended to the channel sequence.
        (ATMS_retrieval.py lines 61-93, Embed.py DataEmbedding lines 124-162.)
    Stage B -- ShallowNet-style spatio-temporal conv that collapses the
        channel dimension and produces a compact time-frequency embedding.
        (ATMS_retrieval.py PatchEmbedding lines 97-125.)
    Stage C -- 2-layer MLP with a residual block projecting to the CLIP-
        aligned embedding dimension (1024-d by default).
        (ATMS_retrieval.py Proj_eeg lines 157-167.)

Adaptations for our dataset
---------------------------
* Input shape: (B, 122, 500) instead of (B, 63, 250).
* Transformer d_model tied to seq_len = 500 (paper kept d_model == seq_len).
* PatchEmbedding spatial kernel grown from (63, 1) to (122, 1) and the
  downstream flatten dim recomputed accordingly.
* d_ff scaled from 256 -> 512 to preserve the paper's ~1x d_model/d_ff ratio.
* Classification head (Linear embed_dim -> n_classes) tacked on top of the
  encoder to produce 20-way logits -- the subject-embedding token inside the
  transformer provides per-subject conditioning, so a single shared head is
  used (decision recorded in the project README).
* All transformer components are re-implemented inline (FullAttention,
  AttentionLayer, EncoderLayer, Encoder, DataEmbedding, SubjectEmbedding) to
  avoid pulling in braindecode / reformer_pytorch from the reference repo.
  Each block is annotated with a line-number citation to the original file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Rearrange(nn.Module):
    """Replaces einops `Rearrange('b e h w -> b (h w) e')` so the file is
    dependency-free (the user's current env does not have einops installed)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, e, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, e)


# -----------------------------------------------------------------------------
# Stage A.1 : positional + subject + value embedding
# Reference: models/subject_layers/Embed.py lines 124-162 (DataEmbedding),
#            lines 8-26 (PositionalEmbedding), lines 109-121 (SubjectEmbedding).
# -----------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class SubjectEmbedding(nn.Module):
    """One learned token per subject; prepended to the channel sequence."""

    def __init__(self, num_subjects: int, d_model: int):
        super().__init__()
        self.subject_embedding = nn.Embedding(num_subjects, d_model)
        self.shared_embedding = nn.Parameter(torch.randn(1, d_model))

    def forward(self, subject_ids: torch.Tensor) -> torch.Tensor:
        if subject_ids is None or torch.any(
            subject_ids >= self.subject_embedding.num_embeddings
        ):
            batch_size = subject_ids.size(0)
            return self.shared_embedding.expand(batch_size, 1, -1)
        return self.subject_embedding(subject_ids).unsqueeze(1)


class DataEmbedding(nn.Module):
    """Value (time -> d_model) + positional + subject token."""

    def __init__(self, c_in: int, d_model: int, dropout: float, num_subjects: int):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.subject_embedding = SubjectEmbedding(num_subjects, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> value embedding operates on T -> (B, C, d_model)
        x = self.value_embedding(x)
        x = x + self.position_embedding(x)
        subj = self.subject_embedding(subject_ids)  # (B, 1, d_model)
        x = torch.cat([subj, x], dim=1)  # (B, C+1, d_model)
        return self.dropout(x)


# -----------------------------------------------------------------------------
# Stage A.2 : attention + encoder
# Reference: models/subject_layers/SelfAttention_Family.py lines 48-75
#            (FullAttention) and lines 179-213 (AttentionLayer).
#            models/subject_layers/Transformer_EncDec.py lines 27-80
#            (EncoderLayer, Encoder).
# -----------------------------------------------------------------------------
class FullAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v):
        # q,k,v: (B, L, H, E)
        B, L, H, E = q.shape
        scale = 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", q, k)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, v)
        return out.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attention_dropout: float):
        super().__init__()
        d_keys = d_model // n_heads
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.inner_attention = FullAttention(attention_dropout=attention_dropout)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        q = self.query_projection(x).view(B, L, H, -1)
        k = self.key_projection(x).view(B, L, H, -1)
        v = self.value_projection(x).view(B, L, H, -1)
        out = self.inner_attention(q, k, v).view(B, L, -1)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = AttentionLayer(d_model, n_heads, attention_dropout=dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(x))
        y = x = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class iTransformer(nn.Module):
    """Channel-as-token transformer block (Stage A of ATM)."""

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        num_subjects: int,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.enc_embedding = DataEmbedding(seq_len, d_model, dropout, num_subjects)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, C+1, d_model) after embedding
        enc_out = self.enc_embedding(x, subject_ids)
        for layer in self.layers:
            enc_out = layer(enc_out)
        enc_out = self.norm(enc_out)
        # Drop the subject token (ATMS_retrieval.py line 91).
        return enc_out[:, : self.num_channels, :]


# -----------------------------------------------------------------------------
# Stage B : ShallowNet-style temporal-spatial conv
# Reference: ATMS_retrieval.py lines 97-125 (PatchEmbedding) and lines 149-154
#            (Enc_eeg = PatchEmbedding + FlattenHead).
# -----------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, num_channels: int, emb_size: int = 40, dropout: float = 0.5):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # Spatial kernel grown 63 -> 122 to collapse all of our channels.
            nn.Conv2d(40, 40, (num_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            _Rearrange(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, d_model) -> (B, 1, C, d_model)
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x


# -----------------------------------------------------------------------------
# Stage C : MLP projection to CLIP-aligned embedding
# Reference: ATMS_retrieval.py Proj_eeg lines 157-167 and ResidualAdd lines 128-137.
# -----------------------------------------------------------------------------
class _Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


def _proj_head(in_dim: int, out_dim: int, drop: float = 0.5) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        _Residual(nn.Sequential(nn.GELU(), nn.Linear(out_dim, out_dim), nn.Dropout(drop))),
        nn.LayerNorm(out_dim),
    )


# -----------------------------------------------------------------------------
# Config + full modules
# -----------------------------------------------------------------------------
@dataclass
class ATMConfig:
    num_channels: int = 122      # our setup (paper: 63)
    seq_len: int = 500           # our setup (paper: 250)
    d_model: int = 500           # tied to seq_len (paper: 250)
    n_heads: int = 4             # paper default
    d_ff: int = 512              # scaled from paper's 256 to keep ~1x ratio
    e_layers: int 	= 1          # paper default
    dropout: float = 0.25        # paper default (transformer)
    dropout_conv: float = 0.5    # paper default (PatchEmbedding tail)
    dropout_proj: float = 0.5    # paper default (Proj_eeg residual)
    num_subjects: int = 13       # our setup (paper: 10)
    embed_dim: int = 1024        # CLIP ViT-H-14 dim (paper default)
    patch_emb_size: int = 40     # paper default
    per_subject_head: bool = False  # one Linear(embed_dim, n_classes) per subject

    def patch_out_time_dim(self) -> int:
        """Time dim after the conv stack, derived from d_model.

        Matches the paper: Conv(1,25) + AvgPool(51, stride 5) applied to T = d_model.
            after Conv(1,25) : T -> T - 24
            after AvgPool    : (T - 24 - 51)//5 + 1 = (T - 75)//5 + 1
        For d_model=500 -> 86 (paper d_model=250 -> 36).
        """
        return (self.d_model - 75) // 5 + 1

    def flatten_dim(self) -> int:
        return self.patch_out_time_dim() * self.patch_emb_size


class ATMEncoder(nn.Module):
    """Full ATM encoder producing a 1024-d embedding per trial.

    Expected input:  (B, num_channels, seq_len)   e.g. (B, 122, 500)
    Expected output: (B, embed_dim)               e.g. (B, 1024)
    """

    def __init__(self, cfg: ATMConfig | None = None):
        super().__init__()
        self.cfg = cfg or ATMConfig()
        c = self.cfg
        self.encoder = iTransformer(
            num_channels=c.num_channels,
            seq_len=c.seq_len,
            d_model=c.d_model,
            n_heads=c.n_heads,
            d_ff=c.d_ff,
            e_layers=c.e_layers,
            dropout=c.dropout,
            num_subjects=c.num_subjects,
        )
        self.patch_embed = PatchEmbedding(
            num_channels=c.num_channels,
            emb_size=c.patch_emb_size,
            dropout=c.dropout_conv,
        )
        self.proj_eeg = _proj_head(c.flatten_dim(), c.embed_dim, drop=c.dropout_proj)
        # Kept so downstream CLIP alignment (Task 2B) can match the paper.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, subject_ids)          # (B, C, d_model)
        x = self.patch_embed(x)                   # (B, T', emb)
        x = x.contiguous().view(x.size(0), -1)    # (B, flatten_dim)
        x = self.proj_eeg(x)                      # (B, embed_dim)
        return x


class PerSubjectHead(nn.Module):
    """Batched per-subject linear head: weight (S, D, C), bias (S, C)."""

    def __init__(self, num_subjects: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_subjects, embed_dim, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_subjects, num_classes))
        bound = 1.0 / math.sqrt(embed_dim)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, emb: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        W = self.weight[subject_ids]              # (B, D, C)
        b = self.bias[subject_ids]                # (B, C)
        return torch.bmm(emb.unsqueeze(1), W).squeeze(1) + b


class ATMClassifier(nn.Module):
    """ATM encoder + classification head (shared Linear OR per-subject)."""

    def __init__(self, cfg: ATMConfig | None = None, num_classes: int = 20):
        super().__init__()
        self.cfg = cfg or ATMConfig()
        self.num_classes = num_classes
        self.encoder = ATMEncoder(self.cfg)
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


def build_atm_classifier(
    num_subjects: int = 13, num_classes: int = 20, embed_dim: int = 1024
) -> ATMClassifier:
    """Factory matching the signature requested in the Phase 2 brief."""
    cfg = ATMConfig(num_subjects=num_subjects, embed_dim=embed_dim)
    return ATMClassifier(cfg=cfg, num_classes=num_classes)


if __name__ == "__main__":
    # Smoke test
    model = build_atm_classifier()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.2f}M")

    x = torch.randn(4, 122, 500)
    sid = torch.randint(0, 13, (4,))
    logits, emb = model(x, sid)
    print("logits", tuple(logits.shape), "emb", tuple(emb.shape))
