"""Model components for EEG -> CLIP alignment (Task 2B).

Provides:
    ProjectionHead            : MLP mapping encoder output (1024) -> CLIP dim (512 or 768).
    setup_clip_text_encoder   : loads CLIP and applies exactly one of:
                                frozen / partial-unfreeze / LoRA / Adapters.
    EMA                       : exponential moving average over student params.

All four fine-tuning strategies from handout Sec. 5.5.7 are implemented so you
can report the full ablation table.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Projection head
# ----------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    """Maps encoder output -> CLIP-aligned embedding.

    Handout allows linear or 1–2 hidden-layer MLP. We use a 2-layer MLP with a
    residual-ish skip and LayerNorm. Heavy dropout (0.5) because the ATM
    encoder output carries substantial noise on this dataset.
    """
    def __init__(
        self,
        in_dim: int = 1024,
        out_dim: int = 768,
        hidden: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # Initialize fc2 small so early training stays near the skip path.
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x, normalize: bool = True):
        h = self.norm_in(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        h = h + self.skip(x)
        return F.normalize(h, dim=-1) if normalize else h


# ----------------------------------------------------------------------------
# Adapters (bottleneck modules inserted inside CLIP transformer layers)
# ----------------------------------------------------------------------------
class BottleneckAdapter(nn.Module):
    """Houlsby-style bottleneck adapter: down-proj -> GELU -> up-proj + residual.

    Inserted as a post-hook on each CLIP text transformer layer. Initializes
    up_proj to zero so insertion is a no-op at init time.
    """
    def __init__(self, hidden_size: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.act = nn.GELU()
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


def _inject_adapters(text_model, bottleneck: int = 64) -> list[nn.Parameter]:
    """Wrap each text-encoder layer with a BottleneckAdapter on its output.

    Compatible with HuggingFace CLIPTextTransformer layout.
    Returns the list of newly-added trainable parameters.
    """
    layers = text_model.encoder.layers  # ModuleList of CLIPEncoderLayer
    hidden = text_model.config.hidden_size
    adapter_params = []
    for i, layer in enumerate(layers):
        adapter = BottleneckAdapter(hidden, bottleneck)
        orig_forward = layer.forward

        def make_forward(orig, ad):
            def forward(*args, **kwargs):
                out = orig(*args, **kwargs)
                # CLIPEncoderLayer returns a tuple (hidden_states, ...optional attn)
                if isinstance(out, tuple):
                    hs = ad(out[0])
                    return (hs,) + out[1:]
                return ad(out)
            return forward

        layer.forward = make_forward(orig_forward, adapter)
        layer.add_module(f"adapter_{i}", adapter)
        adapter_params.extend(adapter.parameters())

    for p in adapter_params:
        p.requires_grad = True
    return adapter_params


# ----------------------------------------------------------------------------
# CLIP text-encoder setup: frozen / partial / LoRA / adapters
# ----------------------------------------------------------------------------
def setup_clip_text_encoder(
    clip_model_name: str,
    strategy: str = "frozen",           # 'frozen' | 'partial' | 'lora' | 'adapter'
    partial_n_layers: int = 2,          # for 'partial': unfreeze last N layers
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    adapter_bottleneck: int = 64,
    unfreeze_text_projection: bool = True,
    device: str = "cuda",
):
    """Load CLIP and configure the text encoder per the chosen strategy.

    The image encoder is ALWAYS frozen — we use it only as a KD teacher and
    work off precomputed image features.

    Returns (clip_model, trainable_params: list[nn.Parameter]).
    """
    from transformers import CLIPModel

    assert strategy in ("frozen", "partial", "lora", "adapter")
    clip = CLIPModel.from_pretrained(clip_model_name)

    # Freeze everything first.
    for p in clip.parameters():
        p.requires_grad = False

    trainable: list[nn.Parameter] = []

    if strategy == "partial":
        # Unfreeze the last N transformer layers of the text encoder.
        layers = clip.text_model.encoder.layers
        n_unfreeze = min(partial_n_layers, len(layers))
        for layer in layers[-n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True
                trainable.append(p)
        # Also unfreeze the final LayerNorm for the text tower.
        if hasattr(clip.text_model, "final_layer_norm"):
            for p in clip.text_model.final_layer_norm.parameters():
                p.requires_grad = True
                trainable.append(p)

    elif strategy == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", target_modules=["q_proj", "v_proj"],
        )
        clip.text_model = get_peft_model(clip.text_model, cfg)
        for p in clip.text_model.parameters():
            if p.requires_grad:
                trainable.append(p)

    elif strategy == "adapter":
        adapter_params = _inject_adapters(clip.text_model, bottleneck=adapter_bottleneck)
        trainable.extend(adapter_params)

    # The small text_projection layer (hidden_size -> projection_dim) is cheap
    # and often helps regardless of strategy.
    if unfreeze_text_projection:
        for p in clip.text_projection.parameters():
            p.requires_grad = True
            trainable.append(p)

    clip.to(device)

    total = sum(p.numel() for p in clip.parameters())
    trn = sum(p.numel() for p in trainable)
    print(f"[clip_text] strategy={strategy} | total={total/1e6:.1f}M | "
          f"trainable={trn/1e6:.3f}M ({100*trn/max(total,1):.3f}%)")
    return clip, trainable


# ----------------------------------------------------------------------------
# Exponential Moving Average
# ----------------------------------------------------------------------------
class EMA:
    """EMA of trainable params. Applied to a model instance.

    Typical decay: 0.999. Usually gives +0.5% on EEG tasks.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if n in self.shadow
        }
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}
