"""
A small low-rank adapter (LoRA) for the text reader: trained on pill imprints (scripts/train_imprint_reader.py) and
switched on for imprint reads only, so box, label and caption reading keep the original weights.
"""
import math
import re

import torch
from torch import nn

TARGETS = re.compile(r"(q_proj|k_proj|v_proj|out_proj|fc1|fc2|qkv|proj)$")  # attention and MLP layers; not lm_head
SCOPE = ("model.language_model.",)  # the text encoder and decoder; the vision tower stays as it is (GPU memory)


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / r, True
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            a, b = self.lora_a.to(x.dtype), self.lora_b.to(x.dtype)
            y = y + (self.dropout(x) @ a.t() @ b.t()) * self.scale
        return y


def add_lora(model, r=16, alpha=32, dropout=0.0, scope=SCOPE):
    """Wrap the attention and MLP linear layers under `scope`; the original weights are frozen. Returns their names."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and TARGETS.search(n) and n.startswith(tuple(scope))]
    for name in names:
        parent, _, child = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        setattr(owner, child, LoRALinear(getattr(owner, child), r, alpha, dropout))
    return names


def set_lora(model, enabled):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def lora_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith(("lora_a", "lora_b"))}


def load_lora(model, path):
    """Add the adapter saved by scripts/train_imprint_reader.py ({config, state}) to a freshly loaded model."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    add_lora(model, saved["config"]["r"], saved["config"]["alpha"], scope=saved["config"].get("scope", SCOPE))
    missing = set(lora_state(model)) - set(saved["state"])
    if missing:
        raise ValueError(f"adapter {path} lacks {len(missing)} layers (another base model?)")
    model.load_state_dict(saved["state"], strict=False)
    for m in model.modules():  # serving: the model's own precision, so no conversion on every call
        if isinstance(m, LoRALinear):
            m.lora_a.data, m.lora_b.data = m.lora_a.data.to(m.base.weight.dtype), m.lora_b.data.to(m.base.weight.dtype)
    return saved["config"]
