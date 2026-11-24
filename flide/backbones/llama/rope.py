"""The llama3 rotary embedding, with the scaling above `original_max_position_embeddings`.

Plain rope uses `theta ** (-2i / dim)`. Llama 3 rescales the low frequencies so a 8k-trained model can serve 128k:
wavelengths shorter than `original / high_freq_factor` are untouched, longer than `original / low_freq_factor` are
divided by `factor`, and the band between them is blended linearly. Getting the blend wrong leaves short prompts correct
and long ones wrong, so the parity test crosses the original context.
"""
from __future__ import annotations

import math

import torch


def inv_freq(dim: int, theta: float, scaling: dict | None = None) -> torch.Tensor:
    """The frequencies of the rotary pairs: `dim // 2` of them, in the checkpoint's own order."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if not scaling or scaling.get("rope_type") not in ("llama3", None):
        return inv
    factor = float(scaling.get("factor", 1.0))
    low = float(scaling.get("low_freq_factor", 1.0))
    high = float(scaling.get("high_freq_factor", 1.0))
    original = float(scaling.get("original_max_position_embeddings", 8192))
    low_wavelen, high_wavelen = original / low, original / high
    out = []
    for f in inv.tolist():
        wavelen = 2 * math.pi / f
        if wavelen < high_wavelen:
            out.append(f)
        elif wavelen > low_wavelen:
            out.append(f / factor)
        else:
            smooth = (original / wavelen - low) / (high - low) if high != low else 0.0
            out.append((1 - smooth) * f / factor + smooth * f)
    return torch.tensor(out, dtype=torch.float32)


def cos_sin(dim: int, theta: float, positions: torch.Tensor, scaling: dict | None = None, dtype=None):
    """`(cos, sin)` of shape `[1, positions, dim]`, each frequency repeated over its pair."""
    inv = inv_freq(dim, theta, scaling).to(positions.device)
    angles = positions.float().reshape(-1, 1) * inv[None, :]
    cos, sin = torch.cos(angles), torch.sin(angles)
    both = torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)
    return (t[None].to(dtype) if dtype else t[None] for t in both)
