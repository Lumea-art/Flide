"""Multi-head latent attention (MLA) for the DeepSeek-V2 family, in the reconstruction form.

A layer keeps one latent per token (`kv_lora_rank`) plus a shared rope part (`qk_rope_head_dim`), and the per-head keys
and values are rebuilt from the latent at attention time. That is what makes this family interesting for the session
cache: what a prefix keeps is the latent, whose size does not grow with the head count.

The arithmetic here is checked token for token against `transformers`' own layer in `tests/test_deepseek_mla.py`
(largest difference 9.3e-10 in float32). The absorbed form, which folds `kv_b_proj` into the queries and is faster but
harder, is not implemented yet.
"""
from __future__ import annotations

import torch


def rotary_freqs(position_ids: torch.Tensor, dim: int, theta: float = 10000.0, dtype=torch.complex64):
    """The complex rotary table the reference uses: `[batch, tokens, dim // 2]`, one rotation per pair."""
    half = dim // 2
    inv = theta ** (-torch.arange(half, dtype=torch.float32, device=position_ids.device) / half)
    angles = position_ids.float()[..., None] * inv
    return torch.polar(torch.ones_like(angles), angles).to(dtype)


def apply_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of `x` [batch, tokens, heads, dim] by `freqs` [batch, tokens, dim // 2]."""
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(pairs * freqs.unsqueeze(2)).flatten(3)
    return out.type_as(x)


def attention(hidden: torch.Tensor, w: dict, freqs: torch.Tensor, heads: int, nope: int, rope: int,
              vdim: int, eps: float, mask: torch.Tensor | None = None, q_lora: int | None = None):
    """One MLA layer's attention: `hidden` [batch, tokens, hidden] -> [batch, tokens, heads * vdim].

    `w` holds the layer's tensors by their checkpoint names (`q_proj` or `q_a_proj`/`q_a_layernorm`/`q_b_proj`,
    `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `o_proj`); `freqs` is the rotary table for the positions.
    """
    b, t, _ = hidden.shape
    if q_lora:
        q = w["q_b_proj"](_rms(w["q_a_proj"](hidden), w["q_a_layernorm"].weight, eps))
    else:
        q = w["q_proj"](hidden)
    q = q.view(b, t, heads, nope + rope)
    q_nope, q_pe = q.split([nope, rope], dim=-1)

    kv = w["kv_a_proj_with_mqa"](hidden)
    c_kv, k_pe = kv.split([w["kv_lora_rank"], rope], dim=-1)
    c_kv = _rms(c_kv, w["kv_a_layernorm"].weight, eps)
    kv_b = w["kv_b_proj"](c_kv).view(b, t, heads, nope + vdim)
    k_nope, v = kv_b.split([nope, vdim], dim=-1)
    k_pe = k_pe.view(b, t, 1, rope)

    q_pe = apply_rotary(q_pe, freqs)
    k_pe = apply_rotary(k_pe, freqs)
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(b, t, heads, rope)], dim=-1)

    scores = torch.einsum("bthd,bshd->bhts", q, k) * (nope + rope) ** -0.5
    if mask is None:
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=hidden.device), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = scores.softmax(-1)
    out = torch.einsum("bhts,bshd->bthd", probs, v).reshape(b, t, heads * vdim)
    return w["o_proj"](out)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).type_as(x)
