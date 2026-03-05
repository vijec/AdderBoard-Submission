"""Qwen3-style 1-layer transformer for 10-digit addition.

Architecture: Single Qwen3 block with GQA, RoPE, SwiGLU MLP, RMSNorm.
Supports weight tying (tie_kv, tie_qo) and norm sharing (share_qk_norm).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ────────────────────────────────────────────────────────────────

NUM_DIGITS = 10
SUM_DIGITS = 11
MAX_ADDEND = 10**NUM_DIGITS - 1
VOCAB_SIZE = 10  # digits 0-9
INPUT_LEN = 24
OUTPUT_LEN = SUM_DIGITS
TOTAL_LEN = INPUT_LEN + OUTPUT_LEN  # 35


# ── RoPE ─────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_len: int, theta: float = 3.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    T = x.shape[2]
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out1 = x1 * cos_t - x2 * sin_t
    out2 = x1 * sin_t + x2 * cos_t
    return torch.stack([out1, out2], dim=-1).flatten(-2)


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ── GQA Attention with RoPE ──────────────────────────────────────────────────

class Qwen3Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                 qk_norm: bool = True, tie_kv: bool = False, tie_qo: bool = False,
                 share_qk_norm: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.use_qk_norm = qk_norm
        self.tie_kv = tie_kv
        self.tie_qo = tie_qo

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        if not tie_kv:
            self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        if not tie_qo:
            self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = self.q_norm if share_qk_norm else RMSNorm(head_dim)

        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_proj = self.k_proj if self.tie_kv else self.v_proj
        v = v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask[:T, :T]
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if self.tie_qo:
            return F.linear(out, self.q_proj.weight.t())
        return self.o_proj(out)


# ── SwiGLU MLP ───────────────────────────────────────────────────────────────

class Qwen3MLP(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer Block ────────────────────────────────────────────────────────

class Qwen3Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, ff: int, rope_cos, rope_sin,
                 qk_norm: bool = True, tie_kv: bool = False,
                 tie_qo: bool = False, share_qk_norm: bool = False,
                 shared_norm: nn.Module = None):
        super().__init__()
        if shared_norm is not None:
            self.ln1 = shared_norm
            self.ln2 = shared_norm
        else:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        self.attn = Qwen3Attention(
            d_model, n_heads, n_kv_heads, head_dim, rope_cos, rope_sin,
            qk_norm=qk_norm, tie_kv=tie_kv, tie_qo=tie_qo,
            share_qk_norm=share_qk_norm,
        )
        self.mlp = Qwen3MLP(d_model, ff)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x
