"""Qwen3 model with circular arc token embedding.

Replaces the 30-param nn.Embedding(10, 3) lookup table with a 3-param circular
arc parametrization:
    emb[d] = [A*cos(start + d*stride), A*sin(start + d*stride), 0]

The lm_head is tied to the dynamically computed embedding table.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.qwen3 import (
    Qwen3Block,
    RMSNorm,
    precompute_rope_freqs,
    VOCAB_SIZE,
    TOTAL_LEN,
)


class CircularArcQwen3(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1, n_kv_heads: int = 1,
                 head_dim: int = 4, ff: int = 3, rope_theta: float = 3.0,
                 max_len: int = TOTAL_LEN + 1, qk_norm: bool = True,
                 tie_kv: bool = False, tie_qo: bool = False,
                 share_norms: bool = False, share_block_norms: bool = False,
                 share_qk_norm: bool = False,
                 arc_init_A: float = 2.5, arc_init_start: float = -1.2,
                 arc_init_stride: float = 0.29):
        super().__init__()
        self.d_model = d_model

        # Circular arc embedding params (3 params)
        self.arc_A = nn.Parameter(torch.tensor(arc_init_A))
        self.arc_start = nn.Parameter(torch.tensor(arc_init_start))
        self.arc_stride = nn.Parameter(torch.tensor(arc_init_stride))

        # RoPE
        rope_cos, rope_sin = precompute_rope_freqs(head_dim, max_len, rope_theta)

        # Shared norm
        if share_norms:
            shared_norm = RMSNorm(d_model)
        elif share_block_norms:
            shared_norm = RMSNorm(d_model)
        else:
            shared_norm = None

        self.block = Qwen3Block(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
            head_dim=head_dim, ff=ff, rope_cos=rope_cos, rope_sin=rope_sin,
            qk_norm=qk_norm, tie_kv=tie_kv, tie_qo=tie_qo,
            share_qk_norm=share_qk_norm, shared_norm=shared_norm,
        )
        self.final_norm = shared_norm if share_norms else RMSNorm(d_model)

        # Causal mask
        mask = torch.full((max_len, max_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)

    def _compute_embedding_table(self) -> torch.Tensor:
        d = torch.arange(VOCAB_SIZE, device=self.arc_A.device, dtype=self.arc_A.dtype)
        angles = self.arc_start + d * self.arc_stride
        if self.d_model == 3:
            return torch.stack([
                self.arc_A * torch.cos(angles),
                self.arc_A * torch.sin(angles),
                torch.zeros_like(angles),
            ], dim=1)
        else:
            return torch.stack([
                self.arc_A * torch.cos(angles),
                self.arc_A * torch.sin(angles),
            ], dim=1)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb_table = self._compute_embedding_table()
        x = emb_table[input_ids]
        x = self.block(x, self.causal_mask)
        x = self.final_norm(x)
        logits = F.linear(x, emb_table)
        return logits
