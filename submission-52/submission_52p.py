"""AdderBoard submission: 52P (52 parameters).

1L Qwen3 with circular arc embedding, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU.
Tied K=V and Q=O projections. Shared RMSNorms + shared QK norm.
Trained with Grokfast-EMA (alpha=0.98, lambda=2.0) + iterated targeted fine-tuning.
"""

from pathlib import Path

import torch

from model.circular_arc import CircularArcQwen3
from model.qwen3 import OUTPUT_LEN
from data import encode

_SCRIPT_DIR = Path(__file__).resolve().parent
_CHECKPOINT = _SCRIPT_DIR / "checkpoints" / "best.pt"

METADATA = {
    "name": "52P",
    "author": "Enara Vijil",
    "params": 52,
    "architecture": "1L Qwen3 + circular arc embedding, d=3, 1h/1kv, hd=4, ff=2, tieKV+tieQO, shareNorms+shareQKnorm, RoPE theta=3, SwiGLU",
    "tricks": [
        "Circular arc embedding (3 params instead of 30)",
        "Tied K=V projections (share key/value weights)",
        "Tied Q=O projections (output = Q transposed)",
        "Shared all RMSNorms (-6 params vs separate)",
        "Shared QK norm (-4 params)",
        "Tied lm_head to dynamic embedding table",
        "RoPE (zero params)",
        "Grokfast-EMA (alpha=0.98, lambda=2.0)",
    ],
}


def build_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(_CHECKPOINT), map_location=device, weights_only=True)
    cfg = ckpt["config"]
    print(cfg)
    model = CircularArcQwen3(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        head_dim=cfg["head_dim"],
        ff=cfg["ff"],
        rope_theta=cfg["rope_theta"],
        qk_norm=True,
        tie_kv=cfg.get("tie_kv", False),
        tie_qo=cfg.get("tie_qo", False),
        share_norms=cfg.get("share_norms", False),
        share_block_norms=cfg.get("share_block_norms", False),
        share_qk_norm=cfg.get("share_qk_norm", False),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, METADATA


def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    inp = torch.tensor([encode(a, b)], dtype=torch.long, device=device)

    with torch.no_grad():
        x = inp
        digits = []
        for _ in range(OUTPUT_LEN):
            logits = model(x)
            next_tok = logits[0, -1, :].argmax().item()
            digits.append(next_tok)
            x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)

    return sum(d * (10 ** i) for i, d in enumerate(digits))
