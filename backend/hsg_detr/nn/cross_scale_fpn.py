"""
CrossScaleFPN — Cross-Scale Attention FPN Fusion.

Replaces the FPN top-down Upsample+Concat path with cross-scale attention,
making attention the PRIMARY feature aggregation mechanism (not additive decoration).

Key difference from CS²GA:
  CS²GA    : sits AFTER FPN, adds small residual (LayerScale ≈ 0.07) on top of
             already-fused features → 93% gradient bypasses attention.
  CrossScaleFPN: REPLACES FPN top-down connection → attention IS the fusion,
             100% gradient flows through attention → forced to learn.

Attention is efficient because K,V stay at large-stride resolution:
  P4(40×40) ← P5(20×20):  Q=1600, K=400  →   640K per head  [very fast]
  P3(80×80) ← P4(40×40):  Q=6400, K=1600 →  10.2M per head  [fast]

No upsampling of K,V needed — attention naturally handles different resolutions.
Output shape/channels = same as Q source (x_q, the small-stride input).

Initialization: out_proj weights = 0 → starts as pure identity (x_q unchanged).
Gradient flows freely through out_proj from epoch 1 (unlike bounded LayerScale).
This "zero-init residual" pattern is used in ViT, DiT, DiffTransformer etc.

AMP-safe: all numerically sensitive ops run inside _fp32_context.

YAML usage (f=[large_layer, small_layer] — large=K,V source, small=Q source):
    [[10, 6], 1, CrossScaleFPN, []]    # P4_neck  ← P5  (layer10=P5, layer6=P4)
    [[-1, 4], 1, CrossScaleFPN, []]    # P3_out   ← P4_neck
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

_FP16_SAFE = 60000.0


def _fp32_context(device: torch.device):
    """Disable autocast so numerical ops run in FP32 regardless of AMP setting."""
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


class CrossScaleFPN(nn.Module):
    """
    Cross-Scale FPN Attention Fusion.

    Replaces Upsample + Concat in the FPN top-down path with cross-scale
    multi-head attention. Q comes from the small-stride feature map (higher
    spatial resolution), K and V come from the large-stride feature map
    (lower spatial resolution, higher semantic level).

    Args:
        c_kv: Input channels of the large-stride feature map (K,V source, e.g. P5).
        c_q:  Input channels of the small-stride feature map (Q source, e.g. P4).
              Also the output channel count — output preserves Q's channel dim.
        shared_dim: Projection dimension for Q/K/V. 0 = auto (uses c_q).
        num_heads: Number of attention heads.

    YAML example (head section of model YAML):
        # Replace FPN top-down P5→P4 path:
        [[10, 6], 1, CrossScaleFPN, []]      # args auto from parse_model
        [-1, 2, C3k2, [512, True]]           # local feature processing after attention

        # Replace FPN top-down P4_neck→P3 path:
        [[-1, 4], 1, CrossScaleFPN, []]
        [-1, 2, C3k2, [256, True]]
    """

    def __init__(
        self,
        c_kv: int,
        c_q: int,
        shared_dim: int = 0,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        # Projection dim: default to c_q, must be divisible by num_heads
        d = shared_dim if shared_dim > 0 else c_q
        d = max(num_heads, (d // num_heads) * num_heads)

        self.d = d
        self.c_q = c_q
        self.c_kv = c_kv
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self._scale = 8.0  # cosine attention scale (same as CS²GA)

        # Q projection from small-stride features (P3 or P4)
        self.q_proj = nn.Conv2d(c_q, d, 1, bias=False)
        # K, V projections from large-stride features (P4 or P5)
        self.k_proj = nn.Conv2d(c_kv, d, 1, bias=False)
        self.v_proj = nn.Conv2d(c_kv, d, 1, bias=False)
        # Output projection back to c_q channels
        self.out_proj = nn.Conv2d(d, c_q, 1, bias=False)

        # Zero-init out_proj: starts as pure identity (x_q + 0 = x_q).
        # Gradient flows through out_proj from ep1 because attention_out is
        # non-zero even at init. Unlike LayerScale (bounded, 93% bypass),
        # this lets attention contribution grow unboundedly as needed.
        nn.init.zeros_(self.out_proj.weight)

    # ------------------------------------------------------------------ #

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: [x_kv, x_q]
               x_kv — large-stride features (K,V source): e.g. P5 at 20×20
               x_q  — small-stride features (Q source):   e.g. P4 at 40×40
        Returns:
            Fused tensor at x_q's spatial size and channel count (c_q).
        """
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"CrossScaleFPN expects list[2], got {type(x)}")

        x_kv, x_q = x[0], x[1]
        input_dtype = x_q.dtype

        with _fp32_context(x_q.device):
            xq_f  = x_q.float()   # (B, c_q,  Hq,  Wq)
            xkv_f = x_kv.float()  # (B, c_kv, Hkv, Wkv)

            B, C_q,  Hq,  Wq  = xq_f.shape
            B, C_kv, Hkv, Wkv = xkv_f.shape

            Nq  = Hq  * Wq   # number of Q tokens  (e.g. 1600 for P4 at 640px)
            Nkv = Hkv * Wkv  # number of K,V tokens (e.g.  400 for P5 at 640px)
            h   = self.num_heads
            dh  = self.head_dim

            # ── Projections ────────────────────────────────────────────
            Q = F.conv2d(xq_f,  self.q_proj.weight.float())   # (B, d, Hq,  Wq)
            K = F.conv2d(xkv_f, self.k_proj.weight.float())   # (B, d, Hkv, Wkv)
            V = F.conv2d(xkv_f, self.v_proj.weight.float())   # (B, d, Hkv, Wkv)

            # ── Reshape to multi-head: (B, h, N, dh) ──────────────────
            Q = Q.reshape(B, h, dh, Nq ).permute(0, 1, 3, 2)  # (B, h, Nq,  dh)
            K = K.reshape(B, h, dh, Nkv).permute(0, 1, 3, 2)  # (B, h, Nkv, dh)
            V = V.reshape(B, h, dh, Nkv).permute(0, 1, 3, 2)  # (B, h, Nkv, dh)

            # ── Cosine attention (bounded logits, same pattern as CS²GA) ──
            # Without normalization, LayerNorm weight growth sharpens logits
            # → within-scale collapse. F.normalize keeps logits in [±scale].
            Q = F.normalize(Q, dim=-1, eps=1e-6)
            K = F.normalize(K, dim=-1, eps=1e-6)

            # attn shape: (B, h, Nq, Nkv)
            # Memory: P4←P5 → 32 × 4 × 1600 × 400 × 4B ≈ 320 MB   [fine]
            #         P3←P4 → 32 × 4 × 6400 × 1600 × 4B ≈ 5.2 GB  [use grad ckpt if OOM]
            attn = torch.matmul(Q, K.transpose(-2, -1)) * self._scale
            attn = attn.clamp(-80.0, 80.0)
            attn = attn.softmax(dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)

            # ── Aggregate V ────────────────────────────────────────────
            out = torch.matmul(attn, V)                         # (B, h, Nq, dh)
            out = out.permute(0, 1, 3, 2).reshape(B, self.d, Hq, Wq)  # (B, d, Hq, Wq)

            # ── Output projection + residual ───────────────────────────
            # out_proj is zero-init → first forward: out = x_q + 0 = x_q
            # As training progresses, out_proj grows to mix in attention output.
            out = F.conv2d(out, self.out_proj.weight.float())   # (B, c_q, Hq, Wq)
            out = xq_f + out                                    # residual

            # Safety clamp before FP16 cast
            out = torch.nan_to_num(
                out, nan=0.0, posinf=_FP16_SAFE, neginf=-_FP16_SAFE
            ).clamp(-_FP16_SAFE, _FP16_SAFE)

        return out.to(dtype=input_dtype)

    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        return (
            f"c_kv={self.c_kv}, c_q={self.c_q}, "
            f"d={self.d}, heads={self.num_heads}"
        )
