"""
SparseGlobalBlock — Hybrid Sparse Self-Attention for HSG-DET.

Two variants:
  - SparseGlobalBlock        : Basic top-k sparse self-attention (residual add).
  - SparseGlobalBlockGated   : Same but with a learnable gating scalar α,
                               useful early in training to suppress attention
                               before the module has converged.

Complexity: O(k² · d)  where k ≪ H·W
  - At P5 stride 32 with 1080p input: H·W ≈ 2040 tokens
  - With k=512 → 270k ops vs 4.2M for full quadratic attention (~15× cheaper)

Both blocks are registered into ultralytics.nn.modules upon module import
so that Ultralytics YAML parsing can find them by name.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Core Block
# ─────────────────────────────────────────────────────────────────────────────

class SparseGlobalBlock(nn.Module):
    """
    Sparse Global Self-Attention block.

    Selects the top-k most salient tokens from a spatial feature map,
    runs a lightweight self-attention among those tokens only, then
    scatters the enriched values back to their original positions.

    The residual connection (x + enriched_delta) ensures the block acts
    as an additive refinement, keeping the gradient path stable.

    Args:
        c (int): Number of input/output channels.
        k (int): Number of sparse tokens to select per sample. Default 512.
    """

    def __init__(self, c: int, k: int = 512) -> None:
        super().__init__()
        self.k = k
        self.c = c

        # Lightweight 1×1 projections — no spatial aggregation, just channel
        # mixing before and after attention to allow learned token embedding.
        self.q_proj = nn.Conv2d(c, c, 1, bias=False)
        self.k_proj = nn.Conv2d(c, c, 1, bias=False)
        self.v_proj = nn.Conv2d(c, c, 1, bias=False)
        self.out_proj = nn.Conv2d(c, c, 1, bias=False)

        # Layer norm applied to selected tokens (operates on channel dim)
        self.norm = nn.LayerNorm(c)

        self._scale = c ** -0.5

    def _attention_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the sparse-attention delta (no residual).

        Returns a tensor of the same shape as ``x`` representing the
        additive correction from sparse global self-attention.  Callers
        are responsible for adding the residual (``x + delta``).
        """
        B, C, H, W = x.shape
        N = H * W
        k_actual = min(self.k, N)  # guard: k must not exceed available tokens

        # ── Project ────────────────────────────────────────────────────────
        q = self.q_proj(x).view(B, C, N)   # [B, C, N]
        kk = self.k_proj(x).view(B, C, N)  # [B, C, N]
        v = self.v_proj(x).view(B, C, N)   # [B, C, N]

        # ── Token Selection (Top-K by L2 activation energy) ─────────────
        # Use the input x (not projected) for importance so the selector
        # doesn't interfere with the learned projections.
        # Compute in FP32 for numerical stability under AMP.
        importance = x.view(B, C, N).float().pow(2).sum(dim=1)   # [B, N]
        importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
        topk_idx = torch.topk(importance, k_actual, dim=1).indices  # [B, k]
        idx_exp = topk_idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, k]

        q_sel = torch.gather(q, 2, idx_exp).transpose(1, 2)   # [B, k, C]
        k_sel = torch.gather(kk, 2, idx_exp).transpose(1, 2)  # [B, k, C]
        v_sel = torch.gather(v, 2, idx_exp).transpose(1, 2)   # [B, k, C]

        # ── Layer Norm on selected tokens ───────────────────────────────
        # Do norm + attention math in FP32 to avoid FP16 overflow/NaN.
        orig_dtype = q_sel.dtype
        norm_w = self.norm.weight
        norm_b = self.norm.bias
        q_sel = F.layer_norm(
            q_sel.float(),
            self.norm.normalized_shape,
            norm_w.float() if norm_w is not None else None,
            norm_b.float() if norm_b is not None else None,
            self.norm.eps,
        )
        k_sel = F.layer_norm(
            k_sel.float(),
            self.norm.normalized_shape,
            norm_w.float() if norm_w is not None else None,
            norm_b.float() if norm_b is not None else None,
            self.norm.eps,
        )
        v_sel = v_sel.float()

        # ── Sparse Self-Attention (stable softmax) ──────────────────────
        # attn: [B, k, k]   — cost O(k² · C) ≪ O(N² · C)
        attn = torch.bmm(q_sel, k_sel.transpose(1, 2)) * float(self._scale)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = attn.clamp(min=-80.0, max=80.0)
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attended = torch.bmm(attn, v_sel)  # [B, k, C]

        attended = attended.to(orig_dtype)

        # ── Scatter back ─────────────────────────────────────────────────
        # Only selected positions are updated; others stay identical to v.
        out = v.clone()
        out.scatter_(2, idx_exp, attended.transpose(1, 2))  # [B, C, N]

        out = out.view(B, C, H, W)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection — block acts as additive refinement
        return x + self._attention_delta(x)


# ─────────────────────────────────────────────────────────────────────────────
# Gated Variant (preferred for training stability)
# ─────────────────────────────────────────────────────────────────────────────

class SparseGlobalBlockGated(nn.Module):
    """
    Gated Sparse Global Self-Attention block.

    Identical to SparseGlobalBlock but multiplies the output by a learnable
    scalar gate α (initialised to 0). This means the block starts as an
    identity transform, gradually enabling global reasoning as α grows.

    Recommended when warm-starting from YOLO pretrained weights so the
    sparse attention path doesn't disrupt the first few epochs.

    Args:
        c (int): Number of input/output channels.
        k (int): Number of sparse tokens. Default 512.
    """

    def __init__(self, c: int, k: int = 512) -> None:
        super().__init__()
        self.block = SparseGlobalBlock(c, k)
        # Initialise gate to 0 → starts as identity
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate * self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Module registration into Ultralytics registry
# ─────────────────────────────────────────────────────────────────────────────

def _register_into_ultralytics() -> None:
    """
    Inject SparseGlobalBlock and SparseGlobalBlockGated into the
    Ultralytics namespaces so YAML parsing can resolve them.

    Ultralytics ``parse_model()`` (in ``tasks.py``) resolves module names
    via ``globals()[m]`` and then checks membership in ``base_modules`` to
    decide whether to inject scaled ``[c1, c2, ...]`` args.  We must:
      1. Inject into ``tasks.__dict__`` so ``globals()[m]`` succeeds.
      2. Add the classes to ``base_modules`` so ``parse_model`` auto-injects
         ``c1`` (actual input channels after width-scaling).

    Safe to call multiple times — idempotent.
    """
    _CLASSES = {
        "SparseGlobalBlock": SparseGlobalBlock,
        "SparseGlobalBlockGated": SparseGlobalBlockGated,
    }
    try:
        import ultralytics.nn.modules as ulm
        import ultralytics.nn as uln
        import ultralytics.nn.tasks as ult

        for name, cls in _CLASSES.items():
            setattr(ulm, name, cls)
            setattr(uln, name, cls)
            # parse_model uses globals() of tasks.py — inject there
            ult.__dict__[name] = cls

        # Add to base_modules so parse_model injects [c1, c2, *rest_args]
        # base_modules is a frozenset — replace with a new one that includes ours
        existing: frozenset = ult.__dict__.get("base_modules", frozenset())
        ult.__dict__["base_modules"] = existing | frozenset(_CLASSES.values())

    except (ImportError, AttributeError):
        # Ultralytics not installed or unexpected version — skip
        pass