"""
Cross-Scale Sparse Global Attention (CS²GA) for HSG-DETR.

Replaces per-scale SparseGlobalTokenBlock with a single module that receives
P3/P4/P5 simultaneously, projects to shared dim, selects top-k tokens per scale,
adds learned scale embeddings, runs cross-scale self-attention, and scatters back
with gated residuals.

AMP-safe: all numerically sensitive operations run inside _fp32_context
(autocast disabled), matching the pattern in sparse_global_token.py.

Compatible with Ultralytics parse_model — returns list[Tensor] which Detect
and RTDETRDecoder both accept as multi-scale feature input.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers (shared with sparse_global_token.py pattern) ─────────────────────

def _make_gn(channels: int) -> nn.GroupNorm:
    groups = min(32, int(channels))
    while int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels), eps=1e-5)


def _fp32_context(device: torch.device):
    """Disable autocast so numerical ops run in FP32 regardless of AMP setting."""
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _nan_guard(x: torch.Tensor, limit: float = 20.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


_FP16_SAFE = 60000.0


class CrossScaleSGA(nn.Module):
    """
    Cross-Scale Sparse Global Attention.

    Receives P3/P4/P5 feature maps simultaneously, projects to a shared
    dimension, selects top-k salient tokens per scale, adds learned scale
    embeddings, runs joint cross-scale self-attention, then scatters back
    with gated residuals.

    Args:
        c1: list of input channels [c_p3, c_p4, c_p5] (injected by parse_model).
        c2: list of output channels (same as c1, channel-preserving).
        shared_dim: Projection dimension for cross-scale attention.
        ratio_p3: Top-k ratio for P3 tokens.
        ratio_p4: Top-k ratio for P4 tokens.
        ratio_p5: Top-k ratio for P5 tokens.
    """

    def __init__(
        self,
        c1: list[int],
        c2: list[int],
        shared_dim: int = 256,
        ratio_p3: float = 0.05,
        ratio_p4: float = 0.10,
        ratio_p5: float = 0.25,
        debug: bool = False,
        scale_embed_alpha: float = 0.0,
        attn_scale_mult: float = 0.1,
    ) -> None:
        super().__init__()
        if isinstance(c1, int):
            raise TypeError(f"CrossScaleSGA expects c1 as list[3], got int {c1}")
        if isinstance(c2, int):
            c2 = list(c1)
        assert len(c1) == 3, f"Expected 3 scales, got {len(c1)}"

        self.c1 = list(c1)
        self.shared_dim = int(shared_dim)
        self.ratio_p3 = float(ratio_p3)
        self.ratio_p4 = float(ratio_p4)
        self.ratio_p5 = float(ratio_p5)
        self.debug_enabled: bool = bool(debug)
        self.scale_embed_alpha = float(scale_embed_alpha)
        self.attn_scale_mult = float(attn_scale_mult)

        # Debug state — updated each forward pass when debug_enabled=True
        self.last_gate_p3: float | None = None
        self.last_gate_p4: float | None = None
        self.last_gate_p5: float | None = None
        self.last_k3: int | None = None
        self.last_k4: int | None = None
        self.last_k5: int | None = None
        self.last_attn_within_frac: float | None = None   # mass on within-scale blocks
        self.last_attn_cross_frac: float | None = None    # mass on cross-scale blocks
        self.last_delta_abs_p3: float | None = None       # mean |delta| before gate
        self.last_delta_abs_p4: float | None = None
        self.last_delta_abs_p5: float | None = None

        d = self.shared_dim

        # Project each scale to shared dimension
        self.proj_p3 = nn.Conv2d(self.c1[0], d, 1, bias=False)
        self.proj_p4 = nn.Conv2d(self.c1[1], d, 1, bias=False)
        self.proj_p5 = nn.Conv2d(self.c1[2], d, 1, bias=False)

        # Scale identity embeddings (0=P3, 1=P4, 2=P5)
        self.scale_embed = nn.Embedding(3, d)
        nn.init.zeros_(self.scale_embed.weight)

        # Attention norm (applied inside FP32 island)
        self.norm = nn.LayerNorm(d)

        # Project back to original channels
        self.out_proj_p3 = nn.Conv2d(d, self.c1[0], 1, bias=False)
        self.out_proj_p4 = nn.Conv2d(d, self.c1[1], 1, bias=False)
        self.out_proj_p5 = nn.Conv2d(d, self.c1[2], 1, bias=False)

        # Pre-norms
        self.pre_norm_p3 = _make_gn(self.c1[0])
        self.pre_norm_p4 = _make_gn(self.c1[1])
        self.pre_norm_p5 = _make_gn(self.c1[2])

        # Delta norms (after out_proj)
        self.delta_norm_p3 = _make_gn(self.c1[0])
        self.delta_norm_p4 = _make_gn(self.c1[1])
        self.delta_norm_p5 = _make_gn(self.c1[2])

        # Gated residual — sigmoid(0) = 0.5 at init
        self.gate_p3 = nn.Parameter(torch.zeros(1))
        self.gate_p4 = nn.Parameter(torch.zeros(1))
        self.gate_p5 = nn.Parameter(torch.zeros(1))

        self._attn_scale = float(d ** -0.5) * max(1e-3, self.attn_scale_mult)

    # ------------------------------------------------------------------ #

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            x: [P3, P4, P5] feature maps from FPN/PAN neck.
        Returns:
            [P3_out, P4_out, P5_out] — same shapes, channel-preserving.
        """
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError(f"CrossScaleSGA expects list[3], got {type(x)}")

        P3, P4, P5 = x
        input_dtype = P3.dtype   # FP16 under AMP, FP32 otherwise

        # ── All numerically sensitive work inside FP32 context ────────────
        with _fp32_context(P3.device):

            # Cast to FP32 for safe computation
            p3_f = P3.float()
            p4_f = P4.float()
            p5_f = P5.float()

            # Pre-norm (cast weight to FP32 for GN)
            p3_n = self.pre_norm_p3(p3_f.to(dtype=self.pre_norm_p3.weight.dtype)).float()
            p4_n = self.pre_norm_p4(p4_f.to(dtype=self.pre_norm_p4.weight.dtype)).float()
            p5_n = self.pre_norm_p5(p5_f.to(dtype=self.pre_norm_p5.weight.dtype)).float()

            # 1×1 projection to shared_dim (FP32 weights)
            p3_proj = F.conv2d(p3_n, self.proj_p3.weight.float())   # (B, d, H3, W3)
            p4_proj = F.conv2d(p4_n, self.proj_p4.weight.float())
            p5_proj = F.conv2d(p5_n, self.proj_p5.weight.float())

            _, _, H3, W3 = p3_proj.shape
            _, _, H4, W4 = p4_proj.shape
            _, _, H5, W5 = p5_proj.shape

            B = p3_proj.shape[0]
            d = self.shared_dim

            # Flatten spatial → tokens
            p3_tok = p3_proj.view(B, d, -1).transpose(1, 2)  # (B, N3, d)
            p4_tok = p4_proj.view(B, d, -1).transpose(1, 2)
            p5_tok = p5_proj.view(B, d, -1).transpose(1, 2)

            # Top-k by L2 saliency (FP32, guard NaN/Inf)
            p3_scores = _nan_guard(p3_tok.norm(dim=-1))
            p4_scores = _nan_guard(p4_tok.norm(dim=-1))
            p5_scores = _nan_guard(p5_tok.norm(dim=-1))

            k3 = max(1, min(int(self.ratio_p3 * p3_tok.shape[1]), p3_tok.shape[1]))
            k4 = max(1, min(int(self.ratio_p4 * p4_tok.shape[1]), p4_tok.shape[1]))
            k5 = max(1, min(int(self.ratio_p5 * p5_tok.shape[1]), p5_tok.shape[1]))

            _, p3_idx = torch.topk(p3_scores, k3, dim=-1)
            _, p4_idx = torch.topk(p4_scores, k4, dim=-1)
            _, p5_idx = torch.topk(p5_scores, k5, dim=-1)

            # Gather selected tokens
            p3_sel = torch.gather(p3_tok, 1, p3_idx.unsqueeze(-1).expand(-1, -1, d))
            p4_sel = torch.gather(p4_tok, 1, p4_idx.unsqueeze(-1).expand(-1, -1, d))
            p5_sel = torch.gather(p5_tok, 1, p5_idx.unsqueeze(-1).expand(-1, -1, d))

            # Optional scale embeddings (FP32). Keep them disabled by default:
            # random scale identity vectors dominate Q/K similarity and make
            # attention collapse into within-scale blocks at initialization.
            dev = P3.device
            embed_alpha = float(max(0.0, min(self.scale_embed_alpha, 1.0)))
            e3 = embed_alpha * self.scale_embed(torch.tensor(0, device=dev)).float().view(1, 1, -1)
            e4 = embed_alpha * self.scale_embed(torch.tensor(1, device=dev)).float().view(1, 1, -1)
            e5 = embed_alpha * self.scale_embed(torch.tensor(2, device=dev)).float().view(1, 1, -1)

            p3_sel = _nan_guard(p3_sel + e3)
            p4_sel = _nan_guard(p4_sel + e4)
            p5_sel = _nan_guard(p5_sel + e5)

            # Concatenate cross-scale tokens (B, K, d)
            all_tok = torch.cat([p3_sel, p4_sel, p5_sel], dim=1)

            # Normalize selected token vectors before joint attention. Without
            # this, scale-specific activation magnitude can dominate the dot
            # products and collapse CS²GA into three near-independent within-
            # scale attention blocks.
            all_tok_attn = F.normalize(all_tok, dim=-1, eps=1e-6)

            # LayerNorm (FP32)
            norm_w = self.norm.weight.float()
            norm_b = self.norm.bias.float()
            q = F.layer_norm(all_tok_attn, self.norm.normalized_shape, norm_w, norm_b, self.norm.eps)
            q = _nan_guard(q)
            k_t = q
            v   = all_tok   # V uses un-normed tokens (standard: norm Q/K, not V)

            # Scaled dot-product attention (FP32)
            attn = torch.bmm(q, k_t.transpose(1, 2)) * self._attn_scale
            attn = attn.clamp(-80.0, 80.0)
            attn = attn - attn.max(dim=-1, keepdim=True).values
            attn = torch.softmax(attn, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)

            attended = torch.bmm(attn, v)   # (B, K, d)
            attended = _nan_guard(attended)

            # Split back by scale
            p3_att = attended[:, :k3, :]
            p4_att = attended[:, k3:k3 + k4, :]
            p5_att = attended[:, k3 + k4:, :]

            # Scatter attended tokens back to full spatial grids (FP32)
            p3_delta = self._scatter(p3_att, p3_idx, B, d, H3, W3)
            p4_delta = self._scatter(p4_att, p4_idx, B, d, H4, W4)
            p5_delta = self._scatter(p5_att, p5_idx, B, d, H5, W5)

            # tanh clamp — prevents delta overflow before out_proj
            # (matches sparse_global_token.py pattern)
            p3_delta = 6.0 * torch.tanh(p3_delta / 6.0)
            p4_delta = 6.0 * torch.tanh(p4_delta / 6.0)
            p5_delta = 6.0 * torch.tanh(p5_delta / 6.0)

            # out_proj in FP32 (explicit weight cast)
            p3_delta = F.conv2d(p3_delta, self.out_proj_p3.weight.float())
            p4_delta = F.conv2d(p4_delta, self.out_proj_p4.weight.float())
            p5_delta = F.conv2d(p5_delta, self.out_proj_p5.weight.float())

            # delta norm in FP32
            p3_delta = self._gn_fp32(p3_delta, self.delta_norm_p3)
            p4_delta = self._gn_fp32(p4_delta, self.delta_norm_p4)
            p5_delta = self._gn_fp32(p5_delta, self.delta_norm_p5)

            # Gated residual on ORIGINAL inputs (cast delta → input_dtype)
            g3 = torch.sigmoid(self.gate_p3.float())
            g4 = torch.sigmoid(self.gate_p4.float())
            g5 = torch.sigmoid(self.gate_p5.float())

            if self.debug_enabled:
                self.last_gate_p3 = float(g3.item())
                self.last_gate_p4 = float(g4.item())
                self.last_gate_p5 = float(g5.item())
                self.last_k3 = int(k3)
                self.last_k4 = int(k4)
                self.last_k5 = int(k5)
                # Attention distribution: mean over batch of within-scale vs cross-scale mass
                with torch.no_grad():
                    _a = attn.detach().mean(0)  # (K, K) mean over batch
                    _within = (
                        _a[:k3, :k3].sum()
                        + _a[k3:k3+k4, k3:k3+k4].sum()
                        + _a[k3+k4:, k3+k4:].sum()
                    )
                    _total = float(_a.sum().clamp(min=1e-8))
                    self.last_attn_within_frac = float(_within / _total)
                    self.last_attn_cross_frac = float(1.0 - self.last_attn_within_frac)
                # Delta magnitude (mean absolute value before gate scaling)
                self.last_delta_abs_p3 = float(p3_delta.detach().abs().mean().item())
                self.last_delta_abs_p4 = float(p4_delta.detach().abs().mean().item())
                self.last_delta_abs_p5 = float(p5_delta.detach().abs().mean().item())

            p3_out = p3_f + g3 * p3_delta
            p4_out = p4_f + g4 * p4_delta
            p5_out = p5_f + g5 * p5_delta

            # The residual path is computed in FP32, then returned to the
            # incoming AMP dtype. Clamp before FP16 cast so rare large
            # activations cannot become +/-inf and poison backward.
            p3_out = torch.nan_to_num(p3_out, nan=0.0, posinf=_FP16_SAFE, neginf=-_FP16_SAFE).clamp(-_FP16_SAFE, _FP16_SAFE)
            p4_out = torch.nan_to_num(p4_out, nan=0.0, posinf=_FP16_SAFE, neginf=-_FP16_SAFE).clamp(-_FP16_SAFE, _FP16_SAFE)
            p5_out = torch.nan_to_num(p5_out, nan=0.0, posinf=_FP16_SAFE, neginf=-_FP16_SAFE).clamp(-_FP16_SAFE, _FP16_SAFE)

        # Cast back to original input dtype (FP16 under AMP)
        return [
            p3_out.to(dtype=input_dtype),
            p4_out.to(dtype=input_dtype),
            p5_out.to(dtype=input_dtype),
        ]

    # ------------------------------------------------------------------ #

    @staticmethod
    def _scatter(
        attended: torch.Tensor,  # (B, k, d) — FP32
        indices: torch.Tensor,   # (B, k)
        B: int, d: int, H: int, W: int,
    ) -> torch.Tensor:
        """Scatter attended tokens back to (B, d, H, W) canvas."""
        N = H * W
        canvas = torch.zeros(B, d, N, device=attended.device, dtype=torch.float32)
        idx_exp = indices.unsqueeze(1).expand(-1, d, -1)          # (B, d, k)
        canvas.scatter_(2, idx_exp, attended.transpose(1, 2))
        return canvas.view(B, d, H, W)

    @staticmethod
    def _gn_fp32(x: torch.Tensor, gn: nn.GroupNorm) -> torch.Tensor:
        """GroupNorm in FP32 regardless of weight storage dtype."""
        w = gn.weight.float() if gn.weight is not None else None
        b = gn.bias.float() if gn.bias is not None else None
        return F.group_norm(x, gn.num_groups, w, b, gn.eps)

    # ------------------------------------------------------------------ #

    def set_debug(self, enabled: bool = True) -> None:
        """Enable or disable debug state recording."""
        self.debug_enabled = enabled

    def get_debug_state(self) -> dict:
        """Return last recorded debug values (None if debug_enabled=False or no forward yet)."""
        return {
            "gate_p3": self.last_gate_p3,
            "gate_p4": self.last_gate_p4,
            "gate_p5": self.last_gate_p5,
            "k3": self.last_k3,
            "k4": self.last_k4,
            "k5": self.last_k5,
            "attn_within_frac": self.last_attn_within_frac,
            "attn_cross_frac": self.last_attn_cross_frac,
            "delta_abs_p3": self.last_delta_abs_p3,
            "delta_abs_p4": self.last_delta_abs_p4,
            "delta_abs_p5": self.last_delta_abs_p5,
        }
