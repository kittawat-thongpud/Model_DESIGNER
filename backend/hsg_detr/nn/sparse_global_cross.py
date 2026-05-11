"""
Cross-Scale Sparse Global Attention (CS²GA) for HSG-DETR.

Replaces per-scale SparseGlobalTokenBlock with a single module that receives
P3/P4/P5 simultaneously, projects to shared dim, selects top-k tokens per scale,
adds learned scale embeddings, runs cross-scale self-attention, and scatters back
with gated residuals.

Compatible with Ultralytics parse_model — returns list[Tensor] which Detect
and RTDETRDecoder both accept as multi-scale feature input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gn(channels: int) -> nn.GroupNorm:
    groups = min(32, int(channels))
    while int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels), eps=1e-5)


def _cast_to_param_dtype(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    try:
        param = next(module.parameters())
        return x.to(dtype=param.dtype)
    except StopIteration:
        return x


def _finite_guard(x: torch.Tensor, limit: float = 20.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


class CrossScaleSGA(nn.Module):
    """
    Cross-Scale Sparse Global Attention.

    Receives P3/P4/P5 feature maps simultaneously, projects to a shared
    dimension, selects top-k salient tokens per scale, adds learned scale
    embeddings, runs joint cross-scale self-attention, then scatters back
    with gated residuals.

    Args:
        c1: list of input channels [c_p3, c_p4, c_p5] (injected by parse_model).
        c2: list of output channels [c_p3, c_p4, c_p5] (same as c1).
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

        # Project each scale to shared dimension
        self.proj_p3 = nn.Conv2d(self.c1[0], self.shared_dim, 1, bias=False)
        self.proj_p4 = nn.Conv2d(self.c1[1], self.shared_dim, 1, bias=False)
        self.proj_p5 = nn.Conv2d(self.c1[2], self.shared_dim, 1, bias=False)

        # Scale identity embeddings (0=P3, 1=P4, 2=P5)
        self.scale_embed = nn.Embedding(3, self.shared_dim)

        # Attention norm (FP32 island)
        self.norm = nn.LayerNorm(self.shared_dim)

        # Project back to original channels
        self.out_proj_p3 = nn.Conv2d(self.shared_dim, self.c1[0], 1, bias=False)
        self.out_proj_p4 = nn.Conv2d(self.shared_dim, self.c1[1], 1, bias=False)
        self.out_proj_p5 = nn.Conv2d(self.shared_dim, self.c1[2], 1, bias=False)

        # Pre-norms (applied before projection)
        self.pre_norm_p3 = _make_gn(self.c1[0])
        self.pre_norm_p4 = _make_gn(self.c1[1])
        self.pre_norm_p5 = _make_gn(self.c1[2])

        # Delta norms (after output projection)
        self.delta_norm_p3 = _make_gn(self.c1[0])
        self.delta_norm_p4 = _make_gn(self.c1[1])
        self.delta_norm_p5 = _make_gn(self.c1[2])

        # Gated residual — sigmoid(0) = 0.5, starts at 50% contribution
        self.gate_p3 = nn.Parameter(torch.zeros(1))
        self.gate_p4 = nn.Parameter(torch.zeros(1))
        self.gate_p5 = nn.Parameter(torch.zeros(1))

        self._attn_scale = float(self.shared_dim ** -0.5)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            x: [P3, P4, P5] feature maps from FPN/PAN neck.
        Returns:
            [P3_out, P4_out, P5_out] — same shapes as input, channel-preserving.
        """
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError(f"CrossScaleSGA expects list[3], got {type(x)}")

        P3, P4, P5 = x
        B = P3.shape[0]

        # Pre-norm (AMP-safe cast)
        p3_n = self.pre_norm_p3(P3.to(dtype=self.pre_norm_p3.weight.dtype))
        p4_n = self.pre_norm_p4(P4.to(dtype=self.pre_norm_p4.weight.dtype))
        p5_n = self.pre_norm_p5(P5.to(dtype=self.pre_norm_p5.weight.dtype))

        # Project to shared_dim
        p3_proj = self.proj_p3(p3_n)  # (B, d, H3, W3)
        p4_proj = self.proj_p4(p4_n)  # (B, d, H4, W4)
        p5_proj = self.proj_p5(p5_n)  # (B, d, H5, W5)

        _, _, H3, W3 = p3_proj.shape
        _, _, H4, W4 = p4_proj.shape
        _, _, H5, W5 = p5_proj.shape

        # Flatten → tokens
        p3_tok = p3_proj.view(B, self.shared_dim, -1).transpose(1, 2)  # (B, N3, d)
        p4_tok = p4_proj.view(B, self.shared_dim, -1).transpose(1, 2)
        p5_tok = p5_proj.view(B, self.shared_dim, -1).transpose(1, 2)

        # Top-k selection by L2 saliency
        k3 = max(1, min(int(self.ratio_p3 * p3_tok.shape[1]), p3_tok.shape[1]))
        k4 = max(1, min(int(self.ratio_p4 * p4_tok.shape[1]), p4_tok.shape[1]))
        k5 = max(1, min(int(self.ratio_p5 * p5_tok.shape[1]), p5_tok.shape[1]))

        _, p3_idx = torch.topk(p3_tok.norm(dim=-1), k3, dim=-1)  # (B, k3)
        _, p4_idx = torch.topk(p4_tok.norm(dim=-1), k4, dim=-1)
        _, p5_idx = torch.topk(p5_tok.norm(dim=-1), k5, dim=-1)

        # Gather selected tokens
        p3_sel = torch.gather(p3_tok, 1, p3_idx.unsqueeze(-1).expand(-1, -1, self.shared_dim))
        p4_sel = torch.gather(p4_tok, 1, p4_idx.unsqueeze(-1).expand(-1, -1, self.shared_dim))
        p5_sel = torch.gather(p5_tok, 1, p5_idx.unsqueeze(-1).expand(-1, -1, self.shared_dim))

        # Add scale embeddings
        dev = P3.device
        p3_sel = p3_sel + self.scale_embed(torch.tensor(0, device=dev)).view(1, 1, -1)
        p4_sel = p4_sel + self.scale_embed(torch.tensor(1, device=dev)).view(1, 1, -1)
        p5_sel = p5_sel + self.scale_embed(torch.tensor(2, device=dev)).view(1, 1, -1)

        # Concatenate cross-scale tokens
        all_tok = torch.cat([p3_sel, p4_sel, p5_sel], dim=1)  # (B, k3+k4+k5, d)

        # Self-attention in FP32
        norm_w = self.norm.weight.float() if self.norm.weight is not None else None
        norm_b = self.norm.bias.float() if self.norm.bias is not None else None
        q = F.layer_norm(all_tok.float(), self.norm.normalized_shape, norm_w, norm_b, self.norm.eps)
        k = q
        v = q

        attn = torch.bmm(q, k.transpose(1, 2)) * self._attn_scale
        attn = attn.clamp(-80.0, 80.0)
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attended = _finite_guard(torch.bmm(attn, v))  # (B, K, d)

        # Split back by scale
        p3_att = attended[:, :k3, :]
        p4_att = attended[:, k3:k3 + k4, :]
        p5_att = attended[:, k3 + k4:, :]

        # Scatter back to spatial grids
        p3_delta = self._scatter(p3_att, p3_idx, B, H3, W3, P3.dtype)
        p4_delta = self._scatter(p4_att, p4_idx, B, H4, W4, P4.dtype)
        p5_delta = self._scatter(p5_att, p5_idx, B, H5, W5, P5.dtype)

        # Output projection + delta norm (AMP-safe)
        p3_delta = self.out_proj_p3(_cast_to_param_dtype(p3_delta, self.out_proj_p3))
        p4_delta = self.out_proj_p4(_cast_to_param_dtype(p4_delta, self.out_proj_p4))
        p5_delta = self.out_proj_p5(_cast_to_param_dtype(p5_delta, self.out_proj_p5))

        p3_delta = self.delta_norm_p3(p3_delta.to(dtype=self.delta_norm_p3.weight.dtype))
        p4_delta = self.delta_norm_p4(p4_delta.to(dtype=self.delta_norm_p4.weight.dtype))
        p5_delta = self.delta_norm_p5(p5_delta.to(dtype=self.delta_norm_p5.weight.dtype))

        # Gated residual on ORIGINAL inputs (not pre-normed)
        p3_out = P3 + torch.sigmoid(self.gate_p3) * p3_delta.to(dtype=P3.dtype)
        p4_out = P4 + torch.sigmoid(self.gate_p4) * p4_delta.to(dtype=P4.dtype)
        p5_out = P5 + torch.sigmoid(self.gate_p5) * p5_delta.to(dtype=P5.dtype)

        return [p3_out, p4_out, p5_out]

    def _scatter(
        self,
        attended: torch.Tensor,   # (B, k, d)
        indices: torch.Tensor,    # (B, k)
        B: int,
        H: int,
        W: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        N = H * W
        d = self.shared_dim
        canvas = torch.zeros(B, d, N, device=attended.device, dtype=dtype)
        idx_exp = indices.unsqueeze(1).expand(-1, d, -1)          # (B, d, k)
        canvas.scatter_(2, idx_exp, attended.transpose(1, 2).to(dtype=dtype))
        return canvas.view(B, d, H, W)
