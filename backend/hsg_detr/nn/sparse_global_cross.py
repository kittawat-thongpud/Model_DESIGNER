"""
Cross-Scale Sparse Global Attention (CS²GA) for HSG-DETR.

Replaces per-scale SparseGlobalTokenBlock with a single module that receives
P3/P4/P5 simultaneously, projects to shared dim, selects top-k tokens per scale,
adds learned scale embeddings, runs cross-scale self-attention, and scatters back
with LayerScale residuals.

Residual design — LayerScale (DeiT/CaiT pattern):
    out = input + layer_scale * delta
    layer_scale: per-channel Parameter(C, 1, 1), init=1e-4
    Starts near-identity, grows freely with no sigmoid saturation.
    Fixes the chicken-and-egg collapse observed with sigmoid gates where
    gate → 0 because delta is noisy early → gradient stops → delta never learns.

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


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _score_guard(x: torch.Tensor) -> torch.Tensor:
    """Guard saliency scores without destroying their rank ordering."""
    max_val = torch.finfo(x.dtype).max if x.is_floating_point() else 1e20
    return torch.nan_to_num(x, nan=0.0, posinf=max_val, neginf=0.0).clamp_min(0.0)


_FP16_SAFE = 60000.0


class CrossScaleSGA(nn.Module):
    """
    Cross-Scale Sparse Global Attention.

    Receives P3/P4/P5 feature maps simultaneously, projects to a shared
    dimension, selects top-k salient tokens per scale, adds learned scale
    embeddings, runs joint cross-scale self-attention, then scatters back
    with LayerScale residuals.

    Args:
        c1: list of input channels [c_p3, c_p4, c_p5] (injected by parse_model).
        c2: list of output channels (same as c1, channel-preserving).
        shared_dim: Projection dimension for cross-scale attention.
        ratio_p3: Top-k ratio for P3 tokens.
        ratio_p4: Top-k ratio for P4 tokens.
        ratio_p5: Top-k ratio for P5 tokens.
        ls_init: LayerScale init value. Default 1e-4 (DeiT standard).
        debug: Enable debug state recording each forward pass.
    """

    def __init__(
        self,
        c1: list[int],
        c2: list[int],
        shared_dim: int = 256,
        ratio_p3: float = 0.05,
        ratio_p4: float = 0.10,
        ratio_p5: float = 0.25,
        ls_init: float = 1e-4,
        debug: bool = False,
        scale_embed_alpha: float = 0.0,
        attn_scale_mult: float = 8.0,
        max_k3: int = 0,
        max_k4: int = 0,
        max_k5: int = 0,
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
        self.ls_init = float(ls_init)
        self.debug_enabled: bool = bool(debug)
        self.scale_embed_alpha = float(scale_embed_alpha)
        self.attn_scale_mult = float(attn_scale_mult)
        self.max_k3 = int(max_k3) if max_k3 is not None else 0
        self.max_k4 = int(max_k4) if max_k4 is not None else 0
        self.max_k5 = int(max_k5) if max_k5 is not None else 0

        # Debug state — updated each forward pass when debug_enabled=True
        self.last_ls_p3: float | None = None    # mean |LayerScale| for P3
        self.last_ls_p4: float | None = None
        self.last_ls_p5: float | None = None
        self.last_k3: int | None = None
        self.last_k4: int | None = None
        self.last_k5: int | None = None
        self.last_attn_within_frac: float | None = None
        self.last_attn_cross_frac: float | None = None
        self.last_delta_abs_p3: float | None = None
        self.last_delta_abs_p4: float | None = None
        self.last_delta_abs_p5: float | None = None
        self.last_p3_score_max: float | None = None
        self.last_p4_score_max: float | None = None
        self.last_p5_score_max: float | None = None
        self.last_p3_score_min_selected: float | None = None
        self.last_p4_score_min_selected: float | None = None
        self.last_p5_score_min_selected: float | None = None
        self.last_p3_score_std: float | None = None
        self.last_p4_score_std: float | None = None
        self.last_p5_score_std: float | None = None
        self.last_attn_entropy: float | None = None

        d = self.shared_dim

        # Project each scale to shared dimension
        self.proj_p3 = nn.Conv2d(self.c1[0], d, 1, bias=False)
        self.proj_p4 = nn.Conv2d(self.c1[1], d, 1, bias=False)
        self.proj_p5 = nn.Conv2d(self.c1[2], d, 1, bias=False)

        # Scale identity embeddings (0=P3, 1=P4, 2=P5)
        self.scale_embed = nn.Embedding(3, d)
        nn.init.zeros_(self.scale_embed.weight)

        # Attention norm
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

        # LayerScale — per-channel (C, 1, 1), init=ls_init ≈ 1e-4
        # Grows freely without sigmoid saturation. Gradient flows proportionally
        # to ls value itself — small but non-zero from epoch 1.
        self.ls_p3 = nn.Parameter(torch.full((self.c1[0], 1, 1), ls_init))
        self.ls_p4 = nn.Parameter(torch.full((self.c1[1], 1, 1), ls_init))
        self.ls_p5 = nn.Parameter(torch.full((self.c1[2], 1, 1), ls_init))

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
        input_dtype = P3.dtype

        with _fp32_context(P3.device):

            p3_f = P3.float()
            p4_f = P4.float()
            p5_f = P5.float()

            # Pre-norm
            p3_n = self.pre_norm_p3(p3_f.to(dtype=self.pre_norm_p3.weight.dtype)).float()
            p4_n = self.pre_norm_p4(p4_f.to(dtype=self.pre_norm_p4.weight.dtype)).float()
            p5_n = self.pre_norm_p5(p5_f.to(dtype=self.pre_norm_p5.weight.dtype)).float()

            # 1×1 projection to shared_dim
            p3_proj = F.conv2d(p3_n, self.proj_p3.weight.float())
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

            # Top-k by L2 saliency. Do not use _nan_guard here: clamping all
            # high-energy scores to 20 destroys rank ordering and makes top-k
            # effectively arbitrary on activated feature maps.
            with torch.no_grad():
                p3_scores = _score_guard(p3_tok.float().norm(dim=-1))
                p4_scores = _score_guard(p4_tok.float().norm(dim=-1))
                p5_scores = _score_guard(p5_tok.float().norm(dim=-1))

            k3_raw = max(1, min(int(self.ratio_p3 * p3_tok.shape[1]), p3_tok.shape[1]))
            k4_raw = max(1, min(int(self.ratio_p4 * p4_tok.shape[1]), p4_tok.shape[1]))
            k5_raw = max(1, min(int(self.ratio_p5 * p5_tok.shape[1]), p5_tok.shape[1]))
            k3 = min(k3_raw, self.max_k3) if self.max_k3 > 0 else k3_raw
            k4 = min(k4_raw, self.max_k4) if self.max_k4 > 0 else k4_raw
            k5 = min(k5_raw, self.max_k5) if self.max_k5 > 0 else k5_raw
            k3 = max(1, k3)
            k4 = max(1, k4)
            k5 = max(1, k5)

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

            # Concat cross-scale tokens (B, K, d)
            all_tok = torch.cat([p3_sel, p4_sel, p5_sel], dim=1)

            # Normalize selected token vectors before joint attention. Without
            # this, scale-specific activation magnitude can dominate the dot
            # products and collapse CS²GA into three near-independent within-
            # scale attention blocks.
            all_tok_attn = F.normalize(all_tok, dim=-1, eps=1e-6)

            # LayerNorm → Q/K; V uses un-normed tokens
            # Cosine attention: normalize Q and K after LayerNorm so logit range
            # stays bounded by _attn_scale regardless of how LayerNorm weights
            # grow during training.  Without this, LayerNorm γ growth gradually
            # sharpens attention → within-scale tokens (more similar to each
            # other) accumulate weight → cross-scale mixing collapses.
            norm_w = self.norm.weight.float()
            norm_b = self.norm.bias.float()
            q = F.layer_norm(all_tok_attn, self.norm.normalized_shape, norm_w, norm_b, self.norm.eps)
            q = _nan_guard(q)
            q = F.normalize(q, dim=-1, eps=1e-6)   # cosine Q/K — logits ∈ [±_attn_scale]
            k_t = q
            v = all_tok

            # Scaled dot-product attention
            attn = torch.bmm(q, k_t.transpose(1, 2)) * self._attn_scale
            attn = attn.clamp(-80.0, 80.0)
            attn = attn - attn.max(dim=-1, keepdim=True).values
            attn = torch.softmax(attn, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)

            attended = torch.bmm(attn, v)
            attended = _nan_guard(attended)

            # Split back by scale
            p3_att = attended[:, :k3, :]
            p4_att = attended[:, k3:k3 + k4, :]
            p5_att = attended[:, k3 + k4:, :]

            # Scatter back to spatial grids
            p3_delta = self._scatter(p3_att, p3_idx, B, d, H3, W3)
            p4_delta = self._scatter(p4_att, p4_idx, B, d, H4, W4)
            p5_delta = self._scatter(p5_att, p5_idx, B, d, H5, W5)
            p3_mask = self._scatter_mask(p3_idx, B, H3, W3)
            p4_mask = self._scatter_mask(p4_idx, B, H4, W4)
            p5_mask = self._scatter_mask(p5_idx, B, H5, W5)

            # tanh clamp before out_proj
            p3_delta = 6.0 * torch.tanh(p3_delta / 6.0)
            p4_delta = 6.0 * torch.tanh(p4_delta / 6.0)
            p5_delta = 6.0 * torch.tanh(p5_delta / 6.0)

            # out_proj
            p3_delta = F.conv2d(p3_delta, self.out_proj_p3.weight.float())
            p4_delta = F.conv2d(p4_delta, self.out_proj_p4.weight.float())
            p5_delta = F.conv2d(p5_delta, self.out_proj_p5.weight.float())

            # delta norm
            # GroupNorm can turn zero canvas locations non-zero; mask again so
            # the sparse branch remains sparse by contract.
            p3_delta = self._gn_fp32(p3_delta, self.delta_norm_p3) * p3_mask
            p4_delta = self._gn_fp32(p4_delta, self.delta_norm_p4) * p4_mask
            p5_delta = self._gn_fp32(p5_delta, self.delta_norm_p5) * p5_mask

            # LayerScale residual: out = input + ls * delta
            # ls_p* shape (C, 1, 1) broadcasts over (B, C, H, W)
            ls3 = self.ls_p3.float()
            ls4 = self.ls_p4.float()
            ls5 = self.ls_p5.float()

            if self.debug_enabled:
                self.last_ls_p3 = float(ls3.abs().mean().item())
                self.last_ls_p4 = float(ls4.abs().mean().item())
                self.last_ls_p5 = float(ls5.abs().mean().item())
                self.last_k3 = int(k3)
                self.last_k4 = int(k4)
                self.last_k5 = int(k5)
                with torch.no_grad():
                    _a = attn.detach().mean(0)
                    _within = (
                        _a[:k3, :k3].sum()
                        + _a[k3:k3 + k4, k3:k3 + k4].sum()
                        + _a[k3 + k4:, k3 + k4:].sum()
                    )
                    _total = float(_a.sum().clamp(min=1e-8))
                    self.last_attn_within_frac = float(_within / _total)
                    self.last_attn_cross_frac = float(1.0 - self.last_attn_within_frac)
                    _attn_prob = attn.detach().float().clamp_min(1e-12)
                    self.last_attn_entropy = float((-_attn_prob * _attn_prob.log()).sum(dim=-1).mean().item())
                    self.last_p3_score_max = float(p3_scores.detach().amax().item())
                    self.last_p4_score_max = float(p4_scores.detach().amax().item())
                    self.last_p5_score_max = float(p5_scores.detach().amax().item())
                    self.last_p3_score_min_selected = float(torch.gather(p3_scores, 1, p3_idx).amin().item())
                    self.last_p4_score_min_selected = float(torch.gather(p4_scores, 1, p4_idx).amin().item())
                    self.last_p5_score_min_selected = float(torch.gather(p5_scores, 1, p5_idx).amin().item())
                    self.last_p3_score_std = float(p3_scores.detach().std(unbiased=False).item())
                    self.last_p4_score_std = float(p4_scores.detach().std(unbiased=False).item())
                    self.last_p5_score_std = float(p5_scores.detach().std(unbiased=False).item())
                self.last_delta_abs_p3 = float(p3_delta.detach().abs().mean().item())
                self.last_delta_abs_p4 = float(p4_delta.detach().abs().mean().item())
                self.last_delta_abs_p5 = float(p5_delta.detach().abs().mean().item())

            p3_out = p3_f + ls3 * p3_delta
            p4_out = p4_f + ls4 * p4_delta
            p5_out = p5_f + ls5 * p5_delta

            # Clamp before FP16 cast to prevent overflow
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
        attended: torch.Tensor,
        indices: torch.Tensor,
        B: int, d: int, H: int, W: int,
    ) -> torch.Tensor:
        """Scatter attended tokens back to (B, d, H, W) canvas."""
        N = H * W
        canvas = torch.zeros(B, d, N, device=attended.device, dtype=torch.float32)
        idx_exp = indices.unsqueeze(1).expand(-1, d, -1)
        canvas.scatter_(2, idx_exp, attended.transpose(1, 2))
        return canvas.view(B, d, H, W)

    @staticmethod
    def _scatter_mask(indices: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
        """Scatter selected indices into a (B, 1, H, W) FP32 binary mask."""
        N = H * W
        mask = torch.zeros(B, 1, N, device=indices.device, dtype=torch.float32)
        mask.scatter_(2, indices.unsqueeze(1), 1.0)
        return mask.view(B, 1, H, W)

    @staticmethod
    def _gn_fp32(x: torch.Tensor, gn: nn.GroupNorm) -> torch.Tensor:
        """GroupNorm in FP32 regardless of weight storage dtype."""
        w = gn.weight.float() if gn.weight is not None else None
        b = gn.bias.float() if gn.bias is not None else None
        return F.group_norm(x, gn.num_groups, w, b, gn.eps)

    # ------------------------------------------------------------------ #

    def set_debug(self, enabled: bool = True) -> None:
        self.debug_enabled = enabled

    def get_debug_state(self) -> dict:
        """Return last recorded debug values (None if debug_enabled=False or no forward yet)."""
        return {
            "ls_p3": self.last_ls_p3,
            "ls_p4": self.last_ls_p4,
            "ls_p5": self.last_ls_p5,
            "k3": self.last_k3,
            "k4": self.last_k4,
            "k5": self.last_k5,
            "attn_within_frac": self.last_attn_within_frac,
            "attn_cross_frac": self.last_attn_cross_frac,
            "delta_abs_p3": self.last_delta_abs_p3,
            "delta_abs_p4": self.last_delta_abs_p4,
            "delta_abs_p5": self.last_delta_abs_p5,
            "p3_score_max": self.last_p3_score_max,
            "p4_score_max": self.last_p4_score_max,
            "p5_score_max": self.last_p5_score_max,
            "p3_score_min_selected": self.last_p3_score_min_selected,
            "p4_score_min_selected": self.last_p4_score_min_selected,
            "p5_score_min_selected": self.last_p5_score_min_selected,
            "p3_score_std": self.last_p3_score_std,
            "p4_score_std": self.last_p4_score_std,
            "p5_score_std": self.last_p5_score_std,
            "attn_entropy": self.last_attn_entropy,
        }
