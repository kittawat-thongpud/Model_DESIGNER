"""
SparseGlobalTokenBlock — SGB-centric sparse-token encoder for HSG-DETR.

Single tensor-in / tensor-out block with internal top-k selector, compatible
with Ultralytics ``parse_model``.  Supports three ablation modes:
  - ``dense``  : full self-attention (O(N²d)) — ablation baseline
  - ``topk``   : hard top-k sparse attention (O(k²d)) — production
  - ``hybrid`` : dense local + sparse global fusion — accuracy

        Metadata (selected indices, saliency scores, residual scale) are stored as
instance attributes for debug / visualization without breaking ``parse_model``
(single tensor return).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import RTDETRDecoder


def _finite_or_zero(x: torch.Tensor, limit: float = 20.0) -> torch.Tensor:
    """Clamp tensor to finite range and replace NaN/Inf with 0."""
    return torch.nan_to_num(
        x,
        nan=0.0,
        posinf=limit,
        neginf=-limit,
    ).clamp(-limit, limit)


def _make_gn(channels: int) -> nn.GroupNorm:
    """Return a GroupNorm layer without BatchNorm running buffers."""
    groups = min(32, int(channels))
    while int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels), eps=1e-5)


class SGTokenBlock(nn.Module):
    """
    Sparse-Token Global Self-Attention block.

    Internally performs top-k token selection, sparse self-attention,
    scatter-back, and gated residual fusion.  Returns a single tensor
    so it is transparent to Ultralytics ``parse_model``.

    Args:
        c1 (int): Input channels (auto-injected by parse_model).
        c2 (int): Output channels (auto-injected by parse_model from YAML args[0]).
        ratio (float): Fraction of spatial tokens to retain; k = ratio·N (scales with imgsz).
        mode (str): One of ``"dense"``, ``"topk"``, ``"hybrid"``.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        ratio: float = 0.25,
        mode: str = "topk",
    ) -> None:
        super().__init__()
        assert c1 == c2, (
            f"SparseGlobalTokenBlock is channel-preserving (c1={c1}, c2={c2})"
        )
        self.c = c2
        self.ratio = float(ratio)
        self.mode = mode

        self.pre_norm = _make_gn(c2)
        self.delta_norm = _make_gn(c2)

        # 1×1 projections — no spatial aggregation, just channel mixing
        self.q_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.k_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.v_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.out_proj = nn.Conv2d(c2, c2, 1, bias=False)

        # Layer norm on selected tokens (channel dim)
        self.norm = nn.LayerNorm(c2)

        # Per-channel LayerScale residual.  Starting near identity keeps the
        # CNN/PAN path dominant while the sparse branch calibrates.
        self.gamma = nn.Parameter(torch.full((1, c2, 1, 1), 1e-4))

        self._attn_scale = c2 ** -0.5

        # ── Hybrid mode: lightweight local path ───────────────────────────
        if mode == "hybrid":
            self.local_dw = nn.Sequential(
                nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
                _make_gn(c2),
                nn.SiLU(),
                nn.Conv2d(c2, c2, 1, bias=False),
                _make_gn(c2),
            )
            self.local_gamma = nn.Parameter(torch.full((1, c2, 1, 1), 1e-4))
        else:
            self.local_dw = None
            self.local_gamma = None

        # Debug metadata (populated during forward, accessible via get_debug_state)
        self.last_indices: torch.Tensor | None = None
        self.last_saliency: torch.Tensor | None = None
        self.last_gate: float | None = None
        self.last_mode: str | None = None
        self.last_k: int | None = None
        self.last_N: int | None = None

    # ------------------------------------------------------------------ #
    # Debug / introspection
    # ------------------------------------------------------------------ #

    def get_debug_state(self) -> dict:
        """Return non-gradient metadata from the last forward pass."""
        return {
            "indices": self.last_indices,
            "saliency": self.last_saliency,
            "gate": self.last_gate,
            "mode": self.last_mode,
            "k": self.last_k,
            "N": self.last_N,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_saliency(self, x: torch.Tensor) -> torch.Tensor:
        """L2 activation energy per spatial token, in FP32."""
        B, C, H, W = x.shape
        N = H * W
        importance = x.view(B, C, N).float().pow(2).sum(dim=1)  # [B, N]
        return torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)

    def _select_k(self, N: int) -> int:
        """Compute effective k based on mode and constraints."""
        if self.mode == "dense":
            return int(N)
        return max(1, min(int(float(self.ratio) * int(N)), int(N)))

    def _sparse_attention_delta(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        kk: torch.Tensor,
        v: torch.Tensor,
        topk_idx: torch.Tensor,
        k_actual: int,
    ) -> torch.Tensor:
        """
        Run self-attention on selected tokens and scatter back.

        All attention math in FP32 for AMP stability.
        """
        B, C, H, W = x.shape
        N = H * W
        idx_exp = topk_idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, k]

        q_sel = torch.gather(q, 2, idx_exp).transpose(1, 2)   # [B, k, C]
        k_sel = torch.gather(kk, 2, idx_exp).transpose(1, 2)  # [B, k, C]
        v_sel = torch.gather(v, 2, idx_exp).transpose(1, 2)   # [B, k, C]

        # Layer norm in FP32
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
        v_sel = F.layer_norm(
            v_sel.float(),
            self.norm.normalized_shape,
            norm_w.float() if norm_w is not None else None,
            norm_b.float() if norm_b is not None else None,
            self.norm.eps,
        )

        # Sparse self-attention (stable softmax)
        attn = torch.bmm(q_sel, k_sel.transpose(1, 2)) * float(self._attn_scale)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = attn.clamp(min=-80.0, max=80.0)
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attended = torch.bmm(attn, v_sel)  # [B, k, C]
        attended = torch.nan_to_num(attended, nan=0.0, posinf=0.0, neginf=0.0)
        attended = attended.to(orig_dtype)

        # Scatter back to spatial grid
        out = v.clone().view(B, C, N)  # [B, C, N]

        # Guard: sanitize attended before scatter to prevent NaN propagation
        attended_clean = torch.nan_to_num(attended.transpose(1, 2), nan=0.0, posinf=0.0, neginf=0.0)

        # Guard: ensure idx_exp is valid for scatter
        idx_exp = torch.clamp(idx_exp, 0, N - 1)
        idx_exp = torch.nan_to_num(idx_exp, nan=0).long()

        out.scatter_(2, idx_exp, attended_clean)
        out = out.view(B, C, H, W)
        delta = self.out_proj(out)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        delta = 6.0 * torch.tanh(delta.float() / 6.0)
        delta = self.delta_norm(delta).to(orig_dtype)
        return delta

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = int(H) * int(W)

        k_actual = self._select_k(N)
        self.last_N = N
        self.last_k = k_actual
        self.last_mode = self.mode

        x_branch = self.pre_norm(x.float()).to(dtype=x.dtype)

        # Projections
        q = self.q_proj(x_branch).view(B, C, N)   # [B, C, N]
        kk = self.k_proj(x_branch).view(B, C, N)  # [B, C, N]
        v = self.v_proj(x_branch).view(B, C, N)   # [B, C, N]

        gamma = self.gamma.to(dtype=x.dtype)
        self.last_gate = float(gamma.detach().abs().mean().item())

        if self.mode == "dense":
            # Full self-attention on all tokens (ablation baseline)
            # Use all positions as "selected"
            topk_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            self.last_indices = topk_idx
            self.last_saliency = self._compute_saliency(x_branch)
            delta = self._sparse_attention_delta(x_branch, q, kk, v, topk_idx, N)
            return x + gamma * delta

        # ── topk / hybrid: select salient tokens ───────────────────────────────────
        importance = self._compute_saliency(x_branch)
        self.last_saliency = importance

        # Guard: if all importance is NaN, fall back to uniform selection
        if not torch.isfinite(importance).any():
            importance = torch.ones_like(importance)

        topk_idx = torch.topk(importance, k_actual, dim=1).indices  # [B, k]

        # Guard: validate indices are in valid range [0, N)
        topk_idx = torch.clamp(topk_idx, 0, N - 1)
        topk_idx = torch.nan_to_num(topk_idx, nan=0).long()

        self.last_indices = topk_idx

        delta = self._sparse_attention_delta(x_branch, q, kk, v, topk_idx, k_actual)

        if self.mode == "hybrid" and self.local_dw is not None:
            local_delta = self.local_dw(x_branch)
            local_gamma = self.local_gamma.to(dtype=x.dtype)
            return x + gamma * delta + local_gamma * local_delta

        # Standard topk mode
        return x + gamma * delta


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Backbone Blocks
# ─────────────────────────────────────────────────────────────────────────────


# Backward-compat alias: old checkpoints pickled the class under the old name
# before the rename. Keep so torch.load can unpickle legacy .pt files.
SparseGlobalTokenBlock = SGTokenBlock


class SGStem(nn.Module):
    """
    Sparse-Global stem with two-stage downsampling and a depthwise intermediate
    for spatial detail preservation.

    Replaces the first two ``Conv(stride=2)`` layers in a standard YOLO
    backbone (P1/2 + P2/4 → P2/4 in one block).

    Architecture::

        Conv(c1→c2/4, k=3, s=2)          # 2× downsample
        → DWConv(c2/4→c2/4, k=3, s=1)   # depthwise, preserve structure
        → Conv(c2/4→c2/2, k=1, s=1)      # pointwise expansion
        → Conv(c2/2→c2, k=3, s=2)        # 2× downsample

    Args:
        c1 (int): Input channels (auto-injected by parse_model).
        c2 (int): Output channels (auto-injected by parse_model).
    """

    def __init__(self, c1: int, c2: int, k: int = 3) -> None:
        super().__init__()
        mid = c2 // 4
        mid2 = c2 // 2

        # Stage 1: 2× downsample + detail preservation
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, mid, k, stride=2, padding=k // 2, bias=False),
            _make_gn(mid),
            nn.SiLU(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(mid, mid, k, stride=1, padding=k // 2, groups=mid, bias=False),
            _make_gn(mid),
            nn.SiLU(),
        )
        # Stage 2: channel expansion + 2× downsample
        self.cv3 = nn.Sequential(
            nn.Conv2d(mid, mid2, 1, bias=False),
            _make_gn(mid2),
            nn.SiLU(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(mid2, c2, k, stride=2, padding=k // 2, bias=False),
            _make_gn(c2),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        return self.cv4(x)


class SGDown(nn.Module):
    """
    Clue-preserving downsampling block.

    Separates channel alignment (1×1) from spatial downsampling (3×3 stride-2)
    so saliency cues are enriched before spatial resolution is reduced.
    This helps the downstream ``SparseGlobalTokenBlock`` selector retain
    small-object signals.

    Architecture::

        Conv(c1→c2, k=1, s=1)   # channel alignment, no spatial blur
        → Conv(c2→c2, k=3, s=2) # stride-2 downsampling

    Args:
        c1 (int): Input channels (auto-injected by parse_model).
        c2 (int): Output channels (auto-injected by parse_model).
    """

    def __init__(self, c1: int, c2: int, k: int = 3) -> None:
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            _make_gn(c2),
            nn.SiLU(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c2, c2, k, stride=2, padding=k // 2, bias=False),
            _make_gn(c2),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3c: SGB-Guided RT-DETR Decoder
# ─────────────────────────────────────────────────────────────────────────────


class RTDETRDecoderSGB(RTDETRDecoder):
    """
    SGB-guided RT-DETR decoder.

    Inherits the full RT-DETR decoder pipeline and overrides query selection
    in ``_get_decoder_input`` to combine classification score with **token
    saliency** (L2 activation energy) from the encoder-projected features.

    When ``alpha=0`` the behaviour is identical to the base RTDETRDecoder.
    As ``alpha`` increases, spatially salient tokens receive higher priority
    during top-k query selection, aligning the decoder's initial queries with
    the regions the SGB encoder has already identified as important.

    Args (same as RTDETRDecoder + alpha):
        nc, ch, hd, nq, ndp, nh, ndl, d_ffn, dropout, act, eval_idx,
        nd, label_noise_ratio, box_noise_scale, learnt_init_query
        alpha (float): Cold-start saliency weighting coefficient, bounded by 0.5.
    """

    ALPHA_MAX: float = 0.5  # maximum saliency weighting

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = True,
    ) -> None:
        super().__init__(
            nc, ch, hd, nq, ndp, nh, ndl, d_ffn, dropout, act,
            eval_idx, nd, label_noise_ratio, box_noise_scale, learnt_init_query,
        )
        self.register_buffer("alpha_logit", torch.tensor(-6.0), persistent=True)

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Saliency-weighted query selection.

        Combines the standard IoU-aware classification score with token
        activation energy so queries are biased toward spatially salient
        regions identified by the upstream SGB encoder.
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(
                shapes, dtype=feats.dtype, device=feats.device
            )
            self.shapes = shapes

        # Encoder projection (same as base). Keep these tensors unsanitized for
        # the loss path; only guarded copies are used for top-k selection.
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, hd
        enc_outputs_scores = self.enc_score_head(features)     # bs, h*w, nc

        # ── Saliency-weighted query selection ───────────────────────────
        score_for_select = _finite_or_zero(enc_outputs_scores.detach(), limit=20.0)
        cls_score = score_for_select.float().max(-1).values          # (bs, h*w)

        feat32 = _finite_or_zero(features.detach(), limit=20.0).float()
        token_energy = feat32.square().sum(-1)
        token_energy = torch.nan_to_num(token_energy, nan=0.0, posinf=1e4, neginf=0.0)

        # Per-sample min-max normalisation with NaN/Inf protection
        eps = 1e-6
        cls_min = cls_score.min(1, keepdim=True).values
        cls_max = cls_score.max(1, keepdim=True).values
        cls_range = (cls_max - cls_min).clamp(min=eps)
        cls_score_norm = ((cls_score - cls_min) / cls_range).clamp(-10.0, 10.0)

        energy_min = token_energy.min(1, keepdim=True).values
        energy_max = token_energy.max(1, keepdim=True).values
        energy_range = (energy_max - energy_min).clamp(min=eps)
        energy_norm = ((token_energy - energy_min) / energy_range).clamp(-10.0, 10.0)

        alpha = float(self.ALPHA_MAX) * torch.sigmoid(self.alpha_logit.float())
        combined = (cls_score_norm + alpha * energy_norm).clamp(-20.0, 20.0)
        combined = torch.nan_to_num(combined, nan=-1e4, posinf=20.0, neginf=-1e4)

        # Mask invalid anchors before topk
        valid = self.valid_mask.squeeze(-1).bool()  # [1, N]
        if valid.shape[0] == 1 and bs > 1:
            valid = valid.expand(bs, -1)
        combined = combined.masked_fill(~valid, torch.finfo(combined.dtype).min)

        topk_ind = torch.topk(combined, self.num_queries, dim=1).indices.view(-1)

        # Guard: validate topk indices before using for indexing
        N_features = features.shape[1]  # h*w
        topk_ind = torch.clamp(topk_ind, 0, N_features - 1)
        topk_ind = torch.nan_to_num(topk_ind, nan=0).long()

        # ── Remainder identical to base RTDETRDecoder ──────────────────
        batch_ind = (
            torch.arange(end=bs, dtype=topk_ind.dtype)
            .unsqueeze(-1)
            .repeat(1, self.num_queries)
            .view(-1)
        )

        # Guard: ensure batch_ind is also valid
        batch_ind = torch.clamp(batch_ind, 0, bs - 1)

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = (
            self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            if self.learnt_init_query
            else top_k_features
        )
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores
