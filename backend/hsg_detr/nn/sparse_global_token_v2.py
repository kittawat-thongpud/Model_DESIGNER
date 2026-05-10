"""
HSG-DETR V2 sparse-token encoder + decoder.

Changes from Legacy (sparse_global_token.py):
  AMP safety  — all GroupNorm/Conv inputs cast via .to(dtype=weight.dtype)
                 LayerNorm numerical island uses explicit .float() on both
                 input AND weights, never relies on autocast dtype inference.
  T1-A LFT    — Look-Forward-Twice: decoder box refinement is no longer
                 detached between layers; gradient flows across all 6 layers
                 so each layer's bbox-head receives richer signal.
                 !! Requires clip_grad_norm ≤ 0.1 in training config !!
  T2-A U-min  — Uncertainty-minimal query selection replaces the legacy
                 alpha·energy + cls heuristic.  Selects the top-K encoder
                 features where localization and classification are jointly
                 consistent: U(X) = |cls_norm(X) - loc_norm(X)|.
                 Features with low U are the best decoder anchors.
  T2-B stats  — SGTokenBlockV2.forward() logs last_score_std (σ of L2
                 attention scores) so the token ratio k/n can be tuned with
                 the formal bound:  k_ε/n ≈ Φ_c(σ + Φ⁻¹(ε))
                 [Top-k Theory, arXiv:2512.07647, Theorem 6.4]

Architecture topology (identical to Legacy so V1/Legacy weights load):
  SGTokenBlockV2 : Conv2d Q/K/V/out (1×1), GroupNorm pre/delta, LayerScale γ
  RTDETRDecoderV2: inherits RTDETRDecoder, overrides _get_decoder_input +
                   _decoder_forward_lft
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from ultralytics.nn.modules.head import RTDETRDecoder


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_gn(channels: int) -> nn.GroupNorm:
    groups = min(32, int(channels))
    while int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels), eps=1e-5)


def _safe_unit_interval(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return x.clamp(min=eps, max=1.0 - eps)


def _safe_inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = _safe_unit_interval(x, eps=eps)
    return torch.log(x / (1.0 - x))


def _fp32_island(device: torch.device):
    """Disable AMP autocast for numerically sensitive FP32 ops."""
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _cast_to_param_dtype(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Cast x to match the first parameter dtype of module.  AMP-safe."""
    try:
        param = next(module.parameters())
        return x.to(dtype=param.dtype)
    except StopIteration:
        return x


def _finite_guard(x: torch.Tensor, limit: float = 20.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


# ─────────────────────────────────────────────────────────────────────────────
# SGTokenBlockV2
# ─────────────────────────────────────────────────────────────────────────────

class SGTokenBlockV2(nn.Module):
    """
    Sparse-Token Global Self-Attention block — V2 (AMP-safe redesign).

    Topology identical to Legacy so pretrained weights transfer cleanly.
    Fixes the GroupNorm mixed-dtype error that appeared during AMP validation
    by using `.to(dtype=weight.dtype)` everywhere instead of `.float()`.

    New debug field: last_score_std  — σ of L2 energy scores.
    Use it to compute the minimum k/n ratio via:
        k_ε/n ≈ Φ_c(σ + Φ⁻¹(ε))  [arXiv:2512.07647, Theorem 6.4]
    """

    VALID_MODES: set[str] = {"dense", "topk", "hybrid"}
    DENSE_TOKEN_LIMIT: int = 4096

    def __init__(
        self,
        c1: int,
        c2: int,
        ratio: float = 0.25,
        mode: str = "topk",
        debug_enabled: bool = False,
        gamma_init: float = 1e-4,
        channel_se: bool = False,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        assert c1 == c2, f"SGTokenBlockV2 is channel-preserving (c1={c1}, c2={c2})"
        mode = str(mode).lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}, got {mode!r}")
        ratio = float(ratio)
        assert torch.isfinite(torch.tensor(ratio)), f"Invalid ratio: {ratio}"

        self.c = int(c2)
        self.ratio = max(0.0, min(ratio, 1.0))
        self.mode = "topk" if mode == "hybrid" else mode  # hybrid → topk (no local path in V2)
        self.requested_mode = mode

        # Norms — GroupNorm for spatial feature, LayerNorm for selected tokens
        self.pre_norm = _make_gn(c2)
        self.delta_norm = _make_gn(c2)
        self.norm = nn.LayerNorm(c2)

        # 1×1 Conv2d projections (same as Legacy for weight compatibility)
        self.q_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.k_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.v_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.out_proj = nn.Conv2d(c2, c2, 1, bias=False)

        # Optional channel-selective SE gate on attention output. V2 keeps this
        # disabled by default; V2c enables it explicitly from YAML.
        self.channel_se = bool(channel_se)
        if self.channel_se:
            mid = max(1, int(c2) // int(se_reduction))
            self.se_fc1 = nn.Linear(c2, mid, bias=False)
            self.se_fc2 = nn.Linear(mid, c2, bias=False)

        # LayerScale — starts near zero so CNN path dominates early training
        self.gamma = nn.Parameter(torch.full((1, c2, 1, 1), float(gamma_init)))
        self._attn_scale = float(c2 ** -0.5)

        # Debug state (populated only when debug_enabled=True)
        self.debug_enabled = bool(debug_enabled)
        self.debug_to_cpu = False
        self.last_indices: torch.Tensor | None = None
        self.last_saliency: torch.Tensor | None = None
        self.last_gate: float | None = None
        self.last_mode: str | None = None
        self.last_k: int | None = None
        self.last_N: int | None = None
        self.last_score_std: float | None = None   # T2-B: σ for budget formula

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ensure_attrs(self) -> None:
        defaults = {
            "debug_enabled": False,
            "debug_to_cpu": False,
            "last_indices": None,
            "last_saliency": None,
            "last_gate": None,
            "last_mode": None,
            "last_k": None,
            "last_N": None,
            "last_score_std": None,
        }
        for k, v in defaults.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._ensure_attrs()

    def get_debug_state(self) -> dict:
        self._ensure_attrs()
        return {
            "indices": self.last_indices,
            "saliency": self.last_saliency,
            "gate": self.last_gate,
            "mode": self.last_mode,
            "k": self.last_k,
            "N": self.last_N,
            "score_std": self.last_score_std,
        }

    def set_debug(self, enabled: bool = True, cpu: bool = False) -> None:
        self._ensure_attrs()
        self.debug_enabled = bool(enabled)
        self.debug_to_cpu = bool(cpu)
        if not self.debug_enabled:
            self.last_indices = self.last_saliency = None

    def _store_debug(self, indices: torch.Tensor | None, saliency: torch.Tensor | None) -> None:
        self._ensure_attrs()
        if not self.debug_enabled:
            self.last_indices = self.last_saliency = None
            return
        if indices is not None:
            t = indices.detach()
            self.last_indices = t.cpu() if self.debug_to_cpu else t
        if saliency is not None:
            t = saliency.detach()
            self.last_saliency = t.cpu() if self.debug_to_cpu else t

    def _select_k(self, N: int) -> int:
        if self.mode == "dense":
            if N > self.DENSE_TOKEN_LIMIT:
                raise RuntimeError(f"Dense mode too large: N={N}")
            return N
        return max(1, min(int(round(self.ratio * N)), N))

    # ------------------------------------------------------------------ #
    # Core attention (FP32 island — numerically explicit)
    # ------------------------------------------------------------------ #

    def _sparse_attn_delta(
        self,
        q: torch.Tensor,   # (B, C, N)
        k: torch.Tensor,   # (B, C, N)
        v: torch.Tensor,   # (B, C, N)
        topk_idx: torch.Tensor,  # (B, k)
        B: int, C: int, H: int, W: int,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run selected-token self-attention. Returns delta in input_dtype."""
        N = H * W
        idx_exp = topk_idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, k)
        idx_exp = idx_exp.clamp(0, N - 1)

        q_sel = torch.gather(q, 2, idx_exp).transpose(1, 2)   # (B, k, C)
        k_sel = torch.gather(k, 2, idx_exp).transpose(1, 2)
        v_sel = torch.gather(v, 2, idx_exp).transpose(1, 2)

        # AMP-safe LayerNorm: both input and weights explicitly fp32
        norm_w = self.norm.weight.float() if self.norm.weight is not None else None
        norm_b = self.norm.bias.float() if self.norm.bias is not None else None
        q_f = F.layer_norm(q_sel.float(), self.norm.normalized_shape, norm_w, norm_b, self.norm.eps)
        k_f = F.layer_norm(k_sel.float(), self.norm.normalized_shape, norm_w, norm_b, self.norm.eps)
        v_f = F.layer_norm(v_sel.float(), self.norm.normalized_shape, norm_w, norm_b, self.norm.eps)

        # Stable softmax attention in FP32
        attn = torch.bmm(q_f, k_f.transpose(1, 2)) * self._attn_scale   # (B, k, k)
        attn = attn.clamp(-80.0, 80.0)
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attended = torch.bmm(attn, v_f)                                   # (B, k, C) fp32
        attended = 6.0 * torch.tanh(attended / 6.0)                       # tanh bound

        # Scatter back to zero canvas (non-selected remain 0)
        canvas = torch.zeros(B, C, N, device=q.device, dtype=input_dtype)
        attended_clean = torch.nan_to_num(attended.transpose(1, 2).to(dtype=input_dtype), nan=0.0)
        canvas.scatter_(2, idx_exp, attended_clean)
        canvas = canvas.view(B, C, H, W)

        # Output projection (AMP: cast to Conv weight dtype)
        delta = self.out_proj(_cast_to_param_dtype(canvas, self.out_proj))
        delta = _finite_guard(6.0 * torch.tanh(delta.float() / 6.0), limit=6.0)

        # GroupNorm on delta — AMP-safe: cast to GN weight dtype
        delta = self.delta_norm(delta.to(dtype=self.delta_norm.weight.dtype))

        # Mask: keep only selected positions (GroupNorm can pollute zeros)
        delta_mask = torch.zeros(B, 1, N, device=delta.device, dtype=delta.dtype)
        delta_mask.scatter_(2, topk_idx.unsqueeze(1), 1.0)
        delta = delta * delta_mask.view(B, 1, H, W)

        return delta.to(dtype=input_dtype)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_attrs()
        input_dtype = x.dtype
        B, C, H, W = x.shape
        N = H * W

        k_actual = self._select_k(N)
        self.last_N, self.last_k, self.last_mode = N, k_actual, self.mode
        gamma = self.gamma.to(dtype=input_dtype)
        self.last_gate = float(gamma.detach().abs().mean().item())

        # ── GroupNorm pre-norm: cast input to GN parameter dtype ─────────
        # Using .to(dtype=weight.dtype) — NOT .float() — is the AMP-safe rule.
        # .float() would give fp32 input when GN weights may be fp16, causing
        # the "mixed dtype" RuntimeError seen in validation.
        x_norm = self.pre_norm(x.to(dtype=self.pre_norm.weight.dtype))

        # ── Conv2d projections: cast to Conv weight dtype ─────────────────
        proj_dtype = self.q_proj.weight.dtype
        x_p = x_norm.to(dtype=proj_dtype)
        q = self.q_proj(x_p).view(B, C, N)
        k = self.k_proj(x_p).view(B, C, N)
        v = self.v_proj(x_p).view(B, C, N)

        # ── L2 saliency (parameter-free, detached) ────────────────────────
        with torch.no_grad():
            importance = x_norm.float().view(B, C, N).pow(2).sum(dim=1)  # (B, N)
            importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
            # T2-B: log σ for token budget formula k_ε/n ≈ Φ_c(σ + Φ⁻¹(ε))
            self.last_score_std = float(importance.std(dim=1).mean().item())

        if self.mode == "dense":
            topk_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            self._store_debug(topk_idx, None)
        else:
            if not torch.isfinite(importance).any():
                importance = torch.ones_like(importance)
            topk_idx = torch.topk(importance, k_actual, dim=1).indices
            topk_idx = topk_idx.clamp(0, N - 1).long()
            self._store_debug(topk_idx, importance)

        delta = self._sparse_attn_delta(q, k, v, topk_idx, B, C, H, W, input_dtype)

        # ── Channel-selective SE gate ─────────────────────────────────────
        # Squeeze: global avg of delta over spatial dims (only attended positions
        # are non-zero, so this is effectively avg over selected tokens).
        # Excitation: bottleneck FC → sigmoid → per-channel scale in (0,1).
        # Gate is AMP-safe: FC weights cast to input dtype.
        if self.channel_se:
            sq = delta.mean(dim=(2, 3))                             # (B, C)
            sq_f = sq.to(dtype=self.se_fc1.weight.dtype)
            gate = torch.relu(self.se_fc1(sq_f))                   # (B, mid)
            gate = torch.sigmoid(self.se_fc2(gate))                 # (B, C) ∈ (0,1)
            delta = delta * gate.to(dtype=input_dtype).view(B, C, 1, 1)

        return x + gamma * delta


# ─────────────────────────────────────────────────────────────────────────────
# RTDETRDecoderV2
# ─────────────────────────────────────────────────────────────────────────────

class RTDETRDecoderV2(RTDETRDecoder):
    """
    SGB-guided RT-DETR decoder — V2.

    T1-A Look-Forward-Twice (LFT):
        Gradient now flows through the box-refinement chain across all decoder
        layers.  Previously, `refer_bbox = refined_bbox.detach()` broke the
        chain so each layer's bbox-head learned independently.  With LFT
        removed, layer i+1's loss teaches layer i which anchor deltas produce
        good subsequent refinements.
        !! Set clip_grad_norm ≤ 0.1 in the training config !!
        Reference: DINO arXiv:2203.03605, Section 3 "Look Forward Twice".

    T2-A Uncertainty-Minimal Query Selection:
        Replaces legacy alpha·energy + cls with:
            U(X) = |cls_norm(X) - loc_norm(X)|   [RT-DETR, Eq. 2]
        where cls_norm = normalized max-class sigmoid score,
              loc_norm = normalized predicted box area (w·h proxy).
        Queries are selected by maximizing (cls_norm - alpha·U), so features
        that are simultaneously confident in class AND location win.
        Reference: RT-DETR arXiv:2304.08069, Section 3.2.
    """

    ALPHA_MAX: float = 0.5
    DN_LOGIT_LIMIT: float = 4.0
    REFINE_EPS: float = 1e-3
    BBOX_DELTA_LIMIT: float = 4.0

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
        self.register_buffer("alpha", torch.tensor(0.0), persistent=True)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._ensure_runtime_attrs()

    def _ensure_runtime_attrs(self) -> None:
        alpha = getattr(self, "alpha", None)
        if isinstance(alpha, torch.Tensor) and "alpha" in getattr(self, "_buffers", {}):
            return
        legacy_alpha_logit = getattr(self, "alpha_logit", None)
        if isinstance(legacy_alpha_logit, torch.Tensor):
            value = float(self.ALPHA_MAX) * torch.sigmoid(legacy_alpha_logit.detach().float())
            value = value.reshape(())
        elif isinstance(alpha, torch.Tensor):
            value = alpha.detach().float().reshape(())
        else:
            value = torch.tensor(0.0)
        if "alpha" in getattr(self, "_buffers", {}):
            self._buffers["alpha"] = value
        else:
            if hasattr(self, "alpha"):
                delattr(self, "alpha")
            self.register_buffer("alpha", value, persistent=True)

    def set_alpha(self, value: float) -> None:
        self._ensure_runtime_attrs()
        self.alpha.fill_(max(0.0, min(float(value), float(self.ALPHA_MAX))))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                               strict, missing_keys, unexpected_keys, error_msgs) -> None:
        # Accept checkpoints from RTDETRDecoderSGB (Legacy) — same param layout
        legacy_key = prefix + "alpha_logit"
        alpha_key = prefix + "alpha"
        if legacy_key in state_dict and alpha_key not in state_dict:
            val = float(self.ALPHA_MAX) * torch.sigmoid(state_dict[legacy_key].detach().float())
            state_dict[alpha_key] = val.reshape_as(self.alpha)
        if legacy_key in state_dict:
            del state_dict[legacy_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                       strict, missing_keys, unexpected_keys, error_msgs)

    # ------------------------------------------------------------------ #
    # T2-A Uncertainty-Minimal Query Selection
    # ------------------------------------------------------------------ #

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Uncertainty-minimal query selection (RT-DETR arXiv:2304.08069 Eq. 2-3).

        Selection score = cls_norm(X) - alpha · U(X)
        U(X) = |cls_norm(X) - loc_norm(X)|

        Intuition: a good anchor is high-confidence in both class (cls_norm)
        and location (loc_norm).  When they disagree (high U), the feature is
        internally contradictory — a poor decoder anchor regardless of its
        raw classification score.
        """
        self._ensure_runtime_attrs()
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(
                shapes, dtype=feats.dtype, device=feats.device
            )
            self.shapes = shapes

        features = self.enc_output(self.valid_mask * feats)   # (bs, hw, hd)
        enc_outputs_scores = self.enc_score_head(features)    # (bs, hw, nc)

        with torch.no_grad():
            # Classification confidence: max class sigmoid score
            cls_conf = enc_outputs_scores.detach().float().sigmoid().max(-1).values  # (bs, hw)

            # Localization confidence: predicted box area proxy (w·h) from enc_bbox_head
            # enc_bbox_head is Linear(hd→4); applying to all tokens costs hw·hd·4 FLOPs
            # which is negligible (for N-scale: 8400 × 128 × 4 ≈ 4M ops)
            box_pred = self.enc_bbox_head(features.detach().float()).sigmoid()  # (bs, hw, 4)
            loc_conf = (box_pred[..., 2] * box_pred[..., 3]).clamp(min=1e-6)   # w·h (bs, hw)

            eps = 1e-6
            # Normalize cls to [0,1]
            c_min = cls_conf.min(1, keepdim=True).values
            c_max = cls_conf.max(1, keepdim=True).values
            cls_n = (cls_conf - c_min) / (c_max - c_min).clamp(min=eps)

            # Normalize loc to [0,1]
            l_min = loc_conf.min(1, keepdim=True).values
            l_max = loc_conf.max(1, keepdim=True).values
            loc_n = (loc_conf - l_min) / (l_max - l_min).clamp(min=eps)

            # Uncertainty: discrepancy between cls and loc confidence
            uncertainty = (cls_n - loc_n).abs()   # (bs, hw) ∈ [0, 1]
            u_min = uncertainty.min(1, keepdim=True).values
            u_max = uncertainty.max(1, keepdim=True).values
            uncertainty_n = (uncertainty - u_min) / (u_max - u_min).clamp(min=eps)

            # Combined score: high cls, low uncertainty
            alpha = self.alpha.to(device=feats.device, dtype=cls_n.dtype)
            alpha = alpha.clamp(0.0, float(self.ALPHA_MAX))
            combined = (cls_n - alpha * uncertainty_n).clamp(-20.0, 20.0)
            combined = torch.nan_to_num(combined, nan=-1e4)

            # Mask invalid anchors
            valid = self.valid_mask.squeeze(-1).bool()
            if valid.shape[0] == 1 and bs > 1:
                valid = valid.expand(bs, -1)
            combined = combined.masked_fill(~valid, torch.finfo(combined.dtype).min)

            topk_ind = torch.topk(combined, self.num_queries, dim=1).indices.view(-1)
            topk_ind = topk_ind.clamp(0, features.shape[1] - 1).long()

        batch_ind = (
            torch.arange(bs, dtype=topk_ind.dtype, device=feats.device)
            .unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        ).clamp(0, bs - 1)

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors  = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = _safe_unit_interval(refer_bbox.sigmoid(), eps=self.REFINE_EPS)
        if dn_bbox is not None:
            dn_bbox = dn_bbox.clamp(-self.DN_LOGIT_LIMIT, self.DN_LOGIT_LIMIT)
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

    # ------------------------------------------------------------------ #
    # T1-A Look-Forward-Twice decoder
    # ------------------------------------------------------------------ #

    def _decoder_forward_lft(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decoder loop with Look-Forward-Twice (DINO arXiv:2203.03605, Sec. 3).

        Key difference from Legacy: `refer_bbox` is NOT detached between
        layers during training.  Gradient can therefore flow from layer i+1's
        loss backward through refined_bbox_i → delta_i → bbox_head_i, so the
        earlier bbox-heads receive richer signal about refinement quality.

        Box refinement chain (Eq. from DINO paper):
            δ_i  = bbox_head_i(norm(output_i))          # offset prediction
            b_i  = σ(δ_i + σ⁻¹(b_{i-1}))               # new reference box
                   (gradient flows through b_{i-1} — no .detach())

        !! clip_grad_norm ≤ 0.1 is required; without it gradients explode
           through the 6-layer chain of sigmoid/inverse-sigmoid operations. !!
        """
        output = embed
        dec_bboxes: list[torch.Tensor] = []
        dec_cls:    list[torch.Tensor] = []
        refer_bbox = _safe_unit_interval(refer_bbox.float().sigmoid(), eps=self.REFINE_EPS)

        for i, layer in enumerate(self.decoder.layers):
            ref_safe = _safe_unit_interval(refer_bbox, eps=self.REFINE_EPS)

            # Query positional embedding from current reference
            query_pos = self.query_pos_head(ref_safe.to(dtype=embed.dtype)).to(dtype=output.dtype)
            output = layer(
                output,
                ref_safe.to(dtype=output.dtype),
                feats,
                shapes,
                None,           # padding_mask
                attn_mask,
                query_pos,
            )

            # AMP-safe LayerNorm for head input: explicit fp32
            head_input = F.layer_norm(output.float(), (output.shape[-1],))

            # Box refinement — AMP-safe: cast head input to bbox_head dtype
            bbox_head = self.dec_bbox_head[i]
            bbox = bbox_head(_cast_to_param_dtype(head_input, bbox_head))
            bbox = self.BBOX_DELTA_LIMIT * torch.tanh(bbox.float() / self.BBOX_DELTA_LIMIT)
            refined = _safe_unit_interval(
                torch.sigmoid(bbox + _safe_inverse_sigmoid(ref_safe, eps=self.REFINE_EPS)),
                eps=self.REFINE_EPS,
            )

            # Score head
            score_head = self.dec_score_head[i]
            cls_out = score_head(_cast_to_param_dtype(head_input, score_head))
            dec_cls.append(cls_out)

            if self.training:
                dec_bboxes.append(refined)
                # T1-A LFT: no detach → gradient flows through refer_bbox chain
                refer_bbox = refined
            elif i == self.decoder.eval_idx:
                dec_bboxes.append(refined)
                break
            else:
                refer_bbox = refined

        return torch.stack(dec_bboxes), torch.stack(dec_cls)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: list[torch.Tensor],
        batch: dict | None = None,
    ) -> tuple | torch.Tensor:
        self._ensure_runtime_attrs()
        from ultralytics.models.utils.ops import get_cdn_group

        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )
        if dn_bbox is not None:
            dn_bbox = dn_bbox.clamp(-self.DN_LOGIT_LIMIT, self.DN_LOGIT_LIMIT)

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(
            feats, shapes, dn_embed, dn_bbox
        )
        dec_bboxes, dec_scores = self._decoder_forward_lft(
            embed, refer_bbox, feats, shapes, attn_mask=attn_mask,
        )

        # Cast to embed dtype for AMP compatibility
        out_dtype = embed.dtype
        dec_bboxes = dec_bboxes.to(dtype=out_dtype)
        dec_scores = dec_scores.to(dtype=out_dtype)
        enc_bboxes = enc_bboxes.to(dtype=out_dtype)
        enc_scores = enc_scores.to(dtype=out_dtype)

        out = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return out

        # Eval/export: convert the selected decoder layer to the format expected
        # by the RT-DETR validator: [bbox(4), max_score(1), label(1)].
        layer_idx = int(getattr(self, "eval_idx", -1))
        scores = dec_scores[layer_idx].sigmoid()
        max_scores, labels = scores.max(-1, keepdim=True)
        y = torch.cat((dec_bboxes[layer_idx], max_scores, labels.float()), -1)
        return y if self.export else (y, out)
