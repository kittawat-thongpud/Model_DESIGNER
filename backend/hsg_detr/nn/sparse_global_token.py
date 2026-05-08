"""
SparseGlobalTokenBlock — Token-Guided Spatial Recalibration (TGSR) for HSG-DETR.

TGSR Architecture:
  - Token selection via saliency (L2 + learned)
  - k×k self-attention on selected tokens (sharp softmax)
  - Channel recalibration: multiplicative (non-bypassable)
  - Spatial refinement: learned per-channel modulation

Key innovation: Multiplicative paradigm replaces additive+gamma
  OLD: output = x + gamma * delta          ← gamma→0 = bypass
  NEW: output = x * channel_weights        ← must learn meaningful weights

Zero-initialization ensures start-as-identity behavior.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

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


def _fp32_context(device: torch.device):
    """Disable autocast so numerically sensitive sparse blocks stay in FP32."""
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _safe_unit_interval(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Keep box coordinates away from exact 0/1 so inverse-sigmoid stays bounded."""
    return x.clamp(min=eps, max=1.0 - eps)


def _safe_inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Numerically safer inverse sigmoid used in the decoder refinement loop."""
    x = _safe_unit_interval(x, eps=eps)
    return torch.log(x / (1.0 - x))


class SGTokenBlock(nn.Module):
    """
    Token-Guided Spatial Recalibration (TGSR) block.

    Multiplicative paradigm replaces additive+gamma to prevent structural bypass:
      - Token selection via saliency (L2 + learned blend)
      - k×k self-attention on selected tokens (sharp softmax, better gradients)
      - Channel recalibration: multiplicative (optimizer cannot bypass)
      - Spatial refinement: learned per-channel modulation

    Zero-initialized output layers → starts as identity (chan_w ≈ 1.0)
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
    ) -> None:
        super().__init__()
        assert c1 == c2, f"SGTokenBlock is channel-preserving (c1={c1}, c2={c2})"
        mode = str(mode).lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Expected {sorted(self.VALID_MODES)}")
        ratio = float(ratio)
        if not torch.isfinite(torch.tensor(ratio)):
            raise ValueError(f"Invalid ratio: {ratio}")

        self.c = c2
        self.ratio = max(0.0, min(ratio, 1.0))
        self.mode = mode

        self.pre_norm = _make_gn(c2)

        # ── Saliency head (unchanged) ────────────────────────────────────
        self.saliency_head = nn.Sequential(
            nn.Conv2d(c2, c2 // 4, 1, bias=False),
            _make_gn(c2 // 4),
            nn.SiLU(),
            nn.Conv2d(c2 // 4, 1, 1, bias=False),
        )
        nn.init.zeros_(self.saliency_head[-1].weight)  # AdaptFormer: start as identity
        self.saliency_mix = nn.Parameter(torch.tensor(0.5))

        # ── k×k Self-attention (Linear projections, not Conv2d) ─────────────
        # Pre-LN for stability
        self.norm = nn.LayerNorm(c2)
        self.q_proj = nn.Linear(c2, c2, bias=False)
        self.k_proj = nn.Linear(c2, c2, bias=False)
        self.v_proj = nn.Linear(c2, c2, bias=False)
        # Xavier init for attention stability (lower variance than kaiming)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self._attn_scale = c2 ** -0.5

        # ── Channel Recalibration ──────────────────────────────────────────
        # Two-stage: compress → expand with non-linearity
        r = max(1, c2 // 16)
        self.chan_fc1 = nn.Linear(c2, r, bias=True)
        self.chan_fc2 = nn.Linear(r, c2, bias=True)
        # Small init for chan_fc1 to reduce warmup lag (chan_fc2 zero-init for identity)
        nn.init.normal_(self.chan_fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.chan_fc1.bias)
        # Zero-init: start as identity (chan_w ≈ 1.0 from 1 + tanh(0) = 1)
        nn.init.zeros_(self.chan_fc2.weight)
        nn.init.zeros_(self.chan_fc2.bias)

        # ── Spatial Refinement ───────────────────────────────────────────
        # 1×1 depthwise for per-channel spatial modulation
        self.spatial_conv = nn.Conv2d(c2, c2, 1, groups=c2, bias=True)
        # Near-identity with diversity (std=0.02 gives small variance for learning)
        nn.init.normal_(self.spatial_conv.weight, mean=1.0, std=0.02)
        nn.init.zeros_(self.spatial_conv.bias)

        # Alpha for blending channel vs spatial (set by trainer during warmup)
        self.register_buffer("alpha_scale", torch.tensor(0.0), persistent=True)

        # ── Hybrid mode: local path (multiplicative blend, non-bypassable) ─
        if mode == "hybrid":
            self.local_dw = nn.Sequential(
                nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
                _make_gn(c2),
                nn.SiLU(),
                nn.Conv2d(c2, c2, 1, bias=False),
                _make_gn(c2),
            )
            # Multiplicative blend: sigmoid(-3) ≈ 0.047 default (minimal local contribution at init)
            self.local_blend = nn.Parameter(torch.full((1, c2, 1, 1), -3.0))
        else:
            self.local_dw = None
            self.local_blend = None

        # Debug metadata
        self.debug_enabled = bool(debug_enabled)
        self.debug_to_cpu = False
        self.last_indices: torch.Tensor | None = None
        self.last_saliency: torch.Tensor | None = None
        self.last_gate: float | None = None  # tracks |chan_w - 1.0| (deviation from identity)
        self.last_mode: str | None = None
        self.last_k: int | None = None
        self.last_N: int | None = None

    def __setstate__(self, state: dict) -> None:
        """Restore runtime-only attrs for checkpoints saved before debug flags existed."""
        super().__setstate__(state)
        self._ensure_runtime_attrs()

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Backward compat: old checkpoints have gamma/out_proj, TGSR has chan_fc.
        
        Strategy: Drop old gamma/out_proj keys, initialize chan_fc fresh.
        Architecture change is too large for weight remapping.
        """
        # Remove legacy keys that don't exist in TGSR (including q/k/v_proj Conv2d→Linear)
        legacy_keys_to_drop = [
            'gamma', 'out_proj', 'local_gamma', 'local_dw',
            'q_proj', 'k_proj', 'v_proj',  # Conv2d weight shape [C,C,1,1] ≠ Linear [C,C]
        ]
        keys_to_remove = [
            k for k in list(state_dict.keys())
            if k.startswith(prefix) and any(
                k[len(prefix):].split('.')[0] == lk for lk in legacy_keys_to_drop
            )
        ]
        for k in keys_to_remove:
            state_dict.pop(k, None)
        
        # Warn about dropped keys for operational visibility
        if keys_to_remove:
            import warnings
            short_names = list(set(k.split('.')[-2] for k in keys_to_remove))[:5]
            warnings.warn(
                f"TGSR checkpoint load: dropped {len(keys_to_remove)} legacy keys "
                f"(architecture changed). Re-initializing: {short_names}...",
                UserWarning,
            )
        
        # Call parent to load remaining keys (saliency_head, saliency_mix, etc.)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )
        
        # Re-initialize TGSR-specific layers that weren't in old checkpoints
        # (chan_fc1, chan_fc2, spatial_conv, q/k/v_proj as Linear)
        # They'll start with default init (zero-init for recal layers)

    # ------------------------------------------------------------------ #
    # Debug / introspection
    # ------------------------------------------------------------------ #

    def _ensure_runtime_attrs(self) -> None:
        """Backfill non-parameter attrs that old pickled checkpoints may lack."""
        defaults = {
            "debug_enabled": False,
            "debug_to_cpu": False,
            "last_indices": None,
            "last_saliency": None,
            "last_gate": None,
            "last_mode": None,
            "last_k": None,
            "last_N": None,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def get_debug_state(self) -> dict:
        """Return non-gradient metadata from the last forward pass."""
        self._ensure_runtime_attrs()
        return {
            "indices": self.last_indices,
            "saliency": self.last_saliency,
            "gate": self.last_gate,
            "mode": self.last_mode,
            "k": self.last_k,
            "N": self.last_N,
        }

    def set_debug(self, enabled: bool = True, cpu: bool = False) -> None:
        """Enable or disable detached selector metadata capture."""
        self._ensure_runtime_attrs()
        self.debug_enabled = bool(enabled)
        self.debug_to_cpu = bool(cpu)
        if not self.debug_enabled:
            self.last_indices = None
            self.last_saliency = None

    def _store_debug(
        self,
        indices: torch.Tensor | None = None,
        saliency: torch.Tensor | None = None,
    ) -> None:
        """Store detached metadata only when debug capture is explicitly enabled."""
        self._ensure_runtime_attrs()
        if not self.debug_enabled:
            self.last_indices = None
            self.last_saliency = None
            return

        if indices is not None:
            indices = indices.detach()
            self.last_indices = indices.cpu() if self.debug_to_cpu else indices
        if saliency is not None:
            saliency = saliency.detach()
            self.last_saliency = saliency.cpu() if self.debug_to_cpu else saliency

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_saliency(self, x: torch.Tensor) -> torch.Tensor:
        """L2 activation energy + learned saliency per spatial token, in FP32."""
        B, C, H, W = x.shape
        N = H * W
        # L2 energy (heuristic)
        l2_energy = x.view(B, C, N).float().pow(2).sum(dim=1)  # [B, N]
        l2_energy = torch.nan_to_num(l2_energy, nan=0.0, posinf=0.0, neginf=0.0)
        # Per-sample min-max normalisation
        eps = 1e-6
        l2_min = l2_energy.min(1, keepdim=True).values
        l2_max = l2_energy.max(1, keepdim=True).values
        l2_range = (l2_max - l2_min).clamp(min=eps)
        l2_norm = (l2_energy - l2_min) / l2_range
        # Learned saliency
        learned = self.saliency_head(x.float()).view(B, N)  # [B, N]
        learned = torch.sigmoid(learned)
        learned = torch.nan_to_num(learned, nan=0.0, posinf=0.0, neginf=0.0)
        # Blend: mix parameter clamped to [0,1]
        mix = self.saliency_mix.sigmoid()
        importance = mix * l2_norm + (1.0 - mix) * learned
        return torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)

    def _select_k(self, N: int) -> int:
        """Compute effective k based on mode and constraints."""
        if self.mode == "dense":
            if int(N) > self.DENSE_TOKEN_LIMIT:
                raise RuntimeError(
                    f"Dense SGB too large: N={N}. Use topk/hybrid instead."
                )
            return int(N)
        return max(1, min(int(round(self.ratio * int(N))), int(N)))

    def _k_times_k_attention(
        self,
        selected: torch.Tensor,  # [B, k, C]
    ) -> torch.Tensor:
        """
        k×k self-attention on selected tokens (NOT k×N).
        
        Sharp softmax due to smaller range = better gradients.
        Pre-LN for stability.
        """
        B, k, C = selected.shape
        
        # Pre-LN (match norm's dtype to avoid mixed dtype error)
        orig_dtype = selected.dtype
        selected_n = self.norm(selected.to(self.norm.weight.dtype))
        
        # Linear projections
        q = self.q_proj(selected_n)  # [B, k, C]
        k_ = self.k_proj(selected_n)  # [B, k, C]
        v = self.v_proj(selected_n)   # [B, k, C]
        
        # k×k attention (sharp softmax, better than k×N flat softmax)
        # Use more conservative scale to prevent gradient explosion
        attn = torch.bmm(q, k_.transpose(1, 2)) * (float(self._attn_scale) * 0.5)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        # Tighter clamp for numerical stability in FP16
        attn = attn.clamp(min=-50.0, max=50.0)
        # Softmax stabilization: subtract max (safer than manual)
        attn = attn - attn.max(dim=-1, keepdim=True).values.detach()
        attn = torch.softmax(attn, dim=-1)
        # Handle degenerate cases where all values were -inf
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        # Renormalize if NaN handling changed distribution
        attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn = attn / attn_sum
        
        attended = torch.bmm(attn, v)  # [B, k, C]
        # Immediate clamp to prevent gradient explosion
        attended = torch.clamp(attended, min=-50.0, max=50.0)
        attended = torch.nan_to_num(attended, nan=0.0, posinf=10.0, neginf=-10.0)
        return attended.to(orig_dtype)

    def _channel_recalibration(
        self,
        attended: torch.Tensor,  # [B, k, C]
        x: torch.Tensor,         # [B, C, H, W]
    ) -> torch.Tensor:
        """
        Multiplicative channel recalibration from attended tokens.
        
        Global descriptor (mean+max pool) → FC bottleneck → channel weights.
        Zero-init FC2 → starts as identity (chan_w ≈ 1.0)
        All math in FP32 for AMP stability.
        """
        B, C, H, W = x.shape
        
        # Global descriptor from attended tokens (FP32 for stability)
        attended_f = attended.float()
        ctx_mean = attended_f.mean(dim=1)   # [B, C]
        ctx_max = attended_f.max(dim=1).values  # [B, C]
        ctx = ctx_mean + ctx_max  # [B, C]
        
        # FC bottleneck: C → r → C (FP32)
        chan_w = self.chan_fc1(ctx)   # [B, r]
        chan_w = F.silu(chan_w)
        chan_w = self.chan_fc2(chan_w)  # [B, C]
        
        # 1 + tanh: range (0, 2), zero-init → starts at 1.0 (identity)
        chan_w = 1.0 + torch.tanh(chan_w)
        # Clamp to safe range [0.1, 1.9] to prevent gradient explosion
        chan_w = torch.clamp(chan_w, min=0.1, max=1.9)
        
        # Store deviation from identity for metrics (0.0 = identity, >0 = learning)
        self.last_gate = float((chan_w.detach() - 1.0).abs().mean().item())
        
        # Multiplicative recalibration (broadcast to spatial)
        chan_w = chan_w.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        out = x * chan_w.to(dtype=x.dtype)
        # Final safety clamp for output
        return torch.clamp(out, min=-1000.0, max=1000.0)

    def _spatial_refinement(
        self,
        x: torch.Tensor,  # [B, C, H, W]
    ) -> torch.Tensor:
        """
        Learned spatial modulation via 1×1 depthwise conv.
        Near-identity init (mean=1.0, std=0.02) → starts ~pass-through with diversity.
        """
        # Depthwise 1×1: per-channel spatial modulation (FP32 for AMP compatibility)
        with _fp32_context(x.device):
            spatial_w = self.spatial_conv(x.float())  # [B, C, H, W]
        return spatial_w

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TGSR forward: multiplicative paradigm (non-bypassable).
        
        Flow:
          1. Token selection (saliency)
          2. k×k self-attention on selected
          3. Channel recalibration (multiplicative, dense output)
          4. Spatial refinement (blended by alpha_scale)
        """
        self._ensure_runtime_attrs()
        B, C, H, W = x.shape
        N = int(H) * int(W)
        
        k_actual = self._select_k(N)
        self.last_N = N
        self.last_k = k_actual
        self.last_mode = self.mode
        
        with _fp32_context(x.device):
            # ── 1. Normalize and compute saliency ─────────────────────────
            # Use pre_norm's dtype to avoid mixed dtype error during AMP
            x_norm = self.pre_norm(x.to(self.pre_norm.weight.dtype))
            
            if self.mode == "dense":
                # Use all tokens
                importance = None
                topk_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
                k_actual = N
            else:
                # Saliency from RAW features (preserve L2 magnitude info, pre_norm destroys it)
                importance = self._compute_saliency(x.float())
                if not torch.isfinite(importance).any():
                    importance = torch.ones_like(importance)
                topk_idx = torch.topk(importance, k_actual, dim=1).indices
                topk_idx = torch.clamp(topk_idx, 0, N - 1)
            
            topk_idx = torch.nan_to_num(topk_idx, nan=0).long()
            self._store_debug(topk_idx, importance)
            
            # ── 2. Gather selected tokens ────────────────────────────────
            x_flat = x_norm.view(B, C, N)  # [B, C, N]
            idx_exp = topk_idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, k]
            selected = torch.gather(x_flat, 2, idx_exp)  # [B, C, k]
            selected = selected.transpose(1, 2)  # [B, k, C]
            
            # ── 3. k×k self-attention ───────────────────────────────────
            attended = self._k_times_k_attention(selected)  # [B, k, C]
            # Stabilize: clamp extreme values that can cause NaN in backward
            attended = torch.clamp(attended, min=-100.0, max=100.0)
            if not torch.isfinite(attended).all():
                attended = torch.nan_to_num(attended, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # ── 4. Channel recalibration (multiplicative) ───────────────
            x_recal = self._channel_recalibration(attended, x)  # [B, C, H, W]
            # Stabilize output
            x_recal = torch.clamp(x_recal, min=-1000.0, max=1000.0)
            if not torch.isfinite(x_recal).all():
                x_recal = torch.nan_to_num(x_recal, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # ── 5. Spatial refinement (blended by alpha) ───────────────
            # Always compute spatial path (no if branch) to ensure gradient flow
            # alpha=0 → output = x_recal (spatial contribution zeroed)
            alpha = self.alpha_scale.to(dtype=x.dtype, device=x.device)
            spatial_w = self._spatial_refinement(x_recal)
            # Stabilize spatial weights
            spatial_w = torch.clamp(spatial_w, min=-1000.0, max=1000.0)
            if not torch.isfinite(spatial_w).all():
                spatial_w = torch.nan_to_num(spatial_w, nan=0.0, posinf=100.0, neginf=-100.0)
            out = x_recal + alpha * (spatial_w - x_recal)
            
            # ── 6. Hybrid local path (multiplicative blend, non-bypassable) ─
            if self.mode == "hybrid" and self.local_dw is not None:
                local_feat = self.local_dw(x_norm)
                # Multiplicative blend: sigmoid(0) = 0.5 default
                blend = torch.sigmoid(self.local_blend).to(dtype=x.dtype)
                out = out * (1.0 - blend) + local_feat.to(dtype=x.dtype) * blend
        
        return out.to(dtype=x.dtype)


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
    As the scheduled ``alpha`` buffer increases, spatially salient tokens
    receive higher priority during hard top-k query selection, aligning the
    decoder's initial queries with the regions the SGB encoder has already
    identified as important.

    Args (same as RTDETRDecoder):
        nc, ch, hd, nq, ndp, nh, ndl, d_ffn, dropout, act, eval_idx,
        nd, label_noise_ratio, box_noise_scale, learnt_init_query

    Saliency weight is a scheduled buffer controlled by ``set_alpha()`` and
    bounded by ``ALPHA_MAX``.
    """

    ALPHA_MAX: float = 0.5  # maximum saliency weighting
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
        """Restore scheduled-alpha buffer for checkpoints saved before it existed."""
        super().__setstate__(state)
        self._ensure_runtime_attrs()

    def _ensure_runtime_attrs(self) -> None:
        """Backfill non-parameter buffers that old pickled checkpoints may lack."""
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
        """Set saliency query-selection strength for external warmup schedules."""
        self._ensure_runtime_attrs()
        value = max(0.0, min(float(value), float(self.ALPHA_MAX)))
        self.alpha.fill_(value)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Map legacy alpha_logit checkpoints onto the scheduled alpha buffer."""
        legacy_key = prefix + "alpha_logit"
        alpha_key = prefix + "alpha"
        if legacy_key in state_dict:
            if alpha_key not in state_dict:
                legacy_alpha = float(self.ALPHA_MAX) * torch.sigmoid(
                    state_dict[legacy_key].detach().float()
                )
                state_dict[alpha_key] = legacy_alpha.reshape_as(self.alpha)
            del state_dict[legacy_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        self._ensure_runtime_attrs()
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

        alpha = self.alpha.to(device=features.device, dtype=energy_norm.dtype)
        alpha = alpha.clamp(0.0, float(self.ALPHA_MAX))
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
            torch.arange(end=bs, dtype=topk_ind.dtype, device=features.device)
            .unsqueeze(-1)
            .repeat(1, self.num_queries)
            .view(-1)
        )

        # Guard: ensure batch_ind is also valid
        batch_ind = torch.clamp(batch_ind, 0, bs - 1)

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

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

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """RT-DETR forward with a safer decoder refinement loop for early training."""
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

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        dec_bboxes, dec_scores = self._safe_decoder_forward(
            embed,
            refer_bbox,
            feats,
            shapes,
            attn_mask=attn_mask,
        )
        out = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return out
        # Eval/export: convert raw class scores to (max_score, label) format expected by RT-DETR validator
        # Validator expects [bboxes(4), score(1), label(1)] = 6, not [bboxes(4), scores(nc)] = 4+nc
        scores = dec_scores.squeeze(0).sigmoid()
        max_scores, labels = scores.max(-1, keepdim=True)
        y = torch.cat((dec_bboxes.squeeze(0), max_scores, labels.float()), -1)
        return y if self.export else (y, out)

    def _safe_decoder_forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror the base decoder but keep reference boxes in a safer numeric range."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = _safe_unit_interval(refer_bbox.float().sigmoid(), eps=self.REFINE_EPS)

        for i, layer in enumerate(self.decoder.layers):
            ref_for_layer = _safe_unit_interval(refer_bbox, eps=self.REFINE_EPS)
            output = layer(
                output,
                ref_for_layer.to(dtype=output.dtype),
                feats,
                shapes,
                padding_mask,
                attn_mask,
                self.query_pos_head(ref_for_layer).to(dtype=output.dtype),
            )

            head_input = F.layer_norm(output.float(), (output.shape[-1],))
            bbox = self.dec_bbox_head[i](head_input)
            bbox = self.BBOX_DELTA_LIMIT * torch.tanh(bbox / self.BBOX_DELTA_LIMIT)
            refined_bbox = torch.sigmoid(bbox + _safe_inverse_sigmoid(ref_for_layer, eps=self.REFINE_EPS))
            refined_bbox = _safe_unit_interval(refined_bbox, eps=self.REFINE_EPS)

            if self.training:
                dec_cls.append(self.dec_score_head[i](head_input))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    prev_ref = _safe_unit_interval(last_refined_bbox, eps=self.REFINE_EPS)
                    prev_bbox = torch.sigmoid(bbox + _safe_inverse_sigmoid(prev_ref, eps=self.REFINE_EPS))
                    dec_bboxes.append(_safe_unit_interval(prev_bbox, eps=self.REFINE_EPS))
            elif i == self.decoder.eval_idx:
                dec_cls.append(self.dec_score_head[i](head_input))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
