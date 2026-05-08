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
    """No-op: SGB blocks now run under AMP (bf16/fp16) for training speed.

    Forcing FP32 for the whole block was costing ~30–50 % per-epoch time on
    GPUs with bf16 tensor cores, with no measurable stability benefit once
    the per-op safeguards landed:

      * attention logits clamped to ±30 (fp16-safe)
      * `nan_to_num` after softmax / matmul
      * GroupNorm + pre-LN in FP32 internally
      * bounded gates: chan_w ∈ (0.5, 1.5), spatial delta via tanh
      * dense ctx mean keeps the FC bottleneck input in a finite range

    Kept as a function (not deleted) so the call sites (`with
    _fp32_context(x.device):`) stay structurally unchanged — flip back to
    forced FP32 by restoring the previous body if numerical issues return.
    """
    del device  # unused
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
    Sparse-Spatial Residual (SSR) block — evolution of TGSR.

    True hybrid-sparse design: sparse selector AND sparse output.

      1. Saliency selects top-k spatial positions (L2 + learned blend).
      2. k×k self-attention computes refined embeddings for those k tokens.
      3. **Scatter-back**: the (attended − selected) residual is written
         back to the original k positions only — never broadcast densely.
      4. A 3×3 depthwise mix (`local_mix`) spreads each sparse delta to
         its immediate neighborhood. Zero-init → no spread at start.
      5. Auxiliary per-channel layer-scale (range [0.9, 1.1]) — at most
         ±10 % magnitude tweak; cannot zero or amplify, cannot bypass.

    Equation:
        out = (x · chan_w_narrow) + alpha · spread( scatter( delta_topk ) )

    Sparse property: at any block, only ~9·k positions (k attended + 3×3
    spread) receive a non-trivial spatial update; remaining positions
    pass through with at most ±10 % per-channel scaling.

    Identity at init: alpha_scale starts at 0 and chan_w starts at 1.0,
    so out == x exactly until warmup begins.
    """

    VALID_MODES: set[str] = {"dense", "topk", "hybrid"}
    DENSE_TOKEN_LIMIT: int = 4096
    # Bound the scatter-back residual (attended − selected) via tanh so
    # |delta| ≤ SPATIAL_DELTA_BOUND. Smooth near zero (identity at init),
    # finite under fp16/bf16 AMP, and contains early-training overshoot
    # before the optimizer has shaped q/k/v projections.
    SPATIAL_DELTA_BOUND: float = 1.0

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

        # ── Auxiliary per-channel layer scale (range [0.9, 1.1]) ──────────
        # Demoted from "channel gate" to a tight fine-tune in the SSR
        # design — the dominant signal is the sparse scatter residual.
        r = max(1, c2 // 16)
        self.chan_fc1 = nn.Linear(c2, r, bias=True)
        self.chan_fc2 = nn.Linear(r, c2, bias=True)
        nn.init.normal_(self.chan_fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.chan_fc1.bias)
        # Zero-init fc2 → starts as identity (0.9 + 0.2·sigmoid(0) = 1.0)
        nn.init.zeros_(self.chan_fc2.weight)
        nn.init.zeros_(self.chan_fc2.bias)

        # ── Local Mix (3×3 depthwise) — spreads sparse scatter to neighbors ─
        # Zero-init → at start no spread (sparse update stays at k positions
        # exactly). Learns to expand each k-position spike into a 3×3 patch
        # as needed. Costs ~9 ops/channel/position, depthwise so cheap.
        self.local_mix = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)
        nn.init.zeros_(self.local_mix.weight)

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
            'spatial_conv',  # removed in SSR rewrite (replaced by local_mix 3×3)
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
        
        # Pre-LN in FP32
        orig_dtype = selected.dtype
        selected_n = self.norm(selected.float())
        
        # Linear projections
        q = self.q_proj(selected_n)  # [B, k, C]
        k_ = self.k_proj(selected_n)  # [B, k, C]
        v = self.v_proj(selected_n)   # [B, k, C]
        
        # k×k attention (sharp softmax, better than k×N flat softmax).
        # Tighter logit clamp (±30) keeps softmax in a regime where it stays
        # finite under fp16 / bf16 too — the prior ±80 only made sense in
        # FP32. After the subtract-max trick the typical logit range is
        # already <= 0, so −30 clipping costs no expressive power.
        attn = torch.bmm(q, k_.transpose(1, 2)) * float(self._attn_scale)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = attn.clamp(min=-30.0, max=30.0)
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        
        attended = torch.bmm(attn, v)  # [B, k, C]
        attended = torch.nan_to_num(attended, nan=0.0, posinf=0.0, neginf=0.0)
        return attended.to(orig_dtype)

    def _channel_recalibration(
        self,
        attended: torch.Tensor,  # [B, k, C]
        x: torch.Tensor,         # [B, C, H, W]
    ) -> torch.Tensor:
        """
        Auxiliary per-channel layer scale (range [0.9, 1.1]).

        In the SSR design the dominant signal is the sparse scatter
        residual; this layer is a small fine-tune, NOT a gate. It can
        adjust per-channel magnitude by at most ±10 % — it cannot zero
        features, cannot 2× amplify, and cannot act as a learnable
        killswitch. Identity at init via zero-init `chan_fc2`.
        """
        B, C, H, W = x.shape

        # Sparse + dense ctx: top-k attended tokens contribute fine detail,
        # full-map mean stabilises gradient flow through every position.
        attended_f = attended.float()
        ctx_sparse = attended_f.mean(dim=1) + attended_f.max(dim=1).values  # [B, C]
        ctx_dense = x.float().mean(dim=(2, 3))                              # [B, C]
        ctx = ctx_sparse + ctx_dense

        chan_w = self.chan_fc1(ctx)
        chan_w = F.silu(chan_w)
        chan_w = self.chan_fc2(chan_w)

        # Range [0.9, 1.1]: 0.9 + 0.2·sigmoid(0) = 1.0 at init.
        # Tight bound enforces "fine-tune, not gate" semantics.
        chan_w = 0.9 + 0.2 * torch.sigmoid(chan_w)

        # Deviation from identity (0.0 = identity, max 0.1 by construction)
        self.last_gate = float((chan_w.detach() - 1.0).abs().mean().item())

        chan_w = chan_w.unsqueeze(-1).unsqueeze(-1)
        return x * chan_w.to(dtype=x.dtype)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse-Spatial Residual (SSR) forward.

        Flow:
          1. Saliency → top-k token indices.
          2. Gather, k×k self-attention on selected tokens.
          3. SCATTER-BACK: tanh-bounded residual (attended − selected) is
             written at the k original positions only — rest of the map
             stays zero in delta.
          4. 3×3 depthwise `local_mix` (zero-init) spreads each sparse
             delta into its 3×3 neighborhood.
          5. Apply alpha-scheduled sparse residual:
                out = (x · chan_w_narrow) + alpha · spread(scatter(delta))
          6. (Hybrid mode) blend with optional dense local_dw branch.

        Identity at init: alpha_scale=0, chan_w=1.0, local_mix=0 →
        out == x exactly until warmup begins. No dense gate output —
        non-attended positions only see the ±10 % chan_w fine-tune.
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
            x_norm = self.pre_norm(x.float())

            if self.mode == "dense":
                importance = None
                topk_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
                k_actual = N
            else:
                # Saliency from RAW features (preserve L2 magnitude info,
                # pre_norm destroys it).
                importance = self._compute_saliency(x.float())
                if not torch.isfinite(importance).any():
                    importance = torch.ones_like(importance)
                topk_idx = torch.topk(importance, k_actual, dim=1).indices
                topk_idx = torch.clamp(topk_idx, 0, N - 1)

            topk_idx = torch.nan_to_num(topk_idx, nan=0).long()
            self._store_debug(topk_idx, importance)

            # ── 2. Gather selected tokens (in normalized space) ──────────
            x_flat_norm = x_norm.view(B, C, N)
            idx_exp = topk_idx.unsqueeze(1).expand(-1, C, -1)        # [B, C, k]
            selected = torch.gather(x_flat_norm, 2, idx_exp)         # [B, C, k]
            selected = selected.transpose(1, 2).contiguous()         # [B, k, C]

            # ── 3. k×k self-attention ─────────────────────────────────
            attended = self._k_times_k_attention(selected)           # [B, k, C]

            # ── 4. SCATTER-BACK as bounded sparse residual ──────────────
            # attended and selected are both in pre_norm/LN space; their
            # difference is the per-token correction the attention block
            # learned. Tanh-bound for AMP/early-train safety.
            bound = float(self.SPATIAL_DELTA_BOUND)
            delta_kc = bound * torch.tanh(
                (attended - selected) / bound
            )                                                        # [B, k, C]
            delta_ck = delta_kc.transpose(1, 2).contiguous()         # [B, C, k]

            # Build dense delta tensor: zeros everywhere except top-k positions
            delta_flat = torch.zeros(
                B, C, N, dtype=delta_ck.dtype, device=delta_ck.device,
            )
            delta_flat.scatter_(2, idx_exp, delta_ck)
            delta_dense = delta_flat.view(B, C, H, W)

            # ── 5. Local 3×3 spread (zero-init: identity at start) ──────
            # local_mix turns each k-position spike into a 3×3 patch as it
            # learns. Output remains effectively sparse: ≤ 9·k non-zero
            # positions per block, vs. all H·W in a dense gate.
            delta_dense = delta_dense + self.local_mix(delta_dense)

            # ── 6. Auxiliary per-channel layer scale (range [0.9, 1.1]) ─
            x_recal = self._channel_recalibration(attended, x)        # [B,C,H,W]

            # ── 7. Apply alpha-scheduled sparse residual ────────────────
            alpha = self.alpha_scale.to(dtype=x.dtype, device=x.device)
            out = x_recal + alpha * delta_dense.to(dtype=x.dtype)

            # ── 8. Hybrid local path (optional, unchanged) ──────────────
            if self.mode == "hybrid" and self.local_dw is not None:
                local_feat = self.local_dw(x_norm)
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
