"""
SparseGlobalTokenBlock for HSG-DETR.

The HSG-DETR sparse branch is intentionally selected-token-only:
  - score spatial tokens with parameter-free L2 energy
  - project Q/K/V only after top-k gather
  - run k x k sparse attention in an FP32 island
  - scatter the selected delta onto a zero canvas
  - fuse with a small non-zero per-channel LayerScale gamma

Non-selected spatial positions keep the CNN residual path, but they do not
receive sparse-attention updates. This keeps the block honest to the "sparse
attention with CNN hybrid" design in the paper draft.
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


def _bounded_tanh(x: torch.Tensor, limit: float = 6.0) -> torch.Tensor:
    """Soft-bound a tensor without hard clipping gradients in the central range."""
    limit = float(limit)
    if limit <= 0:
        return x
    return limit * torch.tanh(x / limit)


def _module_param_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Return the dtype of a module parameter without assuming a specific layout."""
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return fallback


def _make_gn(channels: int) -> nn.GroupNorm:
    """Return a GroupNorm layer without BatchNorm running buffers."""
    groups = min(32, int(channels))
    while int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels), eps=1e-5)


def _safe_unit_interval(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Keep box coordinates away from exact 0/1 so inverse-sigmoid stays bounded."""
    return x.clamp(min=eps, max=1.0 - eps)


def _safe_inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Numerically safer inverse sigmoid used in the decoder refinement loop."""
    x = _safe_unit_interval(x, eps=eps)
    return torch.log(x / (1.0 - x))


class SGTokenBlock(nn.Module):
    """
    Selected-token-only sparse attention block for HSG-DETR.

    The sparse branch projects and updates only the top-k selected spatial
    tokens. Non-selected locations keep the CNN residual path and receive no
    sparse-branch delta.
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
        gamma_init: float = 0.10,
        finite_limit: float = 20.0,
        attn_logit_limit: float = 50.0,
        delta_limit: float = 6.0,
        gamma_floor: float | None = None,
    ) -> None:
        super().__init__()
        assert c1 == c2, f"SGTokenBlock is channel-preserving (c1={c1}, c2={c2})"
        mode = str(mode).lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Expected {sorted(self.VALID_MODES)}")
        ratio = float(ratio)
        if not torch.isfinite(torch.tensor(ratio)):
            raise ValueError(f"Invalid ratio: {ratio}")

        self.c = int(c2)
        self.ratio = max(0.0, min(ratio, 1.0))
        # Accept legacy "hybrid" configs, but the new baseline has no local path.
        self.mode = "topk" if mode == "hybrid" else mode
        self.requested_mode = mode
        self.finite_limit = float(finite_limit)
        self.attn_logit_limit = float(attn_logit_limit)
        self.delta_limit = float(delta_limit)
        if gamma_floor is None:
            if self.ratio >= 0.20:
                gamma_floor = 0.010
            elif self.ratio >= 0.10:
                gamma_floor = 0.0075
            else:
                gamma_floor = 0.005
        self.gamma_floor = max(0.0, float(gamma_floor))

        self.pre_norm = _make_gn(c2)
        self.norm = nn.LayerNorm(c2)
        self.q_proj = nn.Linear(c2, c2, bias=False)
        self.k_proj = nn.Linear(c2, c2, bias=False)
        self.v_proj = nn.Linear(c2, c2, bias=False)
        self.out_proj = nn.Linear(c2, c2, bias=False)
        self.gamma = nn.Parameter(torch.full((1, c2, 1, 1), float(gamma_init)))
        self._attn_scale = c2 ** -0.5

        # Learnable saliency head: 2-layer pointwise CNN that scores each spatial token.
        # Gradient flows back via the soft gate in forward(), teaching the backbone to
        # produce high-score features at object locations rather than relying on raw L2.
        _sal_mid = max(int(c2) // 8, 16)
        self.saliency_head = nn.Sequential(
            nn.Conv2d(c2, _sal_mid, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(_sal_mid, 1, 1, bias=True),
        )
        nn.init.xavier_uniform_(self.saliency_head[0].weight, gain=0.1)
        nn.init.zeros_(self.saliency_head[-1].weight)
        nn.init.constant_(self.saliency_head[-1].bias, 0.0)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)

        self.debug_enabled = bool(debug_enabled)
        self.debug_to_cpu = False
        self.last_indices: torch.Tensor | None = None
        self.last_saliency: torch.Tensor | None = None
        self.last_mode: str | None = None
        self.last_k: int | None = None
        self.last_N: int | None = None
        self.last_selected_ratio: float | None = None
        self.last_gamma_raw_abs_mean: float | None = None
        self.last_gamma_abs_mean: float | None = None
        self.last_gamma_floor: float | None = None
        self.last_delta_norm_selected: float | None = None
        self.last_delta_norm_nonselected: float | None = None
        self.last_delta_scaled_norm_selected: float | None = None
        self.last_selected_grad_norm: float | None = None
        self.last_nonselected_sparse_grad: float | None = None
        self.last_finite_guard_count: int | None = None

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
        """Drop obsolete sparse-block keys and incompatible legacy Conv2d projections."""
        legacy_roots = {
            "saliency_mix",
            "chan_fc1",
            "chan_fc2",
            "spatial_conv",
            "alpha_scale",
            "local_dw",
            "local_blend",
            "local_gamma",
            "gate",
        }
        own_state = self.state_dict()
        keys_to_remove: list[str] = []
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            local_name = key[len(prefix):]
            root = local_name.split(".")[0]
            if root in legacy_roots:
                keys_to_remove.append(key)
                continue
            expected = own_state.get(local_name)
            value = state_dict.get(key)
            if (
                expected is not None
                and torch.is_tensor(value)
                and tuple(value.shape) != tuple(expected.shape)
            ):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            state_dict.pop(key, None)

        if keys_to_remove:
            import warnings

            short_names = sorted({k[len(prefix):].split(".")[0] for k in keys_to_remove})[:8]
            warnings.warn(
                "SGTokenBlock checkpoint load: dropped "
                f"{len(keys_to_remove)} obsolete/incompatible keys: {short_names}",
                UserWarning,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
            "last_mode": None,
            "last_k": None,
            "last_N": None,
            "last_selected_ratio": None,
            "last_gamma_raw_abs_mean": None,
            "last_gamma_abs_mean": None,
            "last_gamma_floor": getattr(self, "gamma_floor", 0.0),
            "last_delta_norm_selected": None,
            "last_delta_norm_nonselected": None,
            "last_delta_scaled_norm_selected": None,
            "last_selected_grad_norm": None,
            "last_nonselected_sparse_grad": None,
            "last_finite_guard_count": None,
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
            "mode": self.last_mode,
            "k": self.last_k,
            "N": self.last_N,
            "selected_ratio": self.last_selected_ratio,
            "gamma_raw_abs_mean": self.last_gamma_raw_abs_mean,
            "gamma_abs_mean": self.last_gamma_abs_mean,
            "gamma_floor": self.last_gamma_floor,
            "delta_norm_selected": self.last_delta_norm_selected,
            "delta_norm_nonselected": self.last_delta_norm_nonselected,
            "delta_scaled_norm_selected": self.last_delta_scaled_norm_selected,
            "selected_grad_norm": self.last_selected_grad_norm,
            "nonselected_sparse_grad": self.last_nonselected_sparse_grad,
            "finite_guard_count": self.last_finite_guard_count,
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
        """Store detached selector metadata only when debug capture is enabled."""
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

    def _compute_saliency(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Learnable saliency head scoring each spatial token.

        Returns un-detached scores (B, N) so the soft gate in forward() carries
        gradient back through saliency_head into the backbone.  The caller must
        .detach() before passing to top-k to avoid differentiating through argmax.
        """
        sal_dtype = self.saliency_head[0].weight.dtype
        B, C, H, W = x_norm.shape
        scores = self.saliency_head(x_norm.to(dtype=sal_dtype))   # (B, 1, H, W)
        scores_flat = scores.view(B, H * W)                        # (B, N)
        # Keep the old activation-energy selector as a non-learnable prior so
        # a zero-initialized saliency head does not make top-k pick fixed
        # arbitrary positions at the start of training.
        energy_prior = x_norm.detach().float().view(B, C, H * W).pow(2).mean(dim=1)
        energy_prior = energy_prior / energy_prior.mean(dim=1, keepdim=True).clamp(min=1e-6)
        scores_flat = scores_flat + energy_prior.to(dtype=scores_flat.dtype)
        return torch.nan_to_num(scores_flat, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _sinusoidal_pos_enc_2d(
        idx: torch.Tensor,
        H: int,
        W: int,
        dim: int,
    ) -> torch.Tensor:
        """2D sinusoidal positional encoding for selected flat spatial indices.

        idx     : (B, k) integer flat indices in [0, H*W)
        Returns : (B, k, dim) float32 encodings, scaled to ~unit norm per token.

        Encoding layout (dim must be divisible by 4):
          [ sin(x·freq), cos(x·freq), sin(y·freq), cos(y·freq) ]
        so Q and K each receive distinct spatial coordinates before dot-product.
        """
        B, k = idx.shape
        device = idx.device
        y_f = (idx // W).float() / max(H - 1, 1)    # (B, k) normalised to [0, 1]
        x_f = (idx %  W).float() / max(W - 1, 1)    # (B, k) normalised to [0, 1]
        d = max(dim // 4, 1)
        freq = 10000.0 ** (
            -torch.arange(d, device=device, dtype=torch.float32) * 4.0 / max(dim, 1)
        )                                             # (d,)
        enc = torch.cat([
            torch.sin(x_f.unsqueeze(-1) * freq),     # (B, k, d)
            torch.cos(x_f.unsqueeze(-1) * freq),
            torch.sin(y_f.unsqueeze(-1) * freq),
            torch.cos(y_f.unsqueeze(-1) * freq),
        ], dim=-1)                                    # (B, k, 4*d)
        if enc.shape[-1] < dim:
            enc = F.pad(enc, (0, dim - enc.shape[-1]))
        elif enc.shape[-1] > dim:
            enc = enc[..., :dim]
        return enc * (dim ** -0.5)                    # normalise magnitude

    def _select_k(self, N: int) -> int:
        """Compute effective k based on mode and constraints."""
        if self.mode == "dense":
            if int(N) > self.DENSE_TOKEN_LIMIT:
                raise RuntimeError(
                    f"Dense SGB too large: N={N}. Use topk instead."
                )
            return int(N)
        return max(1, min(int(round(self.ratio * int(N))), int(N)))

    def _selected_attention_delta(
        self,
        selected: torch.Tensor,
        pos_enc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Q/K/V and sparse self-attention only on selected tokens.

        pos_enc : (B, k, C) float32 2D sinusoidal encoding for selected positions.
                  Added to Q and K so attention is spatially aware.
        """
        proj_dtype = self.q_proj.weight.dtype
        selected_proj = selected.to(dtype=proj_dtype)
        q_raw = self.q_proj(selected_proj)
        k_raw = self.k_proj(selected_proj)
        v_raw = self.v_proj(selected_proj)

        # Cast weight/bias through to() so autocast keeps them as the same
        # storage object rather than creating a temporary that breaks the
        # autograd connection back to the original parameter.
        norm_w = self.norm.weight.to(dtype=torch.float32) if self.norm.weight is not None else None
        norm_b = self.norm.bias.to(dtype=torch.float32) if self.norm.bias is not None else None
        q = F.layer_norm(
            q_raw.float(),
            self.norm.normalized_shape,
            norm_w,
            norm_b,
            self.norm.eps,
        )
        k = F.layer_norm(
            k_raw.float(),
            self.norm.normalized_shape,
            norm_w,
            norm_b,
            self.norm.eps,
        )
        v = v_raw.float()

        # Inject spatial positions so attention can model object topology.
        # Without positional encoding the selected tokens form an unordered set
        # and cannot learn directional or proximity relationships.
        if pos_enc is not None:
            pe = pos_enc.to(dtype=q.dtype)
            q = q + pe
            k = k + pe

        attn = torch.bmm(q, k.transpose(1, 2)) * float(self._attn_scale)
        attn = _finite_or_zero(attn, limit=self.attn_logit_limit)
        attn = attn - attn.max(dim=-1, keepdim=True).values.detach()
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        attended = torch.bmm(attn, v)
        attended = _bounded_tanh(attended, limit=self.delta_limit)
        out_dtype = self.out_proj.weight.dtype
        delta = self.out_proj(attended.to(dtype=out_dtype)).float()
        delta = _bounded_tanh(delta, limit=self.delta_limit)
        return _finite_or_zero(delta, limit=self.delta_limit).to(dtype=selected.dtype)

    def _capture_selected_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward hook used only for debug metrics."""
        try:
            self.last_selected_grad_norm = float(grad.detach().float().norm().item())
            self.last_nonselected_sparse_grad = 0.0
        except Exception:
            pass
        return grad

    def _effective_gamma(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return signed LayerScale with a small floor so sparse contribution cannot vanish.

        Uses a straight-through estimator (STE) over clamp_min so:
          - Forward : magnitude = max(|gamma|, floor)   — floor is enforced
          - Backward: gradient = sign(gamma)            — never zero, no softplus ln(2) bias

        The previous softplus formula had a fatal flaw: softplus(x) → ln(2) ≈ 0.693 as x → 0,
        so when gamma_raw ≈ floor the effective gamma was pinned at floor + ln(2) ≈ 0.703
        regardless of what gamma_raw was — gamma never learned.
        """
        gamma = self.gamma.to(dtype=dtype, device=device)
        floor = float(getattr(self, "gamma_floor", 0.0))
        if floor <= 0.0:
            return gamma
        sign = torch.where(gamma.detach() >= 0, torch.ones_like(gamma), -torch.ones_like(gamma))
        abs_g = gamma.abs()
        # STE: forward uses clamped magnitude, backward gradient flows through abs_g unchanged.
        mag_hard = abs_g.clamp(min=floor)
        mag = abs_g + (mag_hard - abs_g).detach()   # = mag_hard in forward, grad = sign(gamma)
        return sign * mag

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selected-token-only sparse attention.

        Flow:
          finite guard -> GN -> L2 top-k -> gather selected tokens
          -> Linear Q/K/V -> FP32 attention -> Linear out
          -> zero-canvas scatter -> x + gamma * delta
        """
        self._ensure_runtime_attrs()
        B, C, H, W = x.shape
        N = int(H) * int(W)
        k_actual = self._select_k(N)

        self.last_N = N
        self.last_k = k_actual
        self.last_mode = self.mode
        self.last_selected_ratio = float(k_actual) / float(N) if N else 0.0
        self.last_gamma_raw_abs_mean = float(self.gamma.detach().float().abs().mean().item())
        self.last_gamma_floor = float(getattr(self, "gamma_floor", 0.0))
        self.last_selected_grad_norm = None
        self.last_nonselected_sparse_grad = 0.0

        if self.debug_enabled:
            finite = torch.isfinite(x)
            self.last_finite_guard_count = int((~finite).sum().item())
        else:
            self.last_finite_guard_count = 0

        x_safe = _finite_or_zero(x, limit=self.finite_limit)
        norm_dtype = self.pre_norm.weight.dtype
        x_norm = self.pre_norm(x_safe.to(dtype=norm_dtype)).to(dtype=x_safe.dtype)
        x_norm = _finite_or_zero(x_norm, limit=self.finite_limit)

        if self.mode == "dense":
            sal_scores = None
            saliency = None
            topk_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            k_actual = N
        else:
            sal_scores = self._compute_saliency(x_norm)    # (B, N) — keeps grad
            sal_for_topk = sal_scores.detach()
            if not torch.isfinite(sal_for_topk).any():
                sal_for_topk = torch.zeros_like(sal_for_topk)
            topk_idx = torch.topk(sal_for_topk, k_actual, dim=1).indices
            saliency = sal_for_topk                         # detached copy for debug

        topk_idx = topk_idx.clamp(0, N - 1).long()
        self._store_debug(topk_idx, saliency)

        x_flat = x_norm.view(B, C, N)
        idx_exp = topk_idx.unsqueeze(1).expand(-1, C, -1)
        selected = torch.gather(x_flat, 2, idx_exp).transpose(1, 2)

        # Compute 2D positional encoding for the selected positions so sparse
        # attention is spatially aware (token at top-left vs bottom-right matters).
        pos_enc = self._sinusoidal_pos_enc_2d(topk_idx, H, W, C)

        delta_tokens = self._selected_attention_delta(selected, pos_enc=pos_enc)
        if delta_tokens.requires_grad:
            delta_tokens.register_hook(self._capture_selected_grad)

        # Gradient bridge to saliency_head via additive straight-through perturbation.
        # Multiplicative gating (sal_gate * delta) caused collapse: the model learned to
        # output negative saliency scores → sigmoid → 0 → delta → 0 → no gradient → dead.
        # Instead, add a zero-magnitude STE term: forward contributes nothing, backward
        # carries gradient through saliency_head into backbone so it learns object salience.
        if sal_scores is not None:
            selected_sal = torch.gather(sal_scores, 1, topk_idx)           # (B, k)
            # Scale STE signal to delta magnitude so gradient is proportionate, not dominant.
            delta_scale = delta_tokens.detach().abs().mean().clamp(min=1e-6)
            ste = (selected_sal.unsqueeze(-1) - selected_sal.detach().unsqueeze(-1)) * delta_scale
            delta_tokens = delta_tokens + ste.to(dtype=delta_tokens.dtype)

        self.last_delta_norm_selected = float(
            delta_tokens.detach().float().norm(dim=-1).mean().item()
        )
        self.last_delta_norm_nonselected = 0.0

        delta_flat = x_safe.new_zeros(B, C, N)
        delta_flat = delta_flat.scatter(2, idx_exp, delta_tokens.transpose(1, 2).to(dtype=x_safe.dtype))
        delta = delta_flat.view(B, C, H, W)
        gamma_eff = self._effective_gamma(dtype=x_safe.dtype, device=x_safe.device)
        self.last_gamma_abs_mean = float(gamma_eff.detach().float().abs().mean().item())
        try:
            scaled_tokens = delta_tokens * gamma_eff.flatten().view(1, 1, C).to(dtype=delta_tokens.dtype)
            self.last_delta_scaled_norm_selected = float(
                scaled_tokens.detach().float().norm(dim=-1).mean().item()
            )
        except Exception:
            self.last_delta_scaled_norm_selected = None
        out = x_safe + gamma_eff * delta
        return _finite_or_zero(out, limit=self.finite_limit).to(dtype=x.dtype)


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
        topk_ind = topk_ind.long()

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
            self.tgt_embed.weight.to(dtype=features.dtype).unsqueeze(0).repeat(bs, 1, 1)
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
        # Cast back to original dtype for AMP compatibility
        orig_dtype = embed.dtype
        dec_bboxes = dec_bboxes.to(dtype=orig_dtype)
        dec_scores = dec_scores.to(dtype=orig_dtype)
        enc_bboxes = enc_bboxes.to(dtype=orig_dtype)
        enc_scores = enc_scores.to(dtype=orig_dtype)
        
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

    def _safe_decoder_forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror the base decoder while isolating only box refinement in FP32."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        refer_bbox = _safe_unit_interval(refer_bbox.float().sigmoid(), eps=self.REFINE_EPS)

        for i, layer in enumerate(self.decoder.layers):
            ref_for_layer = _safe_unit_interval(refer_bbox, eps=self.REFINE_EPS)
            query_dtype = _module_param_dtype(self.query_pos_head, output.dtype)
            query_pos = self.query_pos_head(ref_for_layer.to(dtype=query_dtype)).to(dtype=output.dtype)
            output = layer(
                output,
                ref_for_layer.to(dtype=output.dtype),
                feats.to(dtype=output.dtype),
                shapes,
                padding_mask,
                attn_mask,
                query_pos,
            )

            head_input = F.layer_norm(output, (output.shape[-1],))
            bbox_head = self.dec_bbox_head[i]
            bbox_dtype = _module_param_dtype(bbox_head, head_input.dtype)
            bbox = bbox_head(head_input.to(dtype=bbox_dtype)).float()
            bbox = self.BBOX_DELTA_LIMIT * torch.tanh(bbox / self.BBOX_DELTA_LIMIT)
            refined_bbox = torch.sigmoid(
                bbox + _safe_inverse_sigmoid(ref_for_layer, eps=self.REFINE_EPS)
            )
            refined_bbox = _safe_unit_interval(refined_bbox, eps=self.REFINE_EPS)

            score_head = self.dec_score_head[i]
            score_dtype = _module_param_dtype(score_head, head_input.dtype)
            dec_cls.append(score_head(head_input.to(dtype=score_dtype)))

            refer_bbox = refined_bbox.detach() if self.training else refined_bbox
            dec_bboxes.append(refined_bbox)

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
