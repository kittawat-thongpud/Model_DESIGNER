"""
Cross-Scale Sparse Global Attention (CS²GA) for HSG-DETR.

Replaces per-scale SparseGlobalTokenBlock with a single module that receives
P3/P4/P5 simultaneously, projects to shared dim, selects top-k tokens per scale,
adds learned scale embeddings, runs cross-scale self-attention, and scatters back
with LayerScale residuals.

Residual design — LayerScale (DeiT/CaiT pattern):
    out = input + layer_scale * delta
    layer_scale: per-channel Parameter(C, 1, 1), init=0.1
    Starts at 10% contribution so attention receives meaningful gradients
    from epoch 1. Grows freely without sigmoid saturation.
    Note: original DeiT uses 1e-4 for pure-transformer stability, but YOLO+CS²GA
    has a strong FPN residual path — 1e-4 starves attention of gradient entirely.

    delta_norm removed: GroupNorm on delta was suppressing delta magnitude even as
    gradients tried to grow it. tanh clamp (±6) still prevents explosion.

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
        ls_init: LayerScale init value. Default 0.1 — attention contributes
            ~10% from epoch 1 so gradient flows behind the strong FPN residual.
            (DeiT uses 1e-4 for pure transformers, but that starves CS²GA here.)
        score_mode: Token-selection criterion. Default "l2" reproduces the
            original behaviour exactly (rank by activation energy ‖t‖₂).
              • "l2"      — raw activation energy (no learnable params)
              • "learned" — rank by a learned per-token logit (semantic selection)
              • "hybrid"  — standardized L2 + learned correction; zero-init head
                            ⇒ selection == L2 at start, then learns a shift.
        gate_mode: Selected-token re-weighting. Default "hard" = no gate.
              • "hard" — selected tokens used as-is (original behaviour)
              • "soft" — multiply each selected token by sigmoid(logit+gate_init_bias).
                         This is the differentiable path that trains the score head
                         (hard top-k alone passes no gradient to it).
        gate_init_bias: Bias added inside the soft gate so it starts ≈1 (open).
            sigmoid(4.0)≈0.98 ⇒ near no-op at init for clean checkpoint transfer.
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
        ls_init: float = 0.1,
        score_mode: str = "l2",
        gate_mode: str = "hard",
        debug: bool = False,
        scale_embed_alpha: float = 0.0,
        attn_scale_mult: float = 8.0,
        max_k3: int = 0,
        max_k4: int = 0,
        max_k5: int = 0,
        gate_init_bias: float = 4.0,
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
        self.score_mode = str(score_mode).lower()
        self.gate_mode = str(gate_mode).lower()
        self.gate_init_bias = float(gate_init_bias)
        assert self.score_mode in ("l2", "learned", "hybrid"), \
            f"score_mode must be l2|learned|hybrid, got {self.score_mode!r}"
        assert self.gate_mode in ("hard", "soft"), \
            f"gate_mode must be hard|soft, got {self.gate_mode!r}"
        # A learnable score head is needed when selection is learned/hybrid OR
        # when the soft gate needs a per-token logit to weight selected tokens.
        self.use_score_head = (
            self.score_mode in ("learned", "hybrid") or self.gate_mode == "soft"
        )
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
        self.last_gate_mean: float | None = None

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

        # LayerScale — per-channel (C, 1, 1), init=ls_init (default 0.1)
        # Attention contributes ~10% from epoch 1, grows freely from there.
        # 1e-4 (DeiT default) starves gradient in YOLO+FPN context — use 0.1.
        self.ls_p3 = nn.Parameter(torch.full((self.c1[0], 1, 1), ls_init))
        self.ls_p4 = nn.Parameter(torch.full((self.c1[1], 1, 1), ls_init))
        self.ls_p5 = nn.Parameter(torch.full((self.c1[2], 1, 1), ls_init))

        # Learnable per-token score head — maps a shared-dim token → scalar logit.
        # Used for (a) learned/hybrid top-k selection and (b) the soft gate.
        # Built only when needed so the default l2/hard path adds zero params and
        # keeps the state_dict identical to the original module (clean transfer).
        if self.use_score_head:
            self.score_p3 = nn.Linear(d, 1)
            self.score_p4 = nn.Linear(d, 1)
            self.score_p5 = nn.Linear(d, 1)
            if self.score_mode == "learned":
                # Pure learned ranking — keep default (small random) init so the
                # top-k ordering is non-degenerate from the first step.
                pass
            else:
                # hybrid selection and/or gate-only: zero-init the head so the
                # logit z≈0 at start. Selection then equals the L2 ranking and
                # the soft gate equals sigmoid(gate_init_bias)≈1 — i.e. the module
                # starts ≈ identical to the l2/hard baseline (minimal-impact init,
                # so a warm-started CS²GA checkpoint transfers cleanly).
                for _h in (self.score_p3, self.score_p4, self.score_p5):
                    nn.init.zeros_(_h.weight)
                    nn.init.zeros_(_h.bias)

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

            # L2 saliency (always computed — used for debug metrics, for the
            # "l2" selection mode, and as the standardized base of "hybrid").
            # Do not use _nan_guard here: clamping all high-energy scores to 20
            # destroys rank ordering and makes top-k arbitrary on activated maps.
            with torch.no_grad():
                p3_scores = _score_guard(p3_tok.float().norm(dim=-1))
                p4_scores = _score_guard(p4_tok.float().norm(dim=-1))
                p5_scores = _score_guard(p5_tok.float().norm(dim=-1))

            # Learnable per-token logit (differentiable — NOT under no_grad so the
            # soft gate can back-propagate into the score head). Semantic token
            # scoring replaces / corrects raw activation-energy selection.
            z3 = z4 = z5 = None
            if self.use_score_head:
                z3 = F.linear(p3_tok, self.score_p3.weight.float(), self.score_p3.bias.float()).squeeze(-1)
                z4 = F.linear(p4_tok, self.score_p4.weight.float(), self.score_p4.bias.float()).squeeze(-1)
                z5 = F.linear(p5_tok, self.score_p5.weight.float(), self.score_p5.bias.float()).squeeze(-1)

            # Ranking score per mode (used only for top-k ordering — detached).
            #   l2     : raw activation energy (original behaviour)
            #   learned: pure learned logit
            #   hybrid : standardized L2 + learned correction (≈L2 at zero-init →
            #            preserves warm-start selection, learns a semantic shift)
            def _std(s: torch.Tensor) -> torch.Tensor:
                m = s.mean(dim=-1, keepdim=True)
                sd = s.std(dim=-1, keepdim=True).clamp_min(1e-6)
                return (s - m) / sd

            if self.score_mode == "l2":
                rank3, rank4, rank5 = p3_scores, p4_scores, p5_scores
            elif self.score_mode == "learned":
                rank3, rank4, rank5 = z3.detach(), z4.detach(), z5.detach()
            else:  # hybrid
                with torch.no_grad():
                    rank3 = _std(p3_scores) + z3.detach()
                    rank4 = _std(p4_scores) + z4.detach()
                    rank5 = _std(p5_scores) + z5.detach()

            k3_raw = max(1, min(int(self.ratio_p3 * p3_tok.shape[1]), p3_tok.shape[1]))
            k4_raw = max(1, min(int(self.ratio_p4 * p4_tok.shape[1]), p4_tok.shape[1]))
            k5_raw = max(1, min(int(self.ratio_p5 * p5_tok.shape[1]), p5_tok.shape[1]))
            k3 = min(k3_raw, self.max_k3) if self.max_k3 > 0 else k3_raw
            k4 = min(k4_raw, self.max_k4) if self.max_k4 > 0 else k4_raw
            k5 = min(k5_raw, self.max_k5) if self.max_k5 > 0 else k5_raw
            k3 = max(1, k3)
            k4 = max(1, k4)
            k5 = max(1, k5)

            _, p3_idx = torch.topk(rank3, k3, dim=-1)
            _, p4_idx = torch.topk(rank4, k4, dim=-1)
            _, p5_idx = torch.topk(rank5, k5, dim=-1)

            # Gather selected tokens
            p3_sel = torch.gather(p3_tok, 1, p3_idx.unsqueeze(-1).expand(-1, -1, d))
            p4_sel = torch.gather(p4_tok, 1, p4_idx.unsqueeze(-1).expand(-1, -1, d))
            p5_sel = torch.gather(p5_tok, 1, p5_idx.unsqueeze(-1).expand(-1, -1, d))

            # Soft gate (differentiable): weight each selected token by
            # sigmoid(logit + gate_init_bias). This is the gradient path that
            # trains the score head — the hard top-k itself is non-differentiable.
            # With zero-init head + gate_init_bias≈4, gates start ≈0.98 (near no-op).
            self.last_gate_mean = None
            if self.gate_mode == "soft" and self.use_score_head:
                g3 = torch.sigmoid(torch.gather(z3, 1, p3_idx) + self.gate_init_bias).unsqueeze(-1)
                g4 = torch.sigmoid(torch.gather(z4, 1, p4_idx) + self.gate_init_bias).unsqueeze(-1)
                g5 = torch.sigmoid(torch.gather(z5, 1, p5_idx) + self.gate_init_bias).unsqueeze(-1)
                p3_sel = p3_sel * g3
                p4_sel = p4_sel * g4
                p5_sel = p5_sel * g5
                if self.debug_enabled:
                    with torch.no_grad():
                        self.last_gate_mean = float(
                            (g3.mean() + g4.mean() + g5.mean()).item() / 3.0
                        )

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

            # tanh clamp before out_proj — prevents explosion without capping gradient
            p3_delta = 6.0 * torch.tanh(p3_delta / 6.0)
            p4_delta = 6.0 * torch.tanh(p4_delta / 6.0)
            p5_delta = 6.0 * torch.tanh(p5_delta / 6.0)

            # out_proj
            p3_delta = F.conv2d(p3_delta, self.out_proj_p3.weight.float())
            p4_delta = F.conv2d(p4_delta, self.out_proj_p4.weight.float())
            p5_delta = F.conv2d(p5_delta, self.out_proj_p5.weight.float())

            # Re-apply sparse mask (no delta_norm — removed to let delta grow freely)
            p3_delta = p3_delta * p3_mask
            p4_delta = p4_delta * p4_mask
            p5_delta = p5_delta * p5_mask

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
            "gate_mean": self.last_gate_mean,
        }
