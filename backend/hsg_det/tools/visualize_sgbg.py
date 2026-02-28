"""
HSG-DET Feature Visualization Tool
====================================
Visualizes internal representations of SparseGlobalBlockGated modules
at P3, P4, P5 scales during inference using PyTorch forward hooks.

Three visualization types per scale:
  1. Top-K Selection Map   — which spatial tokens the module attends to
  2. Attention Heatmap     — pairwise relationship between selected tokens
  3. Refinement Delta      — what the module changed (delta overlay on image)

Usage:
    python visualize_sgbg.py \
        --weights /path/to/best.pt \
        --image   /path/to/image.jpg \
        --output  ./vis_output \
        [--topk 512] [--imgsz 640]

Output directory will contain:
    <stem>_p3_selection.png
    <stem>_p3_attn.png
    <stem>_p3_delta.png
    <stem>_p4_*  ...
    <stem>_p5_*  ...
    <stem>_composite.png   (side-by-side summary)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ── Hook registry ─────────────────────────────────────────────────────────────

class SGBGHooks:
    """
    Attach forward hooks to every SparseGlobalBlockGated in the model.

    Captured per module:
      • topk_idx    : [B, k]   — flat spatial indices of selected tokens
      • attn        : [B, k, k] — softmax attention weights
      • delta       : [B, C, H, W] — the additive refinement before gating
      • x_shape     : (H, W)  — spatial size of the feature map
      • gate_value  : float   — current gate α value
    """

    def __init__(self):
        self.captures: dict[str, dict] = {}  # name → captured tensors
        self._handles = []

    def register(self, model: torch.nn.Module) -> None:
        """Find all SparseGlobalBlockGated submodules and attach hooks.

        Only top-level Gated modules are registered.  The inner SparseGlobalBlock
        (.block) is patched internally by _attach_gated — we must NOT register it
        separately or captures will contain duplicates with colliding slugs.
        """
        from hsg_det.nn.sparse_global import SparseGlobalBlockGated, SparseGlobalBlock

        # Collect names of inner .block submodules inside Gated wrappers so we
        # can skip them when iterating.
        gated_inner_ids: set[int] = set()
        for _, module in model.named_modules():
            if isinstance(module, SparseGlobalBlockGated):
                gated_inner_ids.add(id(module.block))

        for full_name, module in model.named_modules():
            if isinstance(module, SparseGlobalBlockGated):
                self._attach_gated(full_name, module)
            elif isinstance(module, SparseGlobalBlock) and id(module) not in gated_inner_ids:
                # Standalone SparseGlobalBlock (not wrapped by Gated) — rare but possible
                self._attach_basic(full_name, module)

        print(f"[SGBGHooks] Registered hooks on {len(self.captures)} module(s): "
              f"{list(self.captures.keys())}")

    def _attach_gated(self, name: str, module) -> None:
        store = {}
        self.captures[name] = store

        # Patch _attention_delta on the inner SparseGlobalBlock to capture internals
        inner = module.block
        self._patch_inner(inner, store)

        # Also capture gate value and final delta via the gated module's forward hook
        def _fwd_hook(mod, inp, out):
            store["gate_value"] = float(mod.gate.item())
            x = inp[0]
            store["x_shape"] = (x.shape[2], x.shape[3])

        h = module.register_forward_hook(_fwd_hook)
        self._handles.append(h)

    def _attach_basic(self, name: str, module) -> None:
        store = {}
        self.captures[name] = store
        store["gate_value"] = 1.0
        self._patch_inner(module, store)

        def _fwd_hook(mod, inp, out):
            x = inp[0]
            store["x_shape"] = (x.shape[2], x.shape[3])

        h = module.register_forward_hook(_fwd_hook)
        self._handles.append(h)

    def _patch_inner(self, inner_block, store: dict) -> None:
        """Replace _attention_delta with a wrapper that captures intermediates."""
        original_fn = inner_block._attention_delta

        def _instrumented(x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            N = H * W
            k_actual = min(inner_block.k, N)

            # ── Replicate token selection logic ──
            importance = x.view(B, C, N).float().pow(2).sum(dim=1)
            importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
            topk_result = torch.topk(importance, k_actual, dim=1)
            topk_idx = topk_result.indices  # [B, k]

            # ── Run original to get attention weights ──
            # Temporarily swap back to avoid recursion, then restore
            inner_block._attention_delta = original_fn
            delta = original_fn(x)
            inner_block._attention_delta = _instrumented

            # ── Capture attention weights by re-running attention math ──
            with torch.no_grad():
                q = inner_block.q_proj(x).view(B, C, N)
                kk = inner_block.k_proj(x).view(B, C, N)
                idx_exp = topk_idx.unsqueeze(1).expand(-1, C, -1)
                q_sel = torch.gather(q, 2, idx_exp).transpose(1, 2).float()
                k_sel = torch.gather(kk, 2, idx_exp).transpose(1, 2).float()
                norm_w = inner_block.norm.weight
                norm_b = inner_block.norm.bias
                q_sel = F.layer_norm(
                    q_sel, inner_block.norm.normalized_shape,
                    norm_w.float() if norm_w is not None else None,
                    norm_b.float() if norm_b is not None else None,
                    inner_block.norm.eps,
                )
                k_sel = F.layer_norm(
                    k_sel, inner_block.norm.normalized_shape,
                    norm_w.float() if norm_w is not None else None,
                    norm_b.float() if norm_b is not None else None,
                    inner_block.norm.eps,
                )
                scale = C ** -0.5
                attn = torch.bmm(q_sel, k_sel.transpose(1, 2)) * scale
                attn = attn.clamp(-80.0, 80.0)
                attn = attn - attn.max(dim=-1, keepdim=True).values
                attn = torch.softmax(attn, dim=-1)

            # Store on CPU to save VRAM during inference
            store["topk_idx"] = topk_idx.detach().cpu()
            store["attn"] = attn.detach().cpu()
            store["delta"] = delta.detach().cpu()
            store["feature_hw"] = (H, W)
            store["k_actual"] = k_actual

            return delta

        inner_block._attention_delta = _instrumented

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Visualization helpers ──────────────────────────────────────────────────────

def _to_heatmap(arr: np.ndarray, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """Normalise float array → uint8 heatmap BGR."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return cv2.applyColorMap((arr * 255).astype(np.uint8), colormap)


def _overlay(img_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    heatmap_bgr = cv2.resize(heatmap_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap_bgr, alpha, 0)


def vis_selection_map(
    topk_idx: torch.Tensor,  # [B, k]
    feature_hw: tuple[int, int],
    img_bgr: np.ndarray,
    scale_label: str,
    out_path: Path,
) -> np.ndarray:
    """Render Top-K token selection as a sparse dot overlay."""
    H, W = feature_hw
    ih, iw = img_bgr.shape[:2]

    # Batch=0 only (single image inference)
    idx = topk_idx[0].numpy()  # [k]
    mask = np.zeros(H * W, dtype=np.float32)
    mask[idx] = 1.0
    mask = mask.reshape(H, W)

    # Upscale mask to image size
    mask_up = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

    # Overlay selected positions as red dots on the image
    vis = img_bgr.copy()
    vis[mask_up > 0.5] = (0, 0, 220)  # bright red

    # Blend lightly
    result = cv2.addWeighted(img_bgr, 0.4, vis, 0.6, 0)

    # Heatmap version (smooth density if needed)
    hm = _to_heatmap(cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5), cv2.COLORMAP_HOT)
    result_hm = _overlay(img_bgr, hm, alpha=0.55)

    # Title bar
    title_h = 36
    canvas = np.zeros((ih + title_h, iw, 3), dtype=np.uint8)
    canvas[title_h:] = result_hm
    cv2.putText(canvas, f"{scale_label}: Top-K Selection Map  (k={len(idx)})",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f"  [sel]   {out_path.name}")
    return canvas


def vis_attention_heatmap(
    attn: torch.Tensor,       # [B, k, k]
    topk_idx: torch.Tensor,   # [B, k]
    feature_hw: tuple[int, int],
    img_bgr: np.ndarray,
    scale_label: str,
    out_path: Path,
    query_frac: float = 0.5,
    query_pixel: tuple[float, float] | None = None,  # (cx, cy) in image pixels
    query_label: str = "",   # e.g. "car 89%" shown in title
) -> np.ndarray:
    """
    Render attention influence map for a chosen query token.

    Priority:
      query_pixel=(cx,cy) → find the selected token nearest to that image pixel.
      query_frac           → fallback rank-based selection (0=highest importance).
    """
    H, W = feature_hw
    ih, iw = img_bgr.shape[:2]

    a = attn[0].numpy()   # [k, k]
    idx = topk_idx[0].numpy()  # [k]
    k = len(idx)

    # ── Choose query token ────────────────────────────────────────────────────
    if query_pixel is not None:
        cx_img, cy_img = query_pixel
        # Convert image pixel → feature map grid position
        fx = cx_img / iw * W
        fy = cy_img / ih * H
        target_pos = int(fy) * W + int(fx)
        # Find the selected token (in idx) closest to target_pos
        gy = idx // W
        gx = idx % W
        ty, tx = int(fy), int(fx)
        dists = (gy - ty) ** 2 + (gx - tx) ** 2
        q_rank = int(np.argmin(dists))
    else:
        q_rank = int(query_frac * (k - 1))

    q_row = a[q_rank]  # attention weights FROM query q_rank TO all k tokens: [k]

    # Scatter attention weights back to spatial map
    spatial = np.zeros(H * W, dtype=np.float32)
    for j, pos in enumerate(idx):
        spatial[pos] += q_row[j]
    spatial = spatial.reshape(H, W)

    # Mark query position
    q_pos = idx[q_rank]
    q_y, q_x = divmod(int(q_pos), W)

    # Upscale and overlay
    hm = _to_heatmap(spatial, cv2.COLORMAP_JET)
    result = _overlay(img_bgr, hm, alpha=0.6)

    # Mark query pixel on image
    sx = int(q_x / W * iw)
    sy = int(q_y / H * ih)
    cv2.drawMarker(result, (sx, sy), (255, 255, 255), cv2.MARKER_STAR, 18, 2)

    # Also render the raw k×k attention matrix as a small inset
    attn_img = cv2.resize(
        _to_heatmap(a, cv2.COLORMAP_VIRIDIS),
        (min(256, iw // 3), min(256, iw // 3)),
        interpolation=cv2.INTER_NEAREST,
    )
    inset_h, inset_w = attn_img.shape[:2]
    result[-inset_h - 4:-4, -inset_w - 4:-4] = attn_img

    title_h = 36
    canvas = np.zeros((ih + title_h, iw, 3), dtype=np.uint8)
    canvas[title_h:] = result
    qlabel_str = f"  [{query_label}]" if query_label else f"  rank={q_rank}/{k-1}"
    cv2.putText(canvas,
                f"{scale_label}: Attention Map  ★={qlabel_str}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f"  [attn]  {out_path.name}")
    return canvas


def vis_refinement_delta(
    delta: torch.Tensor,       # [B, C, H, W]
    img_bgr: np.ndarray,
    scale_label: str,
    out_path: Path,
    gate_value: float = 1.0,
) -> np.ndarray:
    """
    Visualize the magnitude of the additive refinement from SGBG.

    Shows where the module strengthened or suppressed features.
    Bright = large positive change, dark = suppression.
    """
    ih, iw = img_bgr.shape[:2]

    d = delta[0]  # [C, H, W]

    # L2 magnitude per spatial position → [H, W]
    magnitude = d.float().pow(2).sum(dim=0).sqrt().numpy()

    # Signed max-channel projection (shows direction of strongest channel)
    signed = d.float().max(dim=0).values.numpy()

    # Render magnitude as heatmap overlay
    hm_mag = _to_heatmap(magnitude, cv2.COLORMAP_INFERNO)
    result_mag = _overlay(img_bgr, hm_mag, alpha=0.6)

    # Render signed delta (blue=suppression, red=amplification)
    hm_sign = _to_heatmap(signed, cv2.COLORMAP_TWILIGHT_SHIFTED)
    result_sign = _overlay(img_bgr, hm_sign, alpha=0.55)

    # Side-by-side: magnitude | signed
    combined = np.concatenate([result_mag, result_sign], axis=1)

    title_h = 36
    canvas = np.zeros((ih + title_h, iw * 2, 3), dtype=np.uint8)
    canvas[title_h:] = combined
    label_mag = f"{scale_label}: Δ Magnitude (INFERNO)    gate α={gate_value:.4f}"
    label_sgn = f"{scale_label}: Δ Signed max-ch (TWILIGHT_SHIFTED)"
    cv2.putText(canvas, label_mag, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, label_sgn, (iw + 8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f"  [delta] {out_path.name}")
    return canvas


def vis_composite(panels: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    """Stack all panels vertically into one summary image."""
    if not panels:
        return
    # Normalise widths
    max_w = max(p.shape[1] for _, p in panels)
    rows = []
    for label, p in panels:
        if p.shape[1] < max_w:
            pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
            p = np.concatenate([p, pad], axis=1)
        rows.append(p)
    composite = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), composite)
    print(f"\n[composite] {out_path.name}  ({composite.shape[1]}×{composite.shape[0]})")


# ── Scale name inference ───────────────────────────────────────────────────────

def _infer_scale_label(module_name: str, all_names: list[str]) -> str:
    """Map model layer index → P3/P4/P5 label heuristically."""
    # Sort all captured module names by their first numeric layer index
    def _layer_idx(n: str) -> int:
        import re
        nums = re.findall(r'\d+', n)
        return int(nums[0]) if nums else 999

    sorted_names = sorted(all_names, key=_layer_idx)
    rank = sorted_names.index(module_name) if module_name in sorted_names else 0
    labels = ["P3 (stride 8)", "P4 (stride 16)", "P5 (stride 32)"]
    return labels[min(rank, len(labels) - 1)]


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    weights: str,
    image: str,
    output: str,
    imgsz: int = 640,
    query_frac: float = 0.5,
    device: str = "cpu",
) -> None:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(image)
    stem = img_path.stem

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading weights: {weights}")
    # Register custom modules so YOLO can parse the YAML
    from hsg_det.nn.sparse_global import _register_into_ultralytics
    _register_into_ultralytics()

    from ultralytics import YOLO
    model = YOLO(weights)
    model.model.eval()
    model.model.to(device)

    # ── Attach hooks ────────────────────────────────────────────────────────
    hooks = SGBGHooks()
    hooks.register(model.model)
    if not hooks.captures:
        print("WARNING: No SparseGlobalBlockGated modules found in model.")
        print("  Make sure the model was trained with HSG-DET architecture.")
        sys.exit(1)

    # ── Load & preprocess image ─────────────────────────────────────────────
    img_bgr_orig = cv2.imread(str(img_path))
    if img_bgr_orig is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    ih_orig, iw_orig = img_bgr_orig.shape[:2]

    # Resize for display (keep aspect, max side = imgsz)
    scale = imgsz / max(ih_orig, iw_orig)
    display_w = int(iw_orig * scale)
    display_h = int(ih_orig * scale)
    img_display = cv2.resize(img_bgr_orig, (display_w, display_h))

    # ── Run inference (triggers hooks) ──────────────────────────────────────
    print(f"Running inference on: {img_path.name}  (imgsz={imgsz})")
    with torch.no_grad():
        _ = model.predict(
            str(img_path),
            imgsz=imgsz,
            device=device,
            verbose=False,
            save=False,
        )

    hooks.remove()

    # ── Generate visualizations ─────────────────────────────────────────────
    all_names = list(hooks.captures.keys())
    panels: list[tuple[str, np.ndarray]] = []

    for mod_name, store in hooks.captures.items():
        if "topk_idx" not in store:
            print(f"  [SKIP] {mod_name} — no captures (module may not have fired)")
            continue

        scale_label = _infer_scale_label(mod_name, all_names)
        scale_slug = scale_label.split()[0].lower()  # "p3", "p4", "p5"
        print(f"\n[{scale_label}]  module: {mod_name}")
        print(f"  feature_hw={store['feature_hw']}  k={store['k_actual']}  "
              f"gate α={store.get('gate_value', 1.0):.4f}")

        topk_idx = store["topk_idx"]    # [B, k]
        attn      = store["attn"]        # [B, k, k]
        delta     = store["delta"]       # [B, C, H, W]
        fhw       = store["feature_hw"]  # (H, W)
        gate      = store.get("gate_value", 1.0)

        p_sel   = out_dir / f"{stem}_{scale_slug}_selection.png"
        p_attn  = out_dir / f"{stem}_{scale_slug}_attn.png"
        p_delta = out_dir / f"{stem}_{scale_slug}_delta.png"

        pan_sel   = vis_selection_map(topk_idx, fhw, img_display, scale_label, p_sel)
        pan_attn  = vis_attention_heatmap(attn, topk_idx, fhw, img_display, scale_label, p_attn, query_frac)
        pan_delta = vis_refinement_delta(delta, img_display, scale_label, p_delta, gate)

        panels += [
            (f"{scale_label} Selection", pan_sel),
            (f"{scale_label} Attention", pan_attn),
            (f"{scale_label} Delta",     pan_delta),
        ]

    vis_composite(panels, out_dir / f"{stem}_composite.png")
    print(f"\nAll visualizations saved to: {out_dir}/")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="HSG-DET SGBG Feature Visualizer")
    p.add_argument("--weights", required=True, help="Path to trained .pt weights")
    p.add_argument("--image",   required=True, help="Input image path")
    p.add_argument("--output",  default="./vis_output", help="Output directory")
    p.add_argument("--imgsz",   type=int, default=640, help="Inference image size")
    p.add_argument("--query-frac", type=float, default=0.5,
                   help="Attention heatmap query rank fraction [0.0=top, 1.0=bottom]")
    p.add_argument("--device",  default="cpu", help="Device: cpu / cuda / cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        weights=args.weights,
        image=args.image,
        output=args.output,
        imgsz=args.imgsz,
        query_frac=args.query_frac,
        device=args.device,
    )
