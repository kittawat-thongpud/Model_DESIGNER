#!/usr/bin/env python3
"""Smoke tests for HSG-DETR selected-token sparse debug flow.

Run from the repo root:

    venv/bin/python scripts/test_hsg_detr_sparse_debug.py

The script avoids pytest so it can be used on a training box with only the
project runtime dependencies installed.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _print_step(name: str) -> None:
    print(f"\n[TEST] {name}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _selected_mask(torch, indices, channels: int, height: int, width: int):
    mask = torch.zeros(indices.shape[0], 1, height * width, dtype=torch.bool, device=indices.device)
    mask.scatter_(2, indices.unsqueeze(1), True)
    return mask.view(indices.shape[0], 1, height, width).expand(-1, channels, -1, -1)


def _sparse_params(block):
    for module in (block.q_proj, block.k_proj, block.v_proj, block.out_proj):
        yield from module.parameters()


def _sparse_grad_abs(torch, block) -> float:
    total = torch.tensor(0.0, device=block.gamma.device)
    for param in _sparse_params(block):
        if param.grad is not None:
            total = total + param.grad.detach().abs().sum()
    return float(total.item())


def test_sg_token_contract(torch, device: str) -> None:
    from hsg_detr.nn.sparse_global_token import SGTokenBlock

    _print_step("SGTokenBlock gradient contract")
    torch.manual_seed(7)

    block = SGTokenBlock(8, 8, ratio=0.25, debug_enabled=True).to(device)
    x = torch.randn(1, 8, 4, 4, device=device, requires_grad=True)
    y = block(x)
    state = block.get_debug_state()
    mask = _selected_mask(torch, state["indices"].to(device), channels=8, height=4, width=4)

    _require(y.shape == x.shape, f"shape changed: {tuple(x.shape)} -> {tuple(y.shape)}")
    _require(torch.isfinite(y).all().item(), "forward output contains NaN/Inf")
    _require(abs(float(block.gamma.detach().mean()) - 0.05) < 1e-6, "gamma_init is not 0.05")
    _require(state["k"] == 4 and state["N"] == 16, f"unexpected top-k metadata: {state}")

    diff = y - x
    nonselected_delta = float(diff[~mask].detach().abs().max().item()) if (~mask).any() else 0.0
    _require(nonselected_delta == 0.0, f"non-selected positions received sparse delta: {nonselected_delta}")
    _require(abs(float(state["delta_norm_nonselected"])) == 0.0, "delta_norm_nonselected must be zero")

    loss = y[~mask].sum()
    loss.backward()
    nonselected_sparse_grad = _sparse_grad_abs(torch, block)
    _require(nonselected_sparse_grad == 0.0, f"sparse params updated from non-selected loss: {nonselected_sparse_grad}")

    block.zero_grad(set_to_none=True)
    x = torch.randn(1, 8, 4, 4, device=device, requires_grad=True)
    y = block(x)
    state = block.get_debug_state()
    mask = _selected_mask(torch, state["indices"].to(device), channels=8, height=4, width=4)
    selected_loss = y[mask].pow(2).sum()
    selected_loss.backward()

    selected_sparse_grad = _sparse_grad_abs(torch, block)
    _require(selected_sparse_grad > 0.0, "sparse q/k/v/out params did not receive selected-token gradients")
    _require(block.get_debug_state()["selected_grad_norm"] is not None, "selected_grad_norm hook did not fire")
    _require(block.get_debug_state()["nonselected_sparse_grad"] == 0.0, "nonselected_sparse_grad must remain zero")

    print("PASS selected tokens update sparse params; non-selected tokens do not.")


def test_amp_forward(torch, device: str) -> None:
    from hsg_detr.nn.sparse_global_token import SGTokenBlock

    _print_step("AMP forward finite check")
    block = SGTokenBlock(16, 16, ratio=0.25, debug_enabled=True).to(device).eval()
    x = torch.randn(2, 16, 8, 8, device=device)
    amp_device = "cuda" if device.startswith("cuda") else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16

    with torch.no_grad(), torch.amp.autocast(amp_device, dtype=amp_dtype):
        y = block(x)

    _require(y.shape == x.shape, "AMP forward changed shape")
    _require(torch.isfinite(y).all().item(), "AMP forward produced NaN/Inf")
    print(f"PASS autocast device={amp_device} dtype={amp_dtype}.")


def test_model_forward(torch, device: str, imgsz: int) -> None:
    import hsg_detr  # noqa: F401 - registers HSG-DETR modules with Ultralytics
    from ultralytics import RTDETR

    _print_step("HSG-DETR-N instantiate + eval forward")
    cfg = ROOT / "backend" / "hsg_detr" / "configs" / "hsg_detr_n.yaml"
    model = RTDETR(str(cfg)).model.to(device).eval()
    sgb_blocks = [m for m in model.modules() if m.__class__.__name__ == "SGTokenBlock"]
    _require(len(sgb_blocks) > 0, "model has no SGTokenBlock modules")
    _require(all(abs(float(b.gamma.detach().mean()) - 0.05) < 1e-6 for b in sgb_blocks), "some SGB gamma values are not initialized to 0.05")

    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    amp_device = "cuda" if device.startswith("cuda") else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16
    with torch.no_grad(), torch.amp.autocast(amp_device, dtype=amp_dtype):
        out = model(x)

    tensor_out = out[0] if isinstance(out, (tuple, list)) else out
    _require(torch.is_tensor(tensor_out), f"unexpected model output type: {type(out)!r}")
    _require(torch.isfinite(tensor_out).all().item(), "model forward produced NaN/Inf")
    print(f"PASS blocks={len(sgb_blocks)} output_shape={tuple(tensor_out.shape)}.")


def test_optimizer_groups_and_grad_regions(torch, device: str, imgsz: int) -> None:
    import hsg_detr  # noqa: F401 - registers HSG-DETR modules with Ultralytics
    from ultralytics import RTDETR
    from app.services.custom_trainer import CustomDetectionTrainer

    _print_step("Custom trainer optimizer grouping")
    cfg = ROOT / "backend" / "hsg_detr" / "configs" / "hsg_detr_n.yaml"
    model = RTDETR(str(cfg)).model.to(device)
    trainer = CustomDetectionTrainer.__new__(CustomDetectionTrainer)
    trainer.args = SimpleNamespace(lr0=1e-3)
    trainer.job_id = None

    optimizer = trainer.build_optimizer(model, name="AdamW", lr=1e-3, decay=1e-4)
    names = [group.get("name") for group in optimizer.param_groups]
    sizes = {group.get("name", "?"): len(group["params"]) for group in optimizer.param_groups}

    for expected in ("base", "sgb_sparse", "sgb_gamma", "norm_bias", "decoder"):
        _require(expected in names, f"missing optimizer group: {expected}; got {names}")
        _require(sizes[expected] > 0, f"optimizer group is empty: {expected}")

    gamma_group = next(g for g in optimizer.param_groups if g.get("name") == "sgb_gamma")
    sparse_group = next(g for g in optimizer.param_groups if g.get("name") == "sgb_sparse")
    _require(gamma_group["weight_decay"] == 0.0, "sgb_gamma must not use weight decay")
    _require(abs(float(sparse_group["lr"]) - 2.0e-3) < 1e-12, "sgb_sparse lr multiplier should be 2.0x")
    _require(abs(float(gamma_group["lr"]) - 5.0e-3) < 1e-12, "sgb_gamma lr multiplier should be 5.0x")
    print(f"PASS optimizer groups={sizes}.")

    _print_step("Custom trainer grad region diagnostics")
    trainer.model = model
    model.eval()
    model.zero_grad(set_to_none=True)
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    out = model(x)
    tensor_out = out[0] if isinstance(out, (tuple, list)) else out
    loss = tensor_out.float().sum()
    loss.backward()
    trainer._cache_last_step_grad_norms()
    norms = getattr(trainer, "_last_grad_norms", {})

    for expected in ("backbone", "neck", "decoder", "sgb_sparse", "sgb_gamma"):
        _require(expected in norms, f"missing grad diagnostic region: {expected}; got {norms}")
        _require(float(norms[expected]) > 0.0, f"grad diagnostic region is zero: {expected}; got {norms}")
    print(f"PASS grad diagnostics={norms}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if _cuda_available() else "cpu", help="torch device, e.g. cpu or cuda:0")
    parser.add_argument("--imgsz", type=int, default=256, help="image size for RT-DETR smoke forward")
    parser.add_argument("--skip-model", action="store_true", help="skip RT-DETR instantiate/forward smoke test")
    parser.add_argument("--skip-trainer", action="store_true", help="skip custom trainer optimizer grouping test")
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def main() -> int:
    os.environ.setdefault("PYTHONWARNINGS", "default")
    args = parse_args()

    try:
        import torch
        import ultralytics  # noqa: F401
    except Exception as exc:
        print(f"Missing runtime dependency: {exc}", file=sys.stderr)
        return 2

    try:
        test_sg_token_contract(torch, args.device)
        test_amp_forward(torch, args.device)
        if not args.skip_model:
            test_model_forward(torch, args.device, args.imgsz)
        if not args.skip_trainer:
            test_optimizer_groups_and_grad_regions(torch, args.device, args.imgsz)
    except Exception:
        traceback.print_exc()
        return 1

    print("\nALL HSG-DETR sparse debug smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
