"""
Focused tests for HSG-DETR selected-token SGTokenBlock.

These tests are skipped automatically when torch/ultralytics are not installed
in the local environment.
"""
from __future__ import annotations

import os
import sys

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from hsg_detr.nn.sparse_global_token import SGTokenBlock  # noqa: E402


def _selected_mask(indices: torch.Tensor, channels: int, height: int, width: int) -> torch.Tensor:
    mask = torch.zeros(indices.shape[0], 1, height * width, dtype=torch.bool, device=indices.device)
    mask.scatter_(2, indices.unsqueeze(1), True)
    return mask.view(indices.shape[0], 1, height, width).expand(-1, channels, -1, -1)


def _sparse_params(block: SGTokenBlock):
    for module in (block.q_proj, block.k_proj, block.v_proj, block.out_proj):
        yield from module.parameters()


def test_sg_token_block_shape_and_gamma_init():
    block = SGTokenBlock(16, 16, ratio=0.25, debug_enabled=True)
    assert block.gamma.shape == (1, 16, 1, 1)
    assert float(block.gamma.detach().mean()) == pytest.approx(0.05)

    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    state = block.get_debug_state()
    assert state["k"] == 16
    assert state["selected_ratio"] == pytest.approx(0.25)


def test_sparse_delta_is_zero_on_nonselected_positions():
    block = SGTokenBlock(8, 8, ratio=0.25, debug_enabled=True)
    x = torch.randn(1, 8, 4, 4)
    y = block(x)
    indices = block.get_debug_state()["indices"]
    mask = _selected_mask(indices, channels=8, height=4, width=4)

    diff = y - x
    assert torch.count_nonzero(diff[~mask]) == 0
    assert block.get_debug_state()["delta_norm_nonselected"] == pytest.approx(0.0)


def test_sparse_params_do_not_update_from_nonselected_loss():
    block = SGTokenBlock(8, 8, ratio=0.25, debug_enabled=True)
    x = torch.randn(1, 8, 4, 4, requires_grad=True)
    y = block(x)
    indices = block.get_debug_state()["indices"]
    mask = _selected_mask(indices, channels=8, height=4, width=4)

    loss = y[~mask].sum()
    loss.backward()

    sparse_grad = [
        p.grad.detach().abs().sum()
        for p in _sparse_params(block)
        if p.grad is not None
    ]
    assert not sparse_grad or sum(sparse_grad).item() == pytest.approx(0.0)


def test_sparse_params_receive_selected_token_gradients():
    block = SGTokenBlock(8, 8, ratio=0.25, debug_enabled=True)
    x = torch.randn(1, 8, 4, 4, requires_grad=True)
    y = block(x)
    indices = block.get_debug_state()["indices"]
    mask = _selected_mask(indices, channels=8, height=4, width=4)

    loss = y[mask].pow(2).sum()
    loss.backward()

    sparse_grad = sum(
        (p.grad.detach().abs().sum() for p in _sparse_params(block) if p.grad is not None),
        torch.tensor(0.0),
    )
    assert sparse_grad.item() > 0
    assert block.get_debug_state()["selected_grad_norm"] is not None
    assert block.get_debug_state()["nonselected_sparse_grad"] == pytest.approx(0.0)
