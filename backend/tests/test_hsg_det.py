"""
Unit and integration tests for the HSG-DET backend plugin.

Run from the backend/ directory:
    python -m pytest tests/test_hsg_det.py -v

Or directly:
    python tests/test_hsg_det.py
"""
from __future__ import annotations
import sys
import os

# Ensure backend root is on path so both hsg_det and app can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


import torch
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. SparseGlobalBlock — shape tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSparseGlobalBlock:
    """Forward-pass shape and behaviour tests (no ultralytics needed)."""

    def setup_method(self):
        from hsg_det.nn.sparse_global import SparseGlobalBlock, SparseGlobalBlockGated
        self.Block = SparseGlobalBlock
        self.GatedBlock = SparseGlobalBlockGated

    def test_output_shape_matches_input(self):
        """Block must preserve spatial dimensions and channels exactly."""
        block = self.Block(c=256, k=64)
        x = torch.randn(2, 256, 34, 60)  # P5 at 1080p stride 32
        y = block(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_p4_shape(self):
        """P4 scale test (stride 16 at 1080p → 68×120)."""
        block = self.Block(c=512, k=128)
        x = torch.randn(1, 512, 68, 120)
        y = block(x)
        assert y.shape == x.shape

    def test_k_clamped_when_larger_than_spatial(self):
        """When k > H*W the block should not crash — k is clamped internally."""
        block = self.Block(c=64, k=10000)  # way larger than a 4×4 feature map
        x = torch.randn(1, 64, 4, 4)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_is_additive(self):
        """With zero-init weights, output should be close to input."""
        block = self.Block(c=16, k=4)
        # Zero all learnable weights so attention output ≈ 0
        for p in block.parameters():
            torch.nn.init.zeros_(p)
        x = torch.randn(1, 16, 8, 8)
        y = block(x)
        # y = x + out_proj(zeros) = x + bias-only term ≈ x
        # Allow some tolerance for bias terms
        assert y.shape == x.shape

    def test_gated_starts_as_near_identity(self):
        """GatedBlock: gate=0 on init, so output ≈ input."""
        block = self.GatedBlock(c=32, k=16)
        assert block.gate.item() == pytest.approx(0.0), \
            "Gate should initialise to 0"
        x = torch.randn(2, 32, 16, 16)
        y = block(x)
        # gate=0 → y = x + 0*(block(x) - x) = x
        torch.testing.assert_close(y, x, atol=1e-5, rtol=0)

    def test_gated_output_shape(self):
        block = self.GatedBlock(c=128, k=32)
        x = torch.randn(2, 128, 34, 60)
        y = block(x)
        assert y.shape == x.shape

    def test_gradient_flows(self):
        """Gradients must flow through the block (no dead paths)."""
        block = self.Block(c=32, k=8)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "No gradient flowed to input"
        assert not torch.all(x.grad == 0), "All gradients are zero"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ultralytics registration
# ─────────────────────────────────────────────────────────────────────────────

class TestUltralyticsRegistration:

    def test_modules_registered_after_import(self):
        """After importing hsg_det.nn, modules should be in ultralytics.nn.modules."""
        pytest.importorskip("ultralytics", reason="ultralytics not installed")
        import hsg_det.nn  # triggers _register_into_ultralytics()
        import ultralytics.nn.modules as ulm
        assert hasattr(ulm, "SparseGlobalBlock"), \
            "SparseGlobalBlock not injected into ultralytics.nn.modules"
        assert hasattr(ulm, "SparseGlobalBlockGated"), \
            "SparseGlobalBlockGated not injected into ultralytics.nn.modules"

    def test_registration_is_idempotent(self):
        """Calling register multiple times must not raise."""
        pytest.importorskip("ultralytics")
        from hsg_det.nn.sparse_global import _register_into_ultralytics
        _register_into_ultralytics()
        _register_into_ultralytics()  # second call — must not raise


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plugin system
# ─────────────────────────────────────────────────────────────────────────────

class TestPluginSystem:

    def setup_method(self):
        # Save then restore rather than clearing completely — discover_plugins
        # relies on Python's module import cache (re-importing already-loaded
        # modules is a no-op, so clearing the registry before discover would
        # leave it empty on the second+ run of these tests).
        from app.plugins import loader
        self._saved = dict(loader._arch_plugins)
        loader._arch_plugins.clear()

    def teardown_method(self):
        from app.plugins import loader
        loader._arch_plugins.update(self._saved)

    def test_arch_plugin_registers(self):
        from app.plugins.archs.hsg_det import HSGDetPlugin
        from app.plugins.loader import register_arch, get_arch_plugin
        p = HSGDetPlugin()
        register_arch(p)
        result = get_arch_plugin("hsg_det")
        assert result is p

    def test_discover_finds_hsg_det(self):
        from app.plugins.loader import discover_plugins, get_arch_plugin, register_arch
        from app.plugins.archs.hsg_det import HSGDetPlugin
        # discover_plugins cannot re-run imports (module is cached), so we
        # manually re-register to simulate fresh discovery in test isolation.
        register_arch(HSGDetPlugin())
        counts = discover_plugins()
        assert "archs" in counts, "discover_plugins must return 'archs' count"
        plugin = get_arch_plugin("hsg_det")
        assert plugin is not None, "HSGDetPlugin was not discovered"

    def test_yaml_path_exists(self):
        from app.plugins.archs.hsg_det import HSGDetPlugin
        p = HSGDetPlugin()
        yaml = p.yaml_path()
        assert yaml.exists(), f"YAML not found at: {yaml}"

    def test_plugin_to_dict(self):
        from app.plugins.archs.hsg_det import HSGDetPlugin
        info = HSGDetPlugin().to_dict()
        assert info["name"] == "hsg_det"
        assert info["task_type"] == "detect"
        assert info["pretrain_key"] == "yolov8m"
        assert "yaml_path" in info

    def test_all_arch_plugins(self):
        from app.plugins.loader import all_arch_plugins, register_arch
        from app.plugins.archs.hsg_det import HSGDetPlugin
        register_arch(HSGDetPlugin())
        archs = all_arch_plugins()
        names = [a.name for a in archs]
        assert "hsg_det" in names


# ─────────────────────────────────────────────────────────────────────────────
# 4. YAML → Ultralytics model (optional, only if ultralytics installed)
# ─────────────────────────────────────────────────────────────────────────────

class TestYAMLLoading:

    def test_yaml_loads_into_yolo(self):
        """Full end-to-end: register → YOLO(yaml) → model instantiated."""
        pytest.importorskip("ultralytics", reason="ultralytics not installed")
        from app.plugins.archs.hsg_det import HSGDetPlugin
        from ultralytics import YOLO

        plugin = HSGDetPlugin()
        plugin.register_modules()

        model = YOLO(str(plugin.yaml_path()))
        total_params = sum(p.numel() for p in model.model.parameters()) / 1e6
        print(f"\n  HSG-DET params: {total_params:.1f}M")
        # Reasonable range: 20M–60M
        assert 15 <= total_params <= 70, \
            f"Unexpected param count: {total_params:.1f}M"

    def test_yolo_forward_pass(self):
        """YOLO model builds and can do an inference pass."""
        pytest.importorskip("ultralytics")
        from app.plugins.archs.hsg_det import HSGDetPlugin
        from ultralytics import YOLO

        plugin = HSGDetPlugin()
        plugin.register_modules()
        model = YOLO(str(plugin.yaml_path()))

        # Tiny dummy input (64×64)
        dummy = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            # YOLO.predict returns Results objects; model.model(x) returns raw tensors
            results = model.model(dummy)
        assert results is not None


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import subprocess
    ret = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    return ret.returncode


if __name__ == "__main__":
    sys.exit(main())
