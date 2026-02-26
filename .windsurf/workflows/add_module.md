---
description: Add a new custom neural-network block to Model Designer (implement → register → UI → validate → test)
---

## Context

This project uses a **3-layer registration system** for custom blocks:

1. `backend/hsg_det/nn/` — PyTorch `nn.Module` implementations
2. `backend/app/services/module_registry.py` — exposes blocks to the Designer UI
3. `backend/app/plugins/archs/hsg_det.py` → `register_modules()` — injects into Ultralytics before training

---

## Step 1 — Implement the nn.Module

Create `backend/hsg_det/nn/<block_name>.py`:

```python
import torch, torch.nn as nn

class MyBlock(nn.Module):
    """Channel-preserving block. out_shape == in_shape."""
    def __init__(self, c: int, k: int = 512) -> None:
        super().__init__()
        # ... layers ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]  →  returns same shape
        return x + out  # residual pattern

class MyBlockGated(nn.Module):
    """Gated version — starts as identity (gate=0), safe for pretrained warm-start."""
    def __init__(self, c: int, k: int = 512) -> None:
        super().__init__()
        self.block = MyBlock(c, k)
        self.gate = nn.Parameter(torch.zeros(1))   # ← MUST be zeros

    def forward(self, x):
        return x + self.gate * (self.block(x) - x)

def _register_into_ultralytics() -> None:
    """Inject into ultralytics.nn.modules — idempotent, safe to call many times."""
    try:
        import ultralytics.nn.modules as ulm
        ulm.MyBlock = MyBlock
        ulm.MyBlockGated = MyBlockGated
    except ImportError:
        pass
```

---

## Step 2 — Register into `hsg_det/nn/__init__.py`

```python
from .my_block import MyBlock, MyBlockGated, _register_into_ultralytics
_register_into_ultralytics()
__all__ = ["MyBlock", "MyBlockGated"]
```

---

## Step 3 — Add to `module_registry.py`

Open `backend/app/services/module_registry.py`.
Append to the **`_HSG_DET_MODULES`** list:

```python
{
    "name": "MyBlock",                    # exact class name
    "category": "attention",             # attention | composite | basic | head
    "description": "Short description for the UI panel.",
    "args": [
        {"name": "out_channels", "type": "int",   "default": 256},
        {"name": "k",            "type": "int",   "default": 512},
    ],
    "source": "hsg_det",
},
{
    "name": "MyBlockGated",
    "category": "attention",
    "description": "Gated version — starts as identity, recommended for warm-start training.",
    "args": [
        {"name": "out_channels", "type": "int", "default": 256},
        {"name": "k",            "type": "int", "default": 512},
    ],
    "source": "hsg_det",
},
```

---

## Step 4 — Validate (no pytest required)

```bash
cd backend
python3 -c "
import sys; sys.path.insert(0,'.')
import torch
from hsg_det.nn.my_block import MyBlock, MyBlockGated

x = torch.randn(2, 256, 34, 60)

# Shape
y = MyBlock(256)(x)
assert y.shape == x.shape, 'shape mismatch'
print('✓ shape OK')

# Gated identity
g = MyBlockGated(256)
assert g.gate.item() == 0.0
import torch.testing; torch.testing.assert_close(g(x), x, atol=1e-5, rtol=0)
print('✓ gated identity OK')

# Gradient
x2 = torch.randn(1,256,8,8,requires_grad=True)
MyBlock(256)(x2).sum().backward()
assert x2.grad is not None
print('✓ gradient flows')
print('ALL PASSED')
"
```

---

## Step 5 — Enable Validation (Critical)

In `backend/app/controllers/model_controller.py`:

1. **Add Import**: `import hsg_det` (so YOLO finds the module)
2. **Patch Args**: Manually scale channel args in `validate_model` because `parse_model` won't.

```python
if scale:
    # ...
    if module == "MyBlock":
        args[0] = make_divisible(min(args[0], max_ch) * width, 8)
```

---

## Step 6 — Write Tests

Add a class to `backend/tests/test_hsg_det.py`:

```python
class TestMyBlock:
    def setup_method(self):
        from hsg_det.nn.my_block import MyBlock, MyBlockGated
        self.Block = MyBlock
        self.GatedBlock = MyBlockGated

    def test_output_shape(self):
        b = self.Block(256)
        x = torch.randn(2, 256, 34, 60)
        assert b(x).shape == x.shape

    def test_gated_identity_at_init(self):
        g = self.GatedBlock(32)
        assert g.gate.item() == pytest.approx(0.0)
        x = torch.randn(1, 32, 8, 8)
        torch.testing.assert_close(g(x), x, atol=1e-5, rtol=0)

    def test_gradient_flows(self):
        b = self.Block(32)
        x = torch.randn(1, 32, 8, 8, requires_grad=True)
        b(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)
```

Run:

```bash
cd backend
python3 -m pytest tests/test_hsg_det.py -v --tb=short
```

All tests must pass with **0 FAILED**.

---

## Step 7 — Confirm in Registry

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from app.services.module_registry import get_all_modules
for m in get_all_modules():
    if m['source'] == 'hsg_det':
        print(f\"[{m['category']}] {m['name']}\")
"
```

---

## Checklist

- [ ] `nn.Module` + gated variant implemented in `hsg_det/nn/`
- [ ] `_register_into_ultralytics()` present and idempotent
- [ ] Import added to `hsg_det/nn/__init__.py`
- [ ] Two entries added to `_HSG_DET_MODULES` in `module_registry.py`
- [ ] Step 4 smoke test passes with `ALL PASSED`
- [ ] `validate_model` patched (import + arg scaling) in `model_controller.py`
- [ ] Test class added in `tests/test_hsg_det.py`
- [ ] `pytest tests/test_hsg_det.py` — **0 FAILED**

---

## Common Mistakes

| Mistake                                | Fix                                                         |
| -------------------------------------- | ----------------------------------------------------------- |
| `unknown module MyBlock` in YAML parse | `_register_into_ultralytics()` must run before `YOLO(yaml)` |
| Block missing from Designer panel      | Check `_HSG_DET_MODULES` list + restart backend             |
| Gate not identity at start             | Use `nn.Parameter(torch.zeros(1))`                          |
| Output shape mismatch                  | Use `return x + out` residual pattern                       |
| Registry tests fail after clear        | Use `setup_method` save/restore (see existing tests)        |
