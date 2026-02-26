# FLOPs Calculation Methods

## Overview

ระบบรองรับ 2 วิธีในการคำนวณ FLOPs:

1. **Ultralytics model.info()** (Default) - รวดเร็ว, ใช้งานง่าย
2. **thop library** (Optional) - ละเอียด, แยกตาม layer

---

## Method 1: Ultralytics model.info() (Default)

### การใช้งาน

```python
from ultralytics import YOLO

model = YOLO("model.yaml")
model.info(verbose=False, imgsz=640)
```

### Output

```
Model summary: 225 layers, 45,790,722 parameters, 45,790,722 gradients, 8.9 GFLOPs
```

### ข้อดี
- ✅ Built-in ใน Ultralytics
- ✅ ไม่ต้องติดตั้ง library เพิ่ม
- ✅ รวดเร็ว
- ✅ รองรับทุก YOLO architecture

### ข้อจำกัด
- ❌ ไม่แสดงรายละเอียดแต่ละ layer
- ❌ อาจไม่แม่นยำ 100% สำหรับ custom modules

---

## Method 2: thop Library (Optional)

### การติดตั้ง

```bash
pip install thop
```

### การใช้งาน

```python
from thop import profile, clever_format
import torch

model = YOLO("model.yaml").model
dummy_input = torch.randn(1, 3, 640, 640)

# Calculate FLOPs with per-layer details
flops, params = profile(model, inputs=(dummy_input,), verbose=True)

# Format output
flops, params = clever_format([flops, params], "%.3f")
print(f"FLOPs: {flops}, Params: {params}")
```

### Output (verbose=True)

```
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.activation.SiLU'>.
...
[INFO] Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  0.614 GMac
[INFO] Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  1.229 GMac
...
Total FLOPs: 8.900G, Total Params: 45.791M
```

### ข้อดี
- ✅ แสดงรายละเอียดแต่ละ layer
- ✅ แม่นยำสูง
- ✅ เหมาะสำหรับ paper/thesis
- ✅ รองรับ custom operations

### ข้อจำกัด
- ❌ ต้องติดตั้ง library เพิ่ม
- ❌ ช้ากว่า (ถ้าเปิด verbose)
- ❌ อาจไม่รองรับ operations บางตัว

---

## ระบบใช้วิธีไหน?

### Default Behavior

1. **ลอง Ultralytics model.info()** ก่อน
2. **ถ้าไม่ได้ → ลอง thop** (ถ้าติดตั้งไว้)
3. **ถ้าทั้ง 2 วิธีไม่ได้ → flops = null**

### Code Flow

```python
# Try method 1: Ultralytics
flops = extract_from_model_info()

# Fallback to method 2: thop (if installed)
if flops is None:
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy,))
        flops = flops / 1e9  # Convert to GFLOPs
    except ImportError:
        pass  # thop not installed
```

---

## เปรียบเทียบผลลัพธ์

### YOLOv8n (640×640)

| Method | FLOPs | Time | Detail Level |
|--------|-------|------|--------------|
| model.info() | 8.7 G | ~0.1s | Summary only |
| thop | 8.9 G | ~0.5s | Per-layer |

**Note:** ค่าอาจต่างกันเล็กน้อยเนื่องจากวิธีคำนวณต่างกัน

---

## Use Cases

### ใช้ model.info() เมื่อ:
- ✅ ต้องการความเร็ว
- ✅ ใช้ใน production
- ✅ ไม่ต้องการรายละเอียด
- ✅ ใช้ standard YOLO architecture

### ใช้ thop เมื่อ:
- ✅ เขียน paper/thesis
- ✅ ต้องการวิเคราะห์แต่ละ layer
- ✅ เปรียบเทียบ architectures
- ✅ มี custom operations
- ✅ ต้องการความแม่นยำสูง

---

## Manual Calculation (Advanced)

สำหรับการคำนวณด้วยตนเอง:

```python
def calculate_flops_manual(model, input_size=(1, 3, 640, 640)):
    """Calculate FLOPs manually for research purposes."""
    total_flops = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # FLOPs = 2 × Cin × Cout × K × K × H × W
            cin = module.in_channels
            cout = module.out_channels
            k = module.kernel_size[0]
            # Calculate output H, W based on stride, padding
            h_out = (input_size[2] - k + 2*module.padding[0]) // module.stride[0] + 1
            w_out = (input_size[3] - k + 2*module.padding[1]) // module.stride[1] + 1
            
            flops = 2 * cin * cout * k * k * h_out * w_out
            total_flops += flops
            
            print(f"{name}: {flops/1e9:.3f} GFLOPs")
    
    return total_flops / 1e9  # Convert to GFLOPs
```

---

## Recommendations

### Development
```python
# Quick validation
model.info(verbose=False, imgsz=640)
```

### Research/Paper
```python
# Detailed analysis
from thop import profile, clever_format
flops, params = profile(model, inputs=(dummy,), verbose=True)
flops, params = clever_format([flops, params], "%.3f")
```

### Production
```python
# Use cached values from record.json
record = load_model_metadata(model_id)
flops = record.get("flops")  # Pre-calculated
```

---

## Installation

### Optional: Install thop

```bash
pip install thop
```

หรือเพิ่มใน `requirements.txt`:
```
thop>=0.1.1  # Optional: for detailed FLOPs calculation
```

---

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [thop GitHub Repository](https://github.com/Lyken17/pytorch-OpCounter)
- [FLOPs Calculation Methods Paper](https://arxiv.org/abs/...)
