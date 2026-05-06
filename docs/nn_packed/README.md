# Neural Network Architecture Reference
> ไฟล์สรุปที่อยู่ของ NN modules + YAML config ทั้งหมดใน workspace เพื่อใช้ศึกษาโครงสร้าง

---

## 1. YOLOv8 (Ultralytics installed)

### NN Modules
| File | Description |
|------|-------------|
| `venv/lib/python3.12/site-packages/ultralytics/nn/tasks.py` | **Model builder** — `DetectionModel`, `parse_model()`, `attempt_load_*` |
| `venv/lib/python3.12/site-packages/ultralytics/nn/modules/__init__.py` | Module registry / exports |
| `venv/lib/python3.12/site-packages/ultralytics/nn/modules/block.py` | **C2f, C3, C3k2, SPPF, Bottleneck, CSP, RepNCSPELAN, ELAN** |
| `venv/lib/python3.12/site-packages/ultralytics/nn/modules/conv.py` | **Conv, DWConv, GhostConv, RepConv, LightConv, ChannelAttention** |
| `venv/lib/python3.12/site-packages/ultralytics/nn/modules/head.py` | **Detect, Segment, Pose, OBB, WorldDetect, RTDETRDecoder** |
| `venv/lib/python3.12/site-packages/ultralytics/nn/modules/transformer.py` | **AIFI, TransformerEncoderLayer, MLP, LayerNorm2d** |
| `venv/lib/python3.12/site-packages/ultralytics/nn/modules/activation.py` | **AGLU** |

### YAML Config
| File | Notes |
|------|-------|
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/v8/yolov8.yaml` | **Detection** base |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/v8/yolov8-seg.yaml` | Segmentation |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/v8/yolov8-pose.yaml` | Pose |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/v8/yolov8-obb.yaml` | Oriented BBox |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/v8/yolov8-p6.yaml` | P6 (extra scale) |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/v8/yolov8-ghost.yaml` | GhostNet variant |

---

## 2. YOLOv11 (Ultralytics installed)

### NN Modules
> Same files as YOLOv8 — uses same `nn/modules/` but different YAML architecture

### YAML Config
| File | Notes |
|------|-------|
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/11/yolo11.yaml` | **Detection** base |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/11/yolo11-seg.yaml` | Segmentation |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/11/yolo11-pose.yaml` | Pose |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/11/yolo11-obb.yaml` | Oriented BBox |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/11/yolo11-cls.yaml` | Classification |

---

## 3. YOLOv26 (Ultralytics installed)

### NN Modules
> Same `nn/modules/` base — new blocks registered for v26 (e.g. attention-based)

### YAML Config
| File | Notes |
|------|-------|
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26.yaml` | **Detection** base |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26-seg.yaml` | Segmentation |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26-pose.yaml` | Pose |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26-obb.yaml` | Oriented BBox |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26-cls.yaml` | Classification |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26-p2.yaml` | P2 high-res |
| `venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26-p6.yaml` | P6 large scale |

---

## 4. Mamba-YOLO (vendored)

### NN Modules
| File | Description |
|------|-------------|
| `backend/data/vendor/Mamba-YOLO/ultralytics/nn/tasks.py` | Model builder (patched for Mamba modules) |
| `backend/data/vendor/Mamba-YOLO/ultralytics/nn/modules/__init__.py` | Module registry (includes Mamba) |
| `backend/data/vendor/Mamba-YOLO/ultralytics/nn/modules/mamba_yolo.py` | **SS2D, VSSBlock, SimpleStem, VisionClueMerge** — core SSM blocks |
| `backend/data/vendor/Mamba-YOLO/ultralytics/nn/modules/block.py` | Standard + Mamba-extended blocks |
| `backend/data/vendor/Mamba-YOLO/ultralytics/nn/modules/conv.py` | Conv modules |
| `backend/data/vendor/Mamba-YOLO/ultralytics/nn/modules/head.py` | Detection heads |
| `backend/data/vendor/Mamba-YOLO/ultralytics/nn/modules/transformer.py` | Transformer layers |
| `backend/data/vendor/Mamba-YOLO/ultralytics/nn/modules/common_utils_mbyolo.py` | Mamba utilities |

### YAML Config
| File | Notes |
|------|-------|
| `backend/data/vendor/Mamba-YOLO/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml` | **Tiny** |
| `backend/data/vendor/Mamba-YOLO/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml` | **Base** |
| `backend/data/vendor/Mamba-YOLO/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L.yaml` | **Large** |

---

## 5. RT-DETR v1 (vendored)

### NN Modules
| File | Description |
|------|-------------|
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/src/nn/__init__.py` | Module entry |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/src/nn/backbone/presnet.py` | **PResNet** backbone |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/src/nn/backbone/dla.py` | **DLA** backbone |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/src/nn/backbone/regnet.py` | **RegNet** backbone |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/src/nn/backbone/common.py` | Common backbone utilities |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/src/nn/arch/classification.py` | Arch definition |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/src/nn/criterion/__init__.py` | Loss functions |

### YAML Config (`.yml`)
| File | Notes |
|------|-------|
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml` | **ResNet50-VD** |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_coco.yml` | ResNet101-VD |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml` | ResNet18-VD |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r34vd_6x_coco.yml` | ResNet34-VD |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_dla34_6x_coco.yml` | DLA-34 |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_regnet_6x_coco.yml` | RegNet |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/configs/rtdetr/include/rtdetr_r50vd.yml` | Model definition (included) |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/configs/rtdetr/include/optimizer.yml` | Optimizer config |
| `backend/data/vendor/RT-DETR/rtdetr_pytorch/configs/rtdetr/include/dataloader.yml` | Dataloader config |

---

## 6. RT-DETRv2 (vendored)

### NN Modules
| File | Description |
|------|-------------|
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/__init__.py` | Module entry |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/backbone/hgnetv2.py` | **HGNetV2** (B/L/X/H configs) |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/backbone/csp_darknet.py` | **CSP-Darknet** |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/backbone/csp_resnet.py` | **CSP-ResNet** |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/backbone/presnet.py` | **PResNet** |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/backbone/timm_model.py` | Timm model wrapper |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/backbone/torchvision_model.py` | TorchVision wrapper |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/arch/yolo.py` | YOLO-style architecture |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/criterion/det_criterion.py` | Detection loss |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/postprocessor/detr_postprocessor.py` | DETR postprocessor |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/src/nn/postprocessor/nms_postprocessor.py` | NMS postprocessor |

### YAML Config (`.yml`)
| File | Notes |
|------|-------|
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml` | **ResNet50-VD** |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml` | ResNet101-VD |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml` | ResNet18-VD (120 epochs) |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml` | ResNet34-VD |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_hgnetv2_l_6x_coco.yml` | **HGNetV2-L** |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_hgnetv2_h_6x_coco.yml` | **HGNetV2-H** |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_hgnetv2_x_6x_coco.yml` | **HGNetV2-X** |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml` | ResNet50-VD medium |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml` | DSP (dense sampling) |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/include/rtdetrv2_r50vd.yml` | Model definition |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/include/optimizer.yml` | Optimizer |
| `backend/data/vendor/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/include/dataloader.yml` | Dataloader |

---

## 7. DINO-DETR (vendored)

### NN Modules
| File | Description |
|------|-------------|
| `backend/data/vendor/DINO-DETR/models/dino/dino.py` | **DINO main model** — build_dino() |
| `backend/data/vendor/DINO-DETR/models/dino/deformable_transformer.py` | **DeformableTransformer** (encoder/decoder) |
| `backend/data/vendor/DINO-DETR/models/dino/transformer_deformable.py` | Deformable attention ops |
| `backend/data/vendor/DINO-DETR/models/dino/backbone.py` | **Backbone** (ResNet/Swin/ConvNeXt) |
| `backend/data/vendor/DINO-DETR/models/dino/swin_transformer.py` | **Swin Transformer** |
| `backend/data/vendor/DINO-DETR/models/dino/convnext.py` | **ConvNeXt** |
| `backend/data/vendor/DINO-DETR/models/dino/attention.py` | Multi-scale deformable attention |
| `backend/data/vendor/DINO-DETR/models/dino/dn_components.py` | **DN (Denoising)** components |
| `backend/data/vendor/DINO-DETR/models/dino/matcher.py` | Hungarian matcher |
| `backend/data/vendor/DINO-DETR/models/dino/position_encoding.py` | Positional encoding |
| `backend/data/vendor/DINO-DETR/models/dino/segmentation.py` | Segmentation head |
| `backend/data/vendor/DINO-DETR/models/dino/utils.py` | Module utilities |
| `backend/data/vendor/DINO-DETR/models/registry.py` | Model registry |

### Config (Python config files)
| File | Notes |
|------|-------|
| `backend/data/vendor/DINO-DETR/config/DINO/DINO_4scale.py` | **4-scale** (ResNet backbone) |
| `backend/data/vendor/DINO-DETR/config/DINO/DINO_5scale.py` | **5-scale** |
| `backend/data/vendor/DINO-DETR/config/DINO/DINO_4scale_swin.py` | 4-scale + **Swin** backbone |
| `backend/data/vendor/DINO-DETR/config/DINO/DINO_4scale_convnext.py` | 4-scale + **ConvNeXt** |
| `backend/data/vendor/DINO-DETR/config/DINO/coco_transformer.py` | COCO transformer config |

---

## 8. DINO (Vision Transformer — self-supervised)

### NN Modules
| File | Description |
|------|-------------|
| `backend/data/vendor/DINO/vision_transformer.py` | **ViT (Vision Transformer)** — DINOHead, DINO loss |
| `backend/data/vendor/DINO/utils.py` | Training utilities, multi-crop wrapper |
| `backend/data/vendor/DINO/main_dino.py` | Main training script |
| `backend/data/vendor/DINO/hubconf.py` | TorchHub model definitions |

### Config
> ไม่มี YAML แยก — config อยู่ใน argparse ของ `main_dino.py`

---

## 9. HSG-DETR (project custom)

### NN Modules
| File | Description |
|------|-------------|
| `backend/hsg_detr/nn/__init__.py` | Module registry — registers into Ultralytics `parse_model` |
| `backend/hsg_detr/nn/sparse_global_token.py` | **SGTokenBlock** (sparse global attention), **SGStem**, **SGDown**, **RTDETRDecoderSGB** |

### YAML Config
| File | Notes |
|------|-------|
| `backend/hsg_detr/configs/hsg_detr_n.yaml` | **N (Nano)** |
| `backend/hsg_detr/configs/hsg_detr_s.yaml` | **S (Small)** |
| `backend/hsg_detr/configs/hsg_detr_m.yaml` | **M (Medium)** |
| `backend/hsg_detr/configs/hsg_detr_l.yaml` | **L (Large)** |

---

## Quick Print Script

ใช้ script นี้เพื่อ print classes ทั้งหมดจาก module:

```python
import importlib, inspect, sys
sys.path.insert(0, 'backend/data/vendor/Mamba-YOLO')
sys.path.insert(0, 'backend/data/vendor/RT-DETR/rtdetrv2_pytorch')

# Example: Ultralytics YOLOv8/11/26 modules
from ultralytics.nn.modules import block, conv, head, transformer
for mod in [block, conv, head, transformer]:
    print(f"\n{'='*60}\n{mod.__name__}\n{'='*60}")
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__:
            print(f"  class {name}({', '.join(b.__name__ for b in obj.__bases__)})")

# Mamba-YOLO specific
from ultralytics.nn.modules.mamba_yolo import *
import ultralytics.nn.modules.mamba_yolo as mb
for name, obj in inspect.getmembers(mb, inspect.isclass):
    if obj.__module__ == mb.__name__:
        print(f"  class {name}")
```

---

## Path Prefix Legend

| Prefix | Absolute Path |
|--------|--------------|
| `venv/...` | `/home/rase/kittawat_ws/Model_DESIGNER/venv/lib/python3.12/site-packages/ultralytics/...` |
| `backend/data/vendor/...` | `/home/rase/kittawat_ws/Model_DESIGNER/backend/data/vendor/...` |
| `backend/hsg_detr/...` | `/home/rase/kittawat_ws/Model_DESIGNER/backend/hsg_detr/...` |
