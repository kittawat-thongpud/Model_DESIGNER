"""
Module registry — catalog of all available modules for the Model Designer.

Three sources:
  1. Built-in Ultralytics modules (Conv, C2f, SPPF, Detect, etc.)
  2. PyTorch nn.* modules (nn.Upsample, nn.Identity, nn.MaxPool2d, etc.)
  3. Custom user-defined modules (from module_storage)
"""
from __future__ import annotations
from typing import Any


# ── Built-in Ultralytics modules ─────────────────────────────────────────────
# Organized by category with their default args
# Args format matches YAML: [out_ch, kernel, stride, pad, groups] etc.

_BUILTIN_MODULES: list[dict[str, Any]] = [
    # ── Basic Operations ─────────────────────────────────────────────────────
    {
        "name": "Conv",
        "category": "basic",
        "description": "Conv2d + BatchNorm2d + SiLU activation",
        "args": [
            {"name": "out_channels", "type": "int", "default": 64},
            {"name": "kernel_size", "type": "int", "default": 3},
            {"name": "stride", "type": "int", "default": 1},
            {"name": "padding", "type": "int", "default": None},
            {"name": "groups", "type": "int", "default": 1},
        ],
        "source": "ultralytics",
    },
    {
        "name": "DWConv",
        "category": "basic",
        "description": "Depthwise convolution (groups=in_channels)",
        "args": [
            {"name": "out_channels", "type": "int", "default": 64},
            {"name": "kernel_size", "type": "int", "default": 3},
            {"name": "stride", "type": "int", "default": 1},
        ],
        "source": "ultralytics",
    },
    {
        "name": "RepConv",
        "category": "basic",
        "description": "Reparameterizable convolution (train: multi-branch, deploy: single conv)",
        "args": [
            {"name": "out_channels", "type": "int", "default": 64},
            {"name": "kernel_size", "type": "int", "default": 3},
            {"name": "stride", "type": "int", "default": 1},
        ],
        "source": "ultralytics",
    },
    {
        "name": "GhostConv",
        "category": "basic",
        "description": "Ghost convolution (cheap linear operations for more feature maps)",
        "args": [
            {"name": "out_channels", "type": "int", "default": 64},
            {"name": "kernel_size", "type": "int", "default": 1},
            {"name": "stride", "type": "int", "default": 1},
        ],
        "source": "ultralytics",
    },
    {
        "name": "Focus",
        "category": "basic",
        "description": "Focus layer — space-to-depth + Conv",
        "args": [
            {"name": "out_channels", "type": "int", "default": 64},
            {"name": "kernel_size", "type": "int", "default": 1},
        ],
        "source": "ultralytics",
    },
    {
        "name": "Concat",
        "category": "basic",
        "description": "Concatenate tensors along a dimension",
        "args": [
            {"name": "dimension", "type": "int", "default": 1},
        ],
        "source": "ultralytics",
    },

    # ── Composite Blocks ─────────────────────────────────────────────────────
    {
        "name": "C2f",
        "category": "composite",
        "description": "CSP bottleneck with 2 convolutions (YOLOv8)",
        "args": [
            {"name": "out_channels", "type": "int", "default": 128},
            {"name": "shortcut", "type": "bool", "default": True},
        ],
        "source": "ultralytics",
    },
    {
        "name": "C3k2",
        "category": "composite",
        "description": "C2f variant with kernel-size-2 bottlenecks (YOLO11)",
        "args": [
            {"name": "out_channels", "type": "int", "default": 128},
            {"name": "shortcut", "type": "bool", "default": True},
        ],
        "source": "ultralytics",
    },
    {
        "name": "C2PSA",
        "category": "composite",
        "description": "C2f with Partial Self-Attention (YOLO11)",
        "args": [
            {"name": "out_channels", "type": "int", "default": 128},
        ],
        "source": "ultralytics",
    },
    {
        "name": "C3",
        "category": "composite",
        "description": "CSP bottleneck with 3 convolutions (YOLOv5)",
        "args": [
            {"name": "out_channels", "type": "int", "default": 128},
            {"name": "shortcut", "type": "bool", "default": True},
        ],
        "source": "ultralytics",
    },
    {
        "name": "Bottleneck",
        "category": "composite",
        "description": "Standard bottleneck block",
        "args": [
            {"name": "out_channels", "type": "int", "default": 128},
            {"name": "shortcut", "type": "bool", "default": True},
        ],
        "source": "ultralytics",
    },
    {
        "name": "SPPF",
        "category": "composite",
        "description": "Spatial Pyramid Pooling — Fast",
        "args": [
            {"name": "out_channels", "type": "int", "default": 256},
            {"name": "kernel_size", "type": "int", "default": 5},
        ],
        "source": "ultralytics",
    },
    {
        "name": "SPP",
        "category": "composite",
        "description": "Spatial Pyramid Pooling",
        "args": [
            {"name": "out_channels", "type": "int", "default": 256},
            {"name": "kernels", "type": "list", "default": [5, 9, 13]},
        ],
        "source": "ultralytics",
    },

    # ── Attention ─────────────────────────────────────────────────────────────
    {
        "name": "CBAM",
        "category": "attention",
        "description": "Convolutional Block Attention Module",
        "args": [
            {"name": "out_channels", "type": "int", "default": 128},
            {"name": "kernel_size", "type": "int", "default": 7},
        ],
        "source": "ultralytics",
    },
    {
        "name": "PSA",
        "category": "attention",
        "description": "Partial Self-Attention module",
        "args": [
            {"name": "out_channels", "type": "int", "default": 128},
        ],
        "source": "ultralytics",
    },

    # ── Detector Modules (singleton — always placed last in head) ─────────────
    {
        "name": "Detect",
        "category": "detector",
        "description": "YOLO detection head (singleton — auto-placed last in head)",
        "args": [
            {"name": "nc", "type": "int", "default": 80},
        ],
        "source": "ultralytics",
    },
    {
        "name": "Segment",
        "category": "detector",
        "description": "YOLO instance segmentation head (singleton)",
        "args": [
            {"name": "nc", "type": "int", "default": 80},
            {"name": "nm", "type": "int", "default": 32},
            {"name": "npr", "type": "int", "default": 256},
        ],
        "source": "ultralytics",
    },
    {
        "name": "Pose",
        "category": "detector",
        "description": "YOLO pose estimation head (singleton)",
        "args": [
            {"name": "nc", "type": "int", "default": 1},
            {"name": "kpt_shape", "type": "list", "default": [17, 3]},
        ],
        "source": "ultralytics",
    },
    {
        "name": "Classify",
        "category": "detector",
        "description": "YOLO classification head (singleton)",
        "args": [
            {"name": "nc", "type": "int", "default": 1000},
        ],
        "source": "ultralytics",
    },
    {
        "name": "OBB",
        "category": "detector",
        "description": "Oriented Bounding Box detection head (singleton)",
        "args": [
            {"name": "nc", "type": "int", "default": 80},
        ],
        "source": "ultralytics",
    },

    # ── Specialized ──────────────────────────────────────────────────────────
    {
        "name": "TorchVision",
        "category": "specialized",
        "description": "Load any torchvision model as backbone",
        "args": [
            {"name": "out_channels", "type": "int", "default": 768},
            {"name": "model_name", "type": "str", "default": "convnext_tiny"},
            {"name": "weights", "type": "str", "default": "DEFAULT"},
            {"name": "unwrap", "type": "bool", "default": True},
            {"name": "truncate", "type": "int", "default": 2},
            {"name": "split", "type": "bool", "default": False},
        ],
        "source": "ultralytics",
    },
    {
        "name": "Index",
        "category": "specialized",
        "description": "Extract specific tensor from multi-output model",
        "args": [
            {"name": "out_channels", "type": "int", "default": 192},
            {"name": "index", "type": "int", "default": 0},
        ],
        "source": "ultralytics",
    },
]

# ── PyTorch nn.* modules available in YAML ───────────────────────────────────

_TORCH_NN_MODULES: list[dict[str, Any]] = [
    # ── Convolution Layers ───────────────────────────────────────────────────
    {
        "name": "nn.Conv1d",
        "category": "convolution",
        "description": "1D convolution over input signal",
        "args": [
            {"name": "out_channels", "type": "int", "default": 64},
            {"name": "kernel_size", "type": "int", "default": 3},
            {"name": "stride", "type": "int", "default": 1},
            {"name": "padding", "type": "int", "default": 0},
            {"name": "dilation", "type": "int", "default": 1},
            {"name": "groups", "type": "int", "default": 1},
            {"name": "bias", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Conv2d",
        "category": "convolution",
        "description": "2D convolution over input (images)",
        "args": [
            {"name": "out_channels", "type": "int", "default": 64},
            {"name": "kernel_size", "type": "int", "default": 3},
            {"name": "stride", "type": "int", "default": 1},
            {"name": "padding", "type": "int", "default": 0},
            {"name": "dilation", "type": "int", "default": 1},
            {"name": "groups", "type": "int", "default": 1},
            {"name": "bias", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Conv3d",
        "category": "convolution",
        "description": "3D convolution over input (videos/volumetric data)",
        "args": [
            {"name": "out_channels", "type": "int", "default": 64},
            {"name": "kernel_size", "type": "int", "default": 3},
            {"name": "stride", "type": "int", "default": 1},
            {"name": "padding", "type": "int", "default": 0},
            {"name": "dilation", "type": "int", "default": 1},
            {"name": "groups", "type": "int", "default": 1},
            {"name": "bias", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.ConvTranspose2d",
        "category": "convolution",
        "description": "2D transposed convolution (deconvolution) for upsampling",
        "args": [
            {"name": "out_channels", "type": "int", "default": 64},
            {"name": "kernel_size", "type": "int", "default": 3},
            {"name": "stride", "type": "int", "default": 2},
            {"name": "padding", "type": "int", "default": 1},
            {"name": "output_padding", "type": "int", "default": 1},
            {"name": "groups", "type": "int", "default": 1},
            {"name": "bias", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.ConvTranspose1d",
        "category": "convolution",
        "description": "1D transposed convolution for upsampling",
        "args": [
            {"name": "out_channels", "type": "int", "default": 64},
            {"name": "kernel_size", "type": "int", "default": 3},
            {"name": "stride", "type": "int", "default": 2},
            {"name": "padding", "type": "int", "default": 1},
            {"name": "output_padding", "type": "int", "default": 1},
            {"name": "groups", "type": "int", "default": 1},
            {"name": "bias", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },

    # ── Pooling Layers ───────────────────────────────────────────────────────
    {
        "name": "nn.MaxPool1d",
        "category": "pooling",
        "description": "1D max pooling",
        "args": [
            {"name": "kernel_size", "type": "int", "default": 2},
            {"name": "stride", "type": "int", "default": None},
            {"name": "padding", "type": "int", "default": 0},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.MaxPool2d",
        "category": "pooling",
        "description": "2D max pooling",
        "args": [
            {"name": "kernel_size", "type": "int", "default": 2},
            {"name": "stride", "type": "int", "default": None},
            {"name": "padding", "type": "int", "default": 0},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.MaxPool3d",
        "category": "pooling",
        "description": "3D max pooling",
        "args": [
            {"name": "kernel_size", "type": "int", "default": 2},
            {"name": "stride", "type": "int", "default": None},
            {"name": "padding", "type": "int", "default": 0},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.AvgPool1d",
        "category": "pooling",
        "description": "1D average pooling",
        "args": [
            {"name": "kernel_size", "type": "int", "default": 2},
            {"name": "stride", "type": "int", "default": None},
            {"name": "padding", "type": "int", "default": 0},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.AvgPool2d",
        "category": "pooling",
        "description": "2D average pooling",
        "args": [
            {"name": "kernel_size", "type": "int", "default": 2},
            {"name": "stride", "type": "int", "default": None},
            {"name": "padding", "type": "int", "default": 0},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.AvgPool3d",
        "category": "pooling",
        "description": "3D average pooling",
        "args": [
            {"name": "kernel_size", "type": "int", "default": 2},
            {"name": "stride", "type": "int", "default": None},
            {"name": "padding", "type": "int", "default": 0},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.AdaptiveAvgPool1d",
        "category": "pooling",
        "description": "1D adaptive average pooling",
        "args": [
            {"name": "output_size", "type": "int", "default": 1},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.AdaptiveAvgPool2d",
        "category": "pooling",
        "description": "2D adaptive average pooling",
        "args": [
            {"name": "output_size", "type": "int", "default": 1},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.AdaptiveAvgPool3d",
        "category": "pooling",
        "description": "3D adaptive average pooling",
        "args": [
            {"name": "output_size", "type": "int", "default": 1},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.AdaptiveMaxPool1d",
        "category": "pooling",
        "description": "1D adaptive max pooling",
        "args": [
            {"name": "output_size", "type": "int", "default": 1},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.AdaptiveMaxPool2d",
        "category": "pooling",
        "description": "2D adaptive max pooling",
        "args": [
            {"name": "output_size", "type": "int", "default": 1},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.AdaptiveMaxPool3d",
        "category": "pooling",
        "description": "3D adaptive max pooling",
        "args": [
            {"name": "output_size", "type": "int", "default": 1},
        ],
        "source": "torch.nn",
    },

    # ── Activation Functions ─────────────────────────────────────────────────
    {
        "name": "nn.ReLU",
        "category": "activation",
        "description": "Rectified Linear Unit: max(0, x)",
        "args": [
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.ReLU6",
        "category": "activation",
        "description": "ReLU capped at 6: min(max(0, x), 6)",
        "args": [
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.LeakyReLU",
        "category": "activation",
        "description": "Leaky ReLU with negative slope",
        "args": [
            {"name": "negative_slope", "type": "float", "default": 0.01},
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.PReLU",
        "category": "activation",
        "description": "Parametric ReLU (learnable negative slope)",
        "args": [
            {"name": "num_parameters", "type": "int", "default": 1},
            {"name": "init", "type": "float", "default": 0.25},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.ELU",
        "category": "activation",
        "description": "Exponential Linear Unit",
        "args": [
            {"name": "alpha", "type": "float", "default": 1.0},
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.GELU",
        "category": "activation",
        "description": "Gaussian Error Linear Unit (used in transformers)",
        "args": [
            {"name": "approximate", "type": "str", "default": "none"},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.SiLU",
        "category": "activation",
        "description": "Sigmoid Linear Unit (Swish): x * sigmoid(x)",
        "args": [
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Mish",
        "category": "activation",
        "description": "Mish activation: x * tanh(softplus(x))",
        "args": [
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Sigmoid",
        "category": "activation",
        "description": "Sigmoid: 1 / (1 + exp(-x))",
        "args": [],
        "source": "torch.nn",
    },
    {
        "name": "nn.Tanh",
        "category": "activation",
        "description": "Hyperbolic tangent",
        "args": [],
        "source": "torch.nn",
    },
    {
        "name": "nn.Softmax",
        "category": "activation",
        "description": "Softmax activation",
        "args": [
            {"name": "dim", "type": "int", "default": -1},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.LogSoftmax",
        "category": "activation",
        "description": "Log(Softmax(x))",
        "args": [
            {"name": "dim", "type": "int", "default": -1},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Softmax2d",
        "category": "activation",
        "description": "Softmax over spatial dimensions",
        "args": [],
        "source": "torch.nn",
    },
    {
        "name": "nn.Hardswish",
        "category": "activation",
        "description": "Hard Swish (efficient approximation of Swish)",
        "args": [
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Hardsigmoid",
        "category": "activation",
        "description": "Hard Sigmoid (efficient approximation)",
        "args": [
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },

    # ── Normalization Layers ─────────────────────────────────────────────────
    {
        "name": "nn.BatchNorm1d",
        "category": "normalization",
        "description": "Batch normalization for 1D/2D inputs",
        "args": [
            {"name": "num_features", "type": "int", "default": 64},
            {"name": "eps", "type": "float", "default": 1e-5},
            {"name": "momentum", "type": "float", "default": 0.1},
            {"name": "affine", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.BatchNorm2d",
        "category": "normalization",
        "description": "Batch normalization for 4D inputs (N, C, H, W)",
        "args": [
            {"name": "num_features", "type": "int", "default": 64},
            {"name": "eps", "type": "float", "default": 1e-5},
            {"name": "momentum", "type": "float", "default": 0.1},
            {"name": "affine", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.BatchNorm3d",
        "category": "normalization",
        "description": "Batch normalization for 5D inputs (N, C, D, H, W)",
        "args": [
            {"name": "num_features", "type": "int", "default": 64},
            {"name": "eps", "type": "float", "default": 1e-5},
            {"name": "momentum", "type": "float", "default": 0.1},
            {"name": "affine", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.LayerNorm",
        "category": "normalization",
        "description": "Layer normalization (used in transformers)",
        "args": [
            {"name": "normalized_shape", "type": "int", "default": 512},
            {"name": "eps", "type": "float", "default": 1e-5},
            {"name": "elementwise_affine", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.GroupNorm",
        "category": "normalization",
        "description": "Group normalization",
        "args": [
            {"name": "num_groups", "type": "int", "default": 32},
            {"name": "num_channels", "type": "int", "default": 64},
            {"name": "eps", "type": "float", "default": 1e-5},
            {"name": "affine", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.InstanceNorm1d",
        "category": "normalization",
        "description": "Instance normalization for 1D inputs",
        "args": [
            {"name": "num_features", "type": "int", "default": 64},
            {"name": "eps", "type": "float", "default": 1e-5},
            {"name": "momentum", "type": "float", "default": 0.1},
            {"name": "affine", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.InstanceNorm2d",
        "category": "normalization",
        "description": "Instance normalization for 2D inputs (style transfer)",
        "args": [
            {"name": "num_features", "type": "int", "default": 64},
            {"name": "eps", "type": "float", "default": 1e-5},
            {"name": "momentum", "type": "float", "default": 0.1},
            {"name": "affine", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.InstanceNorm3d",
        "category": "normalization",
        "description": "Instance normalization for 3D inputs",
        "args": [
            {"name": "num_features", "type": "int", "default": 64},
            {"name": "eps", "type": "float", "default": 1e-5},
            {"name": "momentum", "type": "float", "default": 0.1},
            {"name": "affine", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.LocalResponseNorm",
        "category": "normalization",
        "description": "Local Response Normalization (AlexNet)",
        "args": [
            {"name": "size", "type": "int", "default": 5},
            {"name": "alpha", "type": "float", "default": 0.0001},
            {"name": "beta", "type": "float", "default": 0.75},
            {"name": "k", "type": "float", "default": 1.0},
        ],
        "source": "torch.nn",
    },

    # ── Linear Layers ────────────────────────────────────────────────────────
    {
        "name": "nn.Linear",
        "category": "linear",
        "description": "Fully connected layer (dense layer)",
        "args": [
            {"name": "out_features", "type": "int", "default": 128},
            {"name": "bias", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Bilinear",
        "category": "linear",
        "description": "Bilinear transformation",
        "args": [
            {"name": "in1_features", "type": "int", "default": 128},
            {"name": "in2_features", "type": "int", "default": 128},
            {"name": "out_features", "type": "int", "default": 64},
            {"name": "bias", "type": "bool", "default": True},
        ],
        "source": "torch.nn",
    },

    # ── Dropout Layers ───────────────────────────────────────────────────────
    {
        "name": "nn.Dropout",
        "category": "dropout",
        "description": "Randomly zero elements with probability p",
        "args": [
            {"name": "p", "type": "float", "default": 0.5},
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Dropout1d",
        "category": "dropout",
        "description": "Randomly zero entire channels (1D)",
        "args": [
            {"name": "p", "type": "float", "default": 0.5},
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Dropout2d",
        "category": "dropout",
        "description": "Randomly zero entire channels (2D)",
        "args": [
            {"name": "p", "type": "float", "default": 0.5},
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Dropout3d",
        "category": "dropout",
        "description": "Randomly zero entire channels (3D)",
        "args": [
            {"name": "p", "type": "float", "default": 0.5},
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.AlphaDropout",
        "category": "dropout",
        "description": "Alpha Dropout (for SELU activation)",
        "args": [
            {"name": "p", "type": "float", "default": 0.5},
            {"name": "inplace", "type": "bool", "default": False},
        ],
        "source": "torch.nn",
    },

    # ── Utility Layers ───────────────────────────────────────────────────────
    {
        "name": "nn.Flatten",
        "category": "utility",
        "description": "Flatten tensor to 1D (keeping batch dimension)",
        "args": [
            {"name": "start_dim", "type": "int", "default": 1},
            {"name": "end_dim", "type": "int", "default": -1},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Unflatten",
        "category": "utility",
        "description": "Unflatten a tensor dimension",
        "args": [
            {"name": "dim", "type": "int", "default": 1},
            {"name": "unflattened_size", "type": "list", "default": [64, 7, 7]},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.Identity",
        "category": "utility",
        "description": "Pass-through (no-op)",
        "args": [],
        "source": "torch.nn",
    },
    {
        "name": "nn.Upsample",
        "category": "utility",
        "description": "Spatial upsampling",
        "args": [
            {"name": "size", "type": "int", "default": None},
            {"name": "scale_factor", "type": "float", "default": 2.0},
            {"name": "mode", "type": "str", "default": "nearest"},
            {"name": "align_corners", "type": "bool", "default": None},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.UpsamplingNearest2d",
        "category": "utility",
        "description": "2D nearest-neighbor upsampling",
        "args": [
            {"name": "size", "type": "int", "default": None},
            {"name": "scale_factor", "type": "float", "default": 2.0},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.UpsamplingBilinear2d",
        "category": "utility",
        "description": "2D bilinear upsampling",
        "args": [
            {"name": "size", "type": "int", "default": None},
            {"name": "scale_factor", "type": "float", "default": 2.0},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.PixelShuffle",
        "category": "utility",
        "description": "Rearrange (*, C*r^2, H, W) to (*, C, H*r, W*r)",
        "args": [
            {"name": "upscale_factor", "type": "int", "default": 2},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.PixelUnshuffle",
        "category": "utility",
        "description": "Reverse of PixelShuffle",
        "args": [
            {"name": "downscale_factor", "type": "int", "default": 2},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.ZeroPad2d",
        "category": "utility",
        "description": "Zero padding for 2D inputs",
        "args": [
            {"name": "padding", "type": "int", "default": 1},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.ReflectionPad2d",
        "category": "utility",
        "description": "Reflection padding for 2D inputs",
        "args": [
            {"name": "padding", "type": "int", "default": 1},
        ],
        "source": "torch.nn",
    },
    {
        "name": "nn.ReplicationPad2d",
        "category": "utility",
        "description": "Replication padding for 2D inputs",
        "args": [
            {"name": "padding", "type": "int", "default": 1},
        ],
        "source": "torch.nn",
    },
]

# ── HSG-DET custom modules (registered via hsg_det package) ─────────────────
# These blocks must be discovered by Ultralytics before YAML parsing.
# When the designer generates a YAML that includes these nodes, the training
# worker must call HSGDetPlugin.register_modules() first (handled automatically
# by the model_arch='hsg_det' hook in ultra_trainer.py).

_HSG_DET_MODULES: list[dict[str, Any]] = [
    {
        "name": "SparseGlobalBlock",
        "category": "attention",
        "description": (
            "Sparse Global Self-Attention (HSG-DET). "
            "Selects top-k salient tokens, runs O(k²·d) sparse attention, "
            "then scatters enriched values back via residual add. "
            "Complexity ~15× cheaper than full self-attention at P5 1080p."
        ),
        "args": [
            {"name": "out_channels", "type": "int", "default": 256,
             "help": "Number of channels (must equal in_channels — block is channel-preserving)"},
            {"name": "k", "type": "int", "default": 512,
             "help": "Number of sparse tokens to attend over (k≤H×W, clamped if larger)"},
        ],
        "source": "hsg_det",
    },
    {
        "name": "SparseGlobalBlockGated",
        "category": "attention",
        "description": (
            "Gated Sparse Global Self-Attention (HSG-DET). "
            "Same as SparseGlobalBlock but with a learnable gate α initialised to 0. "
            "Block starts as identity and gradually enables global reasoning during training. "
            "Recommended over plain SparseGlobalBlock when warm-starting from YOLO weights."
        ),
        "args": [
            {"name": "out_channels", "type": "int", "default": 256,
             "help": "Number of channels (channel-preserving)"},
            {"name": "k", "type": "int", "default": 512,
             "help": "Number of sparse tokens"},
        ],
        "source": "hsg_det",
    },
]


def get_all_modules() -> list[dict[str, Any]]:
    """Return all available modules: built-in + torch.nn + hsg_det + custom."""
    from . import module_storage

    modules = []

    # Built-in Ultralytics
    for m in _BUILTIN_MODULES:
        modules.append({**m})

    # PyTorch nn.*
    for m in _TORCH_NN_MODULES:
        modules.append({**m})

    # HSG-DET custom blocks
    for m in _HSG_DET_MODULES:
        modules.append({**m})

    # Custom user modules
    for rec in module_storage.list_modules():
        modules.append({
            "name": rec["name"],
            "category": "custom",
            "description": rec.get("description", ""),
            "args": rec.get("args", []),
            "source": "custom",
        })

    return modules


def get_module_info(name: str) -> dict[str, Any] | None:
    """Look up a specific module by name."""
    for m in get_all_modules():
        if m["name"] == name:
            return m
    return None


def get_categories() -> list[str]:
    """Return all unique module categories."""
    cats = set()
    for m in get_all_modules():
        cats.add(m["category"])
    return sorted(cats)


def register_custom_modules() -> None:
    """Register all custom modules into Ultralytics.
    
    This must be called before loading a YOLO model that uses custom modules
    (like SparseGlobalBlock).
    """
    # 1. HSG-DET modules
    try:
        import hsg_det  # noqa: F401
    except ImportError as e:
        print(f"Warning: Could not import hsg_det: {e}")

    # 2. Future: Dynamic user modules from module_storage
    # (Currently they are just config-based, but if we add code generation
    # we would import them here)
