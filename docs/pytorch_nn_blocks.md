# PyTorch nn Blocks in Model Designer

## Overview
The Model Designer now includes **59 PyTorch nn blocks** across **7 categories**, providing comprehensive building blocks for custom neural network architectures.

## Module Statistics
- **Total modules in catalog**: 81
- **PyTorch nn modules**: 59
- **Ultralytics built-in modules**: 22
- **Categories**: 12 (activation, attention, basic, composite, convolution, dropout, head, linear, normalization, pooling, specialized, utility)

## Categories and Modules

### 1. Convolution Layers (5 modules)
- **nn.Conv1d** - 1D convolution over input signal
- **nn.Conv2d** - 2D convolution over input (images)
- **nn.Conv3d** - 3D convolution over input (videos/volumetric data)
- **nn.ConvTranspose1d** - 1D transposed convolution for upsampling
- **nn.ConvTranspose2d** - 2D transposed convolution (deconvolution) for upsampling

### 2. Pooling Layers (12 modules)
#### Max Pooling
- **nn.MaxPool1d** - 1D max pooling
- **nn.MaxPool2d** - 2D max pooling
- **nn.MaxPool3d** - 3D max pooling

#### Average Pooling
- **nn.AvgPool1d** - 1D average pooling
- **nn.AvgPool2d** - 2D average pooling
- **nn.AvgPool3d** - 3D average pooling

#### Adaptive Pooling
- **nn.AdaptiveAvgPool1d** - 1D adaptive average pooling
- **nn.AdaptiveAvgPool2d** - 2D adaptive average pooling
- **nn.AdaptiveAvgPool3d** - 3D adaptive average pooling
- **nn.AdaptiveMaxPool1d** - 1D adaptive max pooling
- **nn.AdaptiveMaxPool2d** - 2D adaptive max pooling
- **nn.AdaptiveMaxPool3d** - 3D adaptive max pooling

### 3. Activation Functions (15 modules)
#### ReLU Variants
- **nn.ReLU** - Rectified Linear Unit: max(0, x)
- **nn.ReLU6** - ReLU capped at 6: min(max(0, x), 6)
- **nn.LeakyReLU** - Leaky ReLU with negative slope
- **nn.PReLU** - Parametric ReLU (learnable negative slope)
- **nn.ELU** - Exponential Linear Unit

#### Modern Activations
- **nn.GELU** - Gaussian Error Linear Unit (used in transformers)
- **nn.SiLU** - Sigmoid Linear Unit (Swish): x * sigmoid(x)
- **nn.Mish** - Mish activation: x * tanh(softplus(x))
- **nn.Hardswish** - Hard Swish (efficient approximation of Swish)
- **nn.Hardsigmoid** - Hard Sigmoid (efficient approximation)

#### Classic Activations
- **nn.Sigmoid** - Sigmoid: 1 / (1 + exp(-x))
- **nn.Tanh** - Hyperbolic tangent
- **nn.Softmax** - Softmax activation
- **nn.LogSoftmax** - Log(Softmax(x))
- **nn.Softmax2d** - Softmax over spatial dimensions

### 4. Normalization Layers (9 modules)
#### Batch Normalization
- **nn.BatchNorm1d** - Batch normalization for 1D/2D inputs
- **nn.BatchNorm2d** - Batch normalization for 4D inputs (N, C, H, W)
- **nn.BatchNorm3d** - Batch normalization for 5D inputs (N, C, D, H, W)

#### Instance Normalization
- **nn.InstanceNorm1d** - Instance normalization for 1D inputs
- **nn.InstanceNorm2d** - Instance normalization for 2D inputs (style transfer)
- **nn.InstanceNorm3d** - Instance normalization for 3D inputs

#### Other Normalization
- **nn.LayerNorm** - Layer normalization (used in transformers)
- **nn.GroupNorm** - Group normalization
- **nn.LocalResponseNorm** - Local Response Normalization (AlexNet)

### 5. Linear Layers (2 modules)
- **nn.Linear** - Fully connected layer (dense layer)
- **nn.Bilinear** - Bilinear transformation

### 6. Dropout Layers (5 modules)
- **nn.Dropout** - Randomly zero elements with probability p
- **nn.Dropout1d** - Randomly zero entire channels (1D)
- **nn.Dropout2d** - Randomly zero entire channels (2D)
- **nn.Dropout3d** - Randomly zero entire channels (3D)
- **nn.AlphaDropout** - Alpha Dropout (for SELU activation)

### 7. Utility Layers (11 modules)
#### Shape Manipulation
- **nn.Flatten** - Flatten tensor to 1D (keeping batch dimension)
- **nn.Unflatten** - Unflatten a tensor dimension
- **nn.Identity** - Pass-through (no-op)

#### Upsampling
- **nn.Upsample** - Spatial upsampling
- **nn.UpsamplingNearest2d** - 2D nearest-neighbor upsampling
- **nn.UpsamplingBilinear2d** - 2D bilinear upsampling
- **nn.PixelShuffle** - Rearrange (*, C*r^2, H, W) to (*, C, H*r, W*r)
- **nn.PixelUnshuffle** - Reverse of PixelShuffle

#### Padding
- **nn.ZeroPad2d** - Zero padding for 2D inputs
- **nn.ReflectionPad2d** - Reflection padding for 2D inputs
- **nn.ReplicationPad2d** - Replication padding for 2D inputs

## Usage in Model Designer

### Accessing the Catalog
The module catalog is accessible via the API endpoint:
```
GET /api/modules/catalog
```

This returns all available modules including:
- Built-in Ultralytics modules (Conv, C2f, SPPF, Detect, etc.)
- PyTorch nn modules (all 59 listed above)
- Custom user-defined modules

### Module Structure
Each module in the catalog has the following structure:
```json
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
    {"name": "bias", "type": "bool", "default": true}
  ],
  "source": "torch.nn"
}
```

### Using in YAML Model Definitions
PyTorch nn modules can be used directly in YAML model configurations:

```yaml
# Example: Custom model with PyTorch nn blocks
backbone:
  - [-1, 1, nn.Conv2d, [64, 3, 1, 1]]      # Conv2d layer
  - [-1, 1, nn.BatchNorm2d, [64]]          # BatchNorm
  - [-1, 1, nn.ReLU, []]                   # ReLU activation
  - [-1, 1, nn.MaxPool2d, [2, 2, 0]]       # MaxPool
  - [-1, 1, nn.Linear, [128]]              # Fully connected
  - [-1, 1, nn.Dropout, [0.5]]             # Dropout
  - [-1, 1, nn.Flatten, [1, -1]]           # Flatten

head:
  - [-1, 1, Detect, [80]]                  # YOLO detection head
```

## Implementation Details

### File Location
`backend/app/services/module_registry.py`

### Key Functions
- `get_all_modules()` - Returns all available modules (built-in + torch.nn + custom)
- `get_module_info(name)` - Look up a specific module by name
- `get_categories()` - Return all unique module categories

### Categories
The system organizes modules into the following categories:
1. **basic** - Basic Ultralytics operations (Conv, DWConv, etc.)
2. **composite** - Composite blocks (C2f, C3, Bottleneck, SPPF, etc.)
3. **attention** - Attention mechanisms (CBAM, PSA)
4. **head** - Task-specific heads (Detect, Segment, Pose, Classify, OBB)
5. **specialized** - Specialized modules (TorchVision, Index)
6. **convolution** - PyTorch convolution layers
7. **pooling** - PyTorch pooling layers
8. **activation** - PyTorch activation functions
9. **normalization** - PyTorch normalization layers
10. **linear** - PyTorch linear layers
11. **dropout** - PyTorch dropout layers
12. **utility** - PyTorch utility layers

## Benefits

### Flexibility
Users can now build completely custom architectures mixing:
- Ultralytics YOLO components
- Standard PyTorch nn layers
- Custom user-defined modules

### Compatibility
All PyTorch nn modules are fully compatible with:
- Ultralytics training pipeline
- Model export (ONNX, TorchScript, etc.)
- Weight transfer system
- Gradient and weight monitoring

### Use Cases
1. **Custom Backbones** - Build backbones with specific conv/norm/activation combinations
2. **Experimental Architectures** - Test different activation functions or normalization strategies
3. **Domain-Specific Models** - Use 1D/3D convolutions for audio/video tasks
4. **Transfer Learning** - Mix pretrained YOLO components with custom layers
5. **Research** - Rapid prototyping of novel architectures

## Notes

- All modules support the standard YAML format: `[from, repeats, module, args]`
- The first argument for most layers is automatically inferred from the previous layer's output channels
- For layers like `nn.Linear`, you only need to specify `out_features` (in_features is auto-calculated)
- Boolean and string arguments are properly typed in the schema
- Default values are provided for all optional parameters
