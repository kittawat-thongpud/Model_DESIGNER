# PyTorch nn Blocks - Quick Reference

## Summary
**59 PyTorch nn modules** added to Model Designer catalog across 7 categories.

## By Category

### Convolution (5)
```
nn.Conv1d, nn.Conv2d, nn.Conv3d
nn.ConvTranspose1d, nn.ConvTranspose2d
```

### Pooling (12)
```
nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d
nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d
nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d
nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d
```

### Activation (15)
```
nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU, nn.ELU
nn.GELU, nn.SiLU, nn.Mish
nn.Sigmoid, nn.Tanh
nn.Softmax, nn.LogSoftmax, nn.Softmax2d
nn.Hardswish, nn.Hardsigmoid
```

### Normalization (9)
```
nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
nn.LayerNorm, nn.GroupNorm
nn.LocalResponseNorm
```

### Linear (2)
```
nn.Linear, nn.Bilinear
```

### Dropout (5)
```
nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d
nn.AlphaDropout
```

### Utility (11)
```
nn.Flatten, nn.Unflatten, nn.Identity
nn.Upsample, nn.UpsamplingNearest2d, nn.UpsamplingBilinear2d
nn.PixelShuffle, nn.PixelUnshuffle
nn.ZeroPad2d, nn.ReflectionPad2d, nn.ReplicationPad2d
```

## Common Usage Examples

### Basic CNN Block
```yaml
- [-1, 1, nn.Conv2d, [64, 3, 1, 1]]
- [-1, 1, nn.BatchNorm2d, [64]]
- [-1, 1, nn.ReLU, []]
- [-1, 1, nn.MaxPool2d, [2]]
```

### Residual Block
```yaml
- [-1, 1, nn.Conv2d, [128, 3, 1, 1]]
- [-1, 1, nn.BatchNorm2d, [128]]
- [-1, 1, nn.ReLU, []]
- [-1, 1, nn.Conv2d, [128, 3, 1, 1]]
- [-1, 1, nn.BatchNorm2d, [128]]
```

### Classifier Head
```yaml
- [-1, 1, nn.AdaptiveAvgPool2d, [1]]
- [-1, 1, nn.Flatten, []]
- [-1, 1, nn.Dropout, [0.5]]
- [-1, 1, nn.Linear, [1000]]
```

### Upsampling Block
```yaml
- [-1, 1, nn.ConvTranspose2d, [64, 3, 2, 1, 1]]
- [-1, 1, nn.BatchNorm2d, [64]]
- [-1, 1, nn.ReLU, []]
```

### Modern Transformer-style
```yaml
- [-1, 1, nn.LayerNorm, [512]]
- [-1, 1, nn.Linear, [2048]]
- [-1, 1, nn.GELU, []]
- [-1, 1, nn.Dropout, [0.1]]
- [-1, 1, nn.Linear, [512]]
```

## API Endpoints

### Get Full Catalog
```
GET /api/modules/catalog
```

### Get Categories
```
GET /api/modules/catalog/categories
```

### Get Module Info
```
GET /api/modules/catalog/{module_name}
```

## Module Arguments Format

Arguments are specified as a list in YAML:
```yaml
[arg1, arg2, arg3, ...]
```

For most layers, the first argument (input channels/features) is **automatically inferred** from the previous layer.

### Examples:
- `nn.Conv2d, [64, 3, 1, 1]` → out_channels=64, kernel=3, stride=1, padding=1
- `nn.Linear, [128]` → out_features=128 (in_features auto-inferred)
- `nn.Dropout, [0.5]` → p=0.5
- `nn.ReLU, []` → no arguments needed

## Notes

✅ All 59 modules are production-ready  
✅ Compatible with Ultralytics training pipeline  
✅ Support for ONNX/TorchScript export  
✅ Full parameter documentation in main docs  
✅ Type-safe argument definitions  

See `docs/pytorch_nn_blocks.md` for detailed documentation.
