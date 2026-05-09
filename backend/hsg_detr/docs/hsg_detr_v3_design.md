# HSG-DETR v3 Design Notes

## Goal

HSG-DETR v3 stops treating sparse context as an ad-hoc selected-token
self-attention block.  The new path is split into scoring, reference selection,
local sparse aggregation, and residual fusion so every parameter group and
metric has a clear role.

## References

- RT-DETR / RT-DETRv2: keep the detector real-time by using efficient hybrid
  encoding and query selection instead of dense DETR-style global attention.
- Deformable DETR: sparse attention is strongest when tied to spatial reference
  points and a small set of sampled keys, not all-pairs attention over arbitrary
  selected tokens.
- DINO / DN-DETR: query/proposal quality and denoising matter for DETR
  convergence; v3 keeps the decoder path explicit so query-side changes can be
  ablated separately.
- AdamW: optimizer groups use decoupled weight decay and separate new modules,
  scale parameters, norms/biases, and decoder weights.

## Module

`ReferenceGuidedSparseBlock` is channel-preserving and can replace
`SGTokenBlock` in YAML files.

Pipeline:

1. `TokenScorer`
   - Learns per-token objectness-like scores.
   - Adds a detached L2 activation prior so zero-initialized scorer heads do not
     select fixed arbitrary positions early in training.
2. Top-k reference selection
   - Uses detached scores for hard selection.
   - Keeps reference count controlled by `ratio`.
3. `LocalSparseAggregator`
   - Uses each selected token as a reference point.
   - Aggregates a local `window_size x window_size` patch with attention.
   - Complexity is `O(k * window_size^2)` instead of `O(k^2)`.
4. LayerScale residual fusion
   - Uses `gamma` with the same floor/STE behavior as v2 so the sparse path
     cannot vanish silently.

## Optimizer Roles

Trainer grouping recognizes both `SGTokenBlock` and
`ReferenceGuidedSparseBlock`.

- `base`: inherited backbone/neck weights, `1.0x lr`, normal weight decay.
- `sgb_sparse`: scorer and aggregator, `2.0x lr`, normal weight decay.
- `sgb_gamma`: LayerScale gamma, `5.0x lr`, no weight decay.
- `norm_bias`: norms and biases, `1.0x lr`, no weight decay.
- `decoder`: RT-DETR decoder, `1.5x lr`, normal weight decay.

The multipliers are intentionally conservative enough for pretrained transfer
while giving random-init scorer/aggregator parameters room to learn.

## Evidence Required

Do not accept v3 as better from architecture alone.  Use short ablations before
long COCO runs:

1. v2 full baseline.
2. v3 full.
3. v3 with v2 token ratios.
4. v3 without local sparse aggregation.
5. v3 without scorer prior.

Minimum acceptance:

- `mAP50_95` improves against the same seed/config baseline.
- Non-finite gradient skips remain rare.
- SGB metrics show non-zero scorer/aggregator gradients.
- Selected reference positions vary with images.
- Epoch time / memory increase is justified by AP gain.
