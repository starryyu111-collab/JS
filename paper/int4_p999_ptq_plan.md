# INT4-P99.9 Fixed Activation Clipping Baseline Plan Review and Revised Implementation Plan

## Review Findings

The existing INT4-P99.9 plan is directionally correct and already captures the
most important point: fixed percentile clipping means every activation layer
uses the same percentile rule, not the same numeric alpha value. Several details
need to be tightened before implementation so this baseline is comparable with
the implemented INT4-MinMax baseline.

1. The plan must be updated for the current repository state.
   INT4-MinMax is now implemented in:

   ```text
   src/quant/int4_minmax.py
   src/experiments/run_int4_minmax_ptq.py
   configs/int4_minmax_cifar10.yaml
   tests/test_int4_minmax.py
   ```

   Therefore INT4-P99.9 should be specified as a small, controlled variation of
   the existing INT4-MinMax flow. The only intended behavioral difference is the
   source of the post-ReLU activation clipping upper bound.

2. The fixed rule must be exactly P99.9 for every observed activation layer.
   The percentile value is fixed at `99.9` and must not be tuned, swept, or
   selected per layer. Each layer still computes its own `alpha_l` from its own
   calibration activation distribution.

3. The alpha value must be layer-specific.
   INT4-P99.9 must not compute one global alpha across all layers. For each
   observed post-ReLU activation module `l`:

   ```text
   alpha_l = percentile(activation_l, 99.9)
   ```

   Different layers are expected to have different `alpha_l` values because
   their activation distributions are different.

4. P99.9 calibration must use the element-wise distribution for each layer.
   The implementation should collect all calibration activation elements for a
   given layer and compute the P99.9 threshold from the concatenated layer
   distribution. It must not average per-batch percentiles.

5. All other quantization settings must match INT4-MinMax exactly.
   Weight quantization, activation dtype, integer ranges, post-ReLU site policy,
   fake quantization formula, FP32 checkpoint loading, calibration split, test
   split, smoke/full result behavior, and common CSV fields should stay aligned
   with INT4-MinMax.

6. P99.9 must remain clearly separate from MSE-Selected.
   INT4-P99.9 does not search candidate thresholds and does not use MSE to pick
   `alpha_l`. It may still report `activation_mse` as an evaluation metric, but
   that metric is not part of threshold selection.

## Revised Objective

Implement the `INT4-P99.9 fixed clipping` baseline for CIFAR-10 post-training
quantization.

This baseline uses one fixed percentile rule for all activation layers:

```text
activation_percentile = 99.9
activation_quantile = 0.999
```

For every observed post-ReLU activation layer, compute that layer's own
clipping threshold from its own calibration activation distribution:

```text
alpha_l = Q_l(0.999)
```

where `Q_l` is the empirical quantile of all collected calibration activation
elements for layer `l`.

The final experiment must output:

```text
top1_accuracy
accuracy_drop
activation_mse
logit_mse
```

The final full-evaluation result must be saved to:

```text
outputs/results/int4_p999_result.csv
```

## Relationship To INT4-MinMax

INT4-P99.9 must be identical to INT4-MinMax except for the activation clipping
upper-bound source.

| Setting | INT4-MinMax | INT4-P99.9 |
|---|---|---|
| Weight dtype | INT4 | INT4 |
| Weight range | `[-7, 7]` | `[-7, 7]` |
| Weight granularity | per-channel | per-channel |
| Weight symmetry | symmetric | symmetric |
| Weight zero-point | `0` | `0` |
| Bias | FP32 | FP32 |
| Activation dtype | UINT4 | UINT4 |
| Activation range | `[0, 15]` | `[0, 15]` |
| Activation target | post-ReLU | post-ReLU |
| Activation granularity | per-tensor per ReLU module | per-tensor per ReLU module |
| Activation clip interval | `[0, observed_max_l]` | `[0, P99.9_l]` |
| Alpha source | calibration maximum | calibration percentile |
| Percentile value | not used | fixed `99.9` |
| Threshold search | no | no |
| MSE-selected threshold | no | no |

The implementation should reuse the INT4-MinMax constants and fake-quantization
logic where possible. If direct reuse makes names misleading, a small common
module may be introduced:

```text
src/quant/int4_common.py
```

That common module may contain shared INT4 constants, qparam dataclasses,
per-channel weight fake quantization, post-ReLU activation fake quantization,
wrappers, and metric helpers. It must not contain experiment orchestration, and
any such refactor must preserve existing INT4-MinMax behavior and tests.

## Key Distinction From MSE-Selected

### INT4-P99.9 Fixed Clipping

- Uses the same fixed percentile rule for all activation layers: `P99.9`.
- Computes one layer-specific alpha from each layer's own calibration
  distribution.
- Does not search over candidate percentiles.
- Does not compare candidate thresholds using MSE.
- Does not choose a different percentile per layer.

For layer `l`:

```text
alpha_l = percentile(activation_l, 99.9)
```

The percentile is the same for every layer, but the resulting alpha value can be
different for every layer.

### INT4-MSE-Selected Clipping

- Uses a predefined candidate percentile set, for example:

  ```text
  {99.0, 99.5, 99.9, 99.95, 100.0}
  ```

- Converts each candidate percentile into a candidate clipping threshold for
  each layer.
- Runs fake quantization and dequantization for every candidate threshold.
- Computes activation reconstruction MSE for every candidate.
- Selects the threshold with the minimum MSE for that layer.

For layer `l`:

```text
p_l* = argmin_p MSE(x_l, fake_quant_dequant(clip(x_l, 0, percentile(x_l, p))))
alpha_l = percentile(x_l, p_l*)
```

where `p` is restricted to the predefined candidate set.

Therefore, INT4-P99.9 is a fixed percentile baseline, while INT4-MSE-Selected is
a layer-wise threshold selection method. INT4-P99.9 may report
`activation_mse`, but it must not use activation MSE to choose thresholds.

## Scope

### Planned File Changes For Implementation

Implementation should add:

```text
src/quant/int4_p999.py
src/experiments/run_int4_p999_ptq.py
configs/int4_p999_cifar10.yaml
tests/test_int4_p999.py
```

Optional only if needed to keep INT4-MinMax and INT4-P99.9 synchronized:

```text
src/quant/int4_common.py
```

If a common module is added, the INT4-MinMax tests must still pass and the
result schema of INT4-MinMax must not change unexpectedly.

### Files That Should Not Change

This baseline should not modify unrelated experiment behavior:

```text
src/experiments/train_fp32.py
src/quant/int8_minmax.py
src/experiments/run_int8_minmax_ptq.py
configs/fp32_cifar10.yaml
configs/int8_minmax_cifar10.yaml
outputs/results/fp32_result.csv
outputs/results/int8_minmax_result.csv
```

Existing INT8-MinMax behavior must remain unchanged. Existing INT4-MinMax
behavior should remain unchanged except for any carefully tested helper
extraction into `src/quant/int4_common.py`.

## Quantization Specification

### Shared Named Constants

The implementation should use named constants rather than scattering bit-range
or percentile magic numbers through the code:

```text
WEIGHT_INT4_QMIN = -7
WEIGHT_INT4_QMAX = 7
ACTIVATION_UINT4_QMIN = 0
ACTIVATION_UINT4_QMAX = 15
FIXED_ACTIVATION_PERCENTILE = 99.9
FIXED_ACTIVATION_QUANTILE = 0.999
```

The percentile value must remain fixed at `99.9` for this baseline.

### Weight Quantization

Target modules:

- `torch.nn.Conv2d`
- `torch.nn.Linear`

Method:

- Per-channel symmetric fake quantization.
- Channel axis: output channel dimension, `dim=0`.
- Integer dtype: signed INT4.
- Integer range: `[-7, 7]`.
- Zero-point: `0`.
- Bias remains FP32.

For each output channel `c`:

```text
qmin_w = -7
qmax_w = 7
max_abs_c = max(abs(weight_c))
scale_c = max_abs_c / qmax_w
```

If `max_abs_c == 0`, use a finite fallback scale:

```text
scale_c = 1.0
```

Fake quantization:

```text
q_w = clamp(round(weight / scale_c), -7, 7)
dequant_w = q_w * scale_c
```

Expected broadcast scale shapes:

- Conv2d: `[out_channels, 1, 1, 1]`
- Linear: `[out_features, 1]`

Weight fake quantization must be applied only to a deep copy of the loaded FP32
model. The FP32 reference model must remain unchanged.

### Activation Quantization

Target tensors:

- Outputs of post-ReLU activation modules only.

Method:

- Per-tensor affine fake quantization.
- Integer dtype: UINT4.
- Integer range: `[0, 15]`.
- Real clipping interval: `[0, alpha_l]`.
- `alpha_l` comes from the P99.9 value of that layer's calibration activation
  distribution.

For each post-ReLU activation layer `l`:

```text
qmin_a = 0
qmax_a = 15
percentile = 99.9
alpha_l = percentile(x_l, 99.9)
scale_l = alpha_l / 15
zero_point_l = 0
```

If `alpha_l <= 0` or the scale is not finite, use the same finite fallback
policy as INT4-MinMax:

```text
clip_max_l = 0.0
scale_l = 1.0
zero_point_l = 0
```

Fake quantization:

```text
x_clipped = clamp(x, 0, alpha_l)
q_x = clamp(round(x_clipped / scale_l), 0, 15)
dequant_x = q_x * scale_l
```

Because the activation target is post-ReLU, `zero_point_l` must be `0`.

## Percentile Calibration Policy

Calibration source:

- CIFAR-10 training split only: `train=True`.

Calibration target:

- Post-ReLU activation outputs.

Percentile definition:

- The P99.9 value must be computed over the element-wise activation
  distribution collected for each layer across the calibration subset.
- It must not be computed as an average of per-batch percentiles.
- It must not be computed from a global distribution shared across layers.
- It must not use CIFAR-10 `train=False`.

For layer `l`, collect:

```text
x_l = concat(all post-ReLU activation elements for layer l on calibration set)
alpha_l = quantile(x_l, 0.999)
```

Exact collection is acceptable for the default small calibration size. If a
memory-saving approximate quantile is introduced later, the approximation must
be documented and tested against a small exact case.

## Post-ReLU Site Policy

The baseline should use the same simple and auditable activation site policy as
INT4-MinMax:

1. Identify `nn.ReLU` modules as activation quantization targets.
2. During calibration, collect post-ReLU outputs for each named ReLU module.
3. Compute each module's own P99.9 alpha.
4. During evaluation, wrap those same ReLU modules so their outputs are clipped
   and fake-quantized using their calibrated alpha.
5. Log and write the observed post-ReLU site names.

Known limitation:

- In CIFAR ResNet blocks, one ReLU module can be reused twice in the same
  forward pass. A module-level observer aggregates both calls under one module
  name. This is acceptable as a per-ReLU-module baseline if documented.
- It must not be described as strict call-site-specific layer-wise clipping.

If strict call-site separation is required later, it should be implemented as a
separate change using call-order-aware observers or explicit model rewrites.

## Experiment Protocol

The INT4-P99.9 experiment should run in this order:

1. Load `configs/int4_p999_cifar10.yaml`.
2. Set the experiment seed.
3. Resolve the device.
4. Build the FP32 model using `src.models.build_model`.
5. Load the FP32 checkpoint using `src.utils.checkpoint.load_checkpoint`.
6. Verify checkpoint `model_name`, if present, matches the configured model.
7. Set the FP32 model to `eval()`.
8. Recompute FP32 top-1 accuracy on CIFAR-10 `train=False`.
9. Deep-copy the loaded FP32 model to create the INT4-P99.9 simulated model.
10. Calibrate post-ReLU activation P99.9 thresholds on a deterministic
    CIFAR-10 `train=True` subset.
11. Apply per-channel symmetric INT4 weight fake quantization to the INT4 copy.
12. Attach post-ReLU activation fake quantization using the calibrated P99.9
    alpha values.
13. Evaluate INT4-P99.9 top-1 accuracy on CIFAR-10 `train=False`.
14. Compute local post-ReLU `activation_mse` on CIFAR-10 `train=False`.
15. Compute `logit_mse` between FP32 logits and INT4-P99.9 fake-quant logits on
    CIFAR-10 `train=False`.
16. Write `outputs/results/int4_p999_result.csv`.

No step may call:

```text
model.train()
optimizer.step()
loss.backward()
save_checkpoint()
```

BatchNorm statistics must not be updated during calibration or evaluation.

## Data Protocol

### Calibration Set

Source:

- `torchvision.datasets.CIFAR10(train=True)`

Sampling:

- Deterministic random subset using the experiment seed.
- Default size: `1024`, configurable.

Transform:

- `ToTensor`
- CIFAR-10 normalization using the same mean and std as FP32, INT8-MinMax, and
  INT4-MinMax.

No training-time augmentation:

- No `RandomCrop`
- No `RandomHorizontalFlip`

Calibration is used only to compute P99.9 activation thresholds. It must not be
used for final accuracy or final MSE evaluation.

### Evaluation Set

Source:

- `torchvision.datasets.CIFAR10(train=False)`

Requirement for final accepted run:

```text
test_size = 10000
evaluated_test_size = 10000
is_smoke = false
```

Smoke runs may use batch limits, but smoke results should not overwrite the
final full-evaluation CSV unless `--result-path` is explicitly provided.

Preferred smoke output path:

```text
outputs/results/int4_p999_result_smoke.csv
```

## Metrics

Required output metrics:

```text
top1_accuracy
accuracy_drop
activation_mse
logit_mse
```

Definitions:

- `top1_accuracy`: INT4-P99.9 top-1 accuracy on CIFAR-10 test set, reported as a
  percentage in `[0, 100]`.
- `fp32_top1_accuracy`: FP32 checkpoint top-1 accuracy recomputed by the
  INT4-P99.9 script on the same test set.
- `int4_top1_accuracy`: same value as `top1_accuracy`, included for explicit
  method readability.
- `accuracy_drop`: `fp32_top1_accuracy - int4_top1_accuracy`, in percentage
  points.
- `activation_mse`: element-weighted local reconstruction MSE between FP32
  post-ReLU activation tensors and the same tensors after P99.9 clipping plus
  calibrated UINT4 fake quantization and dequantization.
- `logit_mse`: element-weighted MSE between FP32 logits and INT4-P99.9
  fake-quant model logits.

`activation_mse` and `logit_mse` must be measured on the evaluation loader, not
the calibration loader.

## Result CSV Schema

The final CSV must be saved to:

```text
outputs/results/int4_p999_result.csv
```

The row should include at least these common columns so it can be compared with
FP32, INT8-MinMax, and INT4-MinMax:

```text
method
model
dataset
seed
checkpoint_path
checkpoint_model_name
calibration_size
calibration_seed
calibration_source
calibration_num_batches
calibration_index_checksum
test_size
evaluated_test_size
top1_accuracy
fp32_top1_accuracy
int4_top1_accuracy
accuracy_drop
activation_mse
logit_mse
activation_quant_dtype
activation_qmin
activation_qmax
weight_qmin
weight_qmax
num_observed_activation_sites
num_quantized_modules
min_activation_scale
max_activation_scale
min_activation_zero_point
max_activation_zero_point
min_weight_scale
max_weight_scale
is_smoke
result_path
log_path
device
batch_size
observed_activation_sites
```

INT4-P99.9-specific columns should include:

```text
activation_clip_method
activation_percentile
activation_quantile
activation_clip_source
activation_clip_min
activation_granularity
activation_site_type
threshold_search
mse_selected
candidate_percentiles
min_activation_alpha
max_activation_alpha
weight_granularity
weight_symmetry
```

Recommended values:

```text
method = INT4-P99.9
activation_clip_method = fixed_percentile
activation_percentile = 99.9
activation_quantile = 0.999
activation_clip_source = calibration_percentile
activation_clip_min = 0
activation_granularity = per_tensor_per_relu_module
activation_site_type = post_relu
threshold_search = false
mse_selected = false
candidate_percentiles =
weight_granularity = per_channel
weight_symmetry = symmetric
```

## Config Draft

Recommended `configs/int4_p999_cifar10.yaml`:

```yaml
dataset:
  name: CIFAR-10
  data_dir: data
  calibration_size: 1024
  num_workers: 2

model:
  name: resnet18_cifar
  num_classes: 10

experiment:
  seed: 42
  deterministic: true
  device: auto
  batch_size: 128

quantization:
  method: INT4-P99.9
  weight:
    dtype: int4
    qmin: -7
    qmax: 7
    granularity: per_channel
    symmetry: symmetric
    channel_axis: 0
  activation:
    dtype: uint4
    qmin: 0
    qmax: 15
    granularity: per_tensor
    symmetry: affine
    site: post_relu
    clip_min: 0
    clip_method: fixed_percentile
    percentile: 99.9
    quantile: 0.999
    clip_max_source: calibration_percentile

paths:
  checkpoint_path: checkpoints/fp32_best.pt
  result_path: outputs/results/int4_p999_result.csv
  log_path: outputs/logs/int4_p999_ptq.log

smoke:
  max_calibration_batches:
  max_test_batches:
```

## Testing Plan

### Tensor-Level Tests

Verify:

- Weight fake quantization preserves input tensor shape.
- Weight quantized integer values are clamped to `[-7, 7]`.
- Per-channel weight scale shape is correct for Conv2d and Linear weights.
- Activation fake quantization uses UINT4 `[0, 15]`.
- Activation clipping uses real interval `[0, alpha_l]`.
- Activation zero-point is `0`.
- Scales are finite and positive.
- Degenerate all-zero activation alpha uses a finite fallback scale.
- Fake-quantized outputs are finite.

### Percentile Tests

Verify:

- The fixed percentile value is exactly `99.9`.
- The quantile value is exactly `0.999`.
- Two layers with different activation distributions can produce different
  alpha values.
- The alpha is computed from each layer's own activation distribution.
- The implementation does not average per-batch percentiles.
- The implementation does not compute one global alpha across layers.
- The implementation does not perform candidate threshold search.
- The implementation does not compute MSE during threshold selection.

### Observer and Wrapper Tests

Verify:

- Observed activation site names are post-ReLU sites.
- The same site names are wrapped for fake quantization.
- Calibration processes at least one batch.
- ReLU wrapper output remains nonnegative and finite.
- Reused ReLU modules are documented as module-level sites.

### MSE Tests

Verify:

- `activation_mse.num_batches > 0`.
- `activation_mse.num_elements > 0`.
- `activation_mse.mse >= 0`.
- `logit_mse >= 0`.
- MSE values are finite.
- `activation_mse` uses the same P99.9 clipping and fake quantization path as
  the evaluation wrapper.
- MSE is an evaluation metric only and is not used to select alpha.

### Experiment Smoke Test

Run:

```bash
python -m src.experiments.run_int4_p999_ptq --config configs/int4_p999_cifar10.yaml --max-calibration-batches 1 --max-test-batches 1
```

Check:

- Required metrics are printed.
- Smoke result path is `outputs/results/int4_p999_result_smoke.csv` unless
  overridden.
- `is_smoke = true`.
- `activation_percentile = 99.9`.
- `activation_quantile = 0.999`.
- `threshold_search = false`.
- `mse_selected = false`.
- Calibration source is `CIFAR10 train=True`.
- Evaluation source is `CIFAR10 train=False`.

### Full Evaluation

Run:

```bash
python -m src.experiments.run_int4_p999_ptq --config configs/int4_p999_cifar10.yaml
```

Check:

- `outputs/results/int4_p999_result.csv` exists.
- `is_smoke = false`.
- `test_size = 10000`.
- `evaluated_test_size = 10000`.
- `accuracy_drop = fp32_top1_accuracy - int4_top1_accuracy` up to formatting
  precision.
- CSV includes `top1_accuracy`, `accuracy_drop`, `activation_mse`, and
  `logit_mse`.
- CSV identifies the method as fixed P99.9, not MSE-selected.

## Acceptance Checklist

Before considering INT4-P99.9 complete:

1. Weight quantization is per-channel symmetric INT4 with range `[-7, 7]`.
2. Activation quantization is per-tensor affine UINT4 with range `[0, 15]`.
3. Activation fake quantization is applied only after ReLU.
4. Every observed activation layer uses the fixed percentile rule `P99.9`.
5. Every observed activation layer computes its own alpha from its own
   calibration activation distribution.
6. The implementation does not compute one global alpha for all layers.
7. The implementation does not average per-batch percentiles.
8. The implementation does not use MSE to choose thresholds.
9. The implementation does not search over percentile candidates.
10. Calibration uses CIFAR-10 `train=True`.
11. Accuracy and MSE evaluation use CIFAR-10 `train=False`.
12. No QAT, retraining, fine-tuning, optimizer step, or checkpoint update
    occurs.
13. The four required metrics are written.
14. Result path is `outputs/results/int4_p999_result.csv`.
15. Final evaluation uses the full CIFAR-10 test set.
16. The CSV can be compared with FP32, INT8-MinMax, and INT4-MinMax through
    common columns.
17. The CSV explicitly records `threshold_search=false` and
    `mse_selected=false`.

## Anti-Requirements

The implementation must not:

- Use one shared alpha value for all activation layers.
- Use a configurable percentile sweep for this baseline.
- Select percentile values per layer.
- Use MSE to select the clipping threshold.
- Reuse the calibration set for final accuracy.
- Change INT4-MinMax quantization behavior.
- Claim real INT4 hardware acceleration.
- Describe module-level reused ReLU observation as strict call-site-specific
  layer-wise clipping.

## Implementation Notes

This baseline should remain intentionally simple and interpretable. It is a
fixed percentile clipping baseline, not the proposed MSE-selected method.

If INT4-P99.9 accuracy is poor, that is not automatically a bug. The immediate
debug signals should be:

- Per-layer activation alpha range.
- Activation scale range.
- Weight scale range.
- Number and names of observed post-ReLU sites.
- Activation MSE.
- Logit MSE.
- Accuracy drop against recomputed FP32 accuracy.

The paper should describe this as simulated PTQ/fake quantization and must not
claim real INT4 hardware acceleration.
