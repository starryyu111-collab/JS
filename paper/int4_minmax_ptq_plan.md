# INT4-MinMax PTQ Plan Review and Revised Implementation Plan

## Review Findings

The initial INT4-MinMax plan is directionally correct, but several details need
to be made explicit before implementation. These details matter because this
baseline will be compared directly with FP32 and INT8-MinMax in the main result
table.

1. Activation placement must be post-ReLU, not Conv2d or Linear output.
   The existing INT8-MinMax implementation observes and fake-quantizes target
   module outputs from `Conv2d` and `Linear`. That is acceptable for the INT8
   sanity check, but it does not satisfy the INT4 requirement. INT4-MinMax must
   observe and fake-quantize only activation tensors after ReLU.

2. Activation quantization interval must be `[0, max]`.
   Because the activation target is post-ReLU, negative activation values are
   not part of the intended calibration interval. The affine UINT4 parameters
   should be derived from a fixed real interval `[0, observed_max]`, where
   `observed_max` is collected from the calibration set.

3. UINT4 activation zero-point should be fixed by the `[0, max]` interval.
   For activation range `[0, max]` and integer range `[0, 15]`, the natural
   affine quantization parameters are:

   ```text
   qmin_a = 0
   qmax_a = 15
   scale_a = observed_max / 15
   zero_point_a = 0
   ```

   This is still per-tensor affine UINT4, but the nonnegative post-ReLU interval
   makes the zero-point exactly zero. If `observed_max <= 0`, a finite fallback
   scale such as `1.0` should be used.

4. Weight INT4 range must be symmetric and explicit.
   The weight range should be `[-7, 7]`, not `[-8, 7]`, so the signed range is
   symmetric and the weight zero-point is exactly `0`.

5. ResNet ReLU module reuse needs an implementation decision.
   In `src/models/resnet_cifar.py`, each `BasicBlock` reuses `self.relu` twice:
   once after `bn1`, and once after the residual addition. If hooks are attached
   directly to ReLU modules, both call sites share one module name and therefore
   one calibration statistic. This is still post-ReLU, but it is not call-site
   specific. The simplest acceptable baseline is to document this as
   "per ReLU module site". If stricter layer-wise call-site separation is needed,
   the implementation should use call-order-aware hooks or explicit wrappers.

6. In-place ReLU needs care.
   Both model families use `nn.ReLU(inplace=True)`. A wrapper that returns a
   fake-quantized tensor after ReLU is safer than trying to mutate tensors from
   hooks. The implementation should avoid relying on a forward hook return value
   unless it is clearly tested with in-place ReLU behavior.

7. Metrics must keep the same meaning as INT8-MinMax.
   `activation_mse` should remain a local reconstruction metric: compare FP32
   post-ReLU activation tensors with the same tensors after calibrated UINT4
   fake quantization and dequantization. `logit_mse` should remain the
   end-to-end MSE between FP32 logits and INT4 fake-quant model logits.

8. CSV compatibility needs a stable common schema.
   The INT4 result CSV should include the common columns already used by the
   INT8-MinMax result, especially `method`, `model`, `dataset`,
   `top1_accuracy`, `fp32_top1_accuracy`, `accuracy_drop`, `activation_mse`,
   `logit_mse`, calibration metadata, quantization ranges, `is_smoke`, and
   paths. INT4-specific fields can be added, but the common columns must stay
   stable so FP32, INT8-MinMax, and INT4-MinMax can be merged into one master
   table.

## Revised Objective

Implement an INT4-MinMax post-training quantization baseline for CIFAR-10.

The experiment must:

- Load the existing FP32 checkpoint.
- Avoid QAT, fine-tuning, retraining, or optimizer steps.
- Use per-channel symmetric INT4 fake quantization for weights.
- Use per-tensor affine UINT4 fake quantization only for post-ReLU activations.
- Use activation clipping interval `[0, max]`, where `max` comes only from the
  calibration set.
- Use CIFAR-10 `train=True` only for calibration.
- Use CIFAR-10 `train=False` only for final accuracy and MSE evaluation.
- Output `top1_accuracy`, `accuracy_drop`, `activation_mse`, and `logit_mse`.
- Save the final result to `outputs/results/int4_minmax_result.csv`.
- Keep the result compatible with FP32 and INT8-MinMax main-table comparison.

## Scope

### Planned File Changes

Implementation should add:

```text
src/quant/int4_minmax.py
src/experiments/run_int4_minmax_ptq.py
configs/int4_minmax_cifar10.yaml
tests/test_int4_minmax.py
```

Optional only if a master table utility is later requested:

```text
src/experiments/build_main_result_table.py
```

### Files That Should Not Change

The implementation should not modify:

```text
src/experiments/train_fp32.py
src/quant/int8_minmax.py
configs/fp32_cifar10.yaml
configs/int8_minmax_cifar10.yaml
outputs/results/fp32_result.csv
outputs/results/int8_minmax_result.csv
```

If small helper reuse becomes necessary, it should be done conservatively and
without changing INT8 behavior.

## Quantization Specification

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

If `max_abs_c == 0`, use a finite fallback scale such as `scale_c = 1.0`.

Fake quantization:

```text
q_w = clamp(round(weight / scale_c), qmin_w, qmax_w)
dequant_w = q_w * scale_c
```

Expected broadcast scale shapes:

- `Conv2d`: `[out_channels, 1, 1, 1]`
- `Linear`: `[out_features, 1]`

Weight fake quantization must be applied only to a deep copy of the loaded FP32
model. The FP32 model used for baseline accuracy and FP32 reference tensors must
remain unchanged.

### Activation Quantization

Target tensors:

- Outputs of post-ReLU activation sites only.

Method:

- Per-tensor affine fake quantization.
- Integer dtype: UINT4.
- Integer range: `[0, 15]`.
- Real clipping interval: `[0, observed_max]`.
- `observed_max` comes from calibration set post-ReLU outputs.

For each post-ReLU activation site:

```text
qmin_a = 0
qmax_a = 15
x_min = 0
x_max = observed_max_from_calibration
scale_a = x_max / (qmax_a - qmin_a)
zero_point_a = 0
```

If `x_max <= 0` or scale is not finite, use:

```text
scale_a = 1.0
zero_point_a = 0
```

Fake quantization:

```text
x_clipped = clamp(x, 0, x_max)
q_x = clamp(round(x_clipped / scale_a), 0, 15)
dequant_x = q_x * scale_a
```

The implementation should store `WEIGHT_INT4_QMIN`, `WEIGHT_INT4_QMAX`,
`ACTIVATION_UINT4_QMIN`, and `ACTIVATION_UINT4_QMAX` as named constants rather
than scattering bit-range numbers through the code.

## Post-ReLU Site Policy

The baseline should start with a simple and auditable site policy:

1. Identify `nn.ReLU` modules as activation quantization targets.
2. During calibration, collect the maximum post-ReLU output for each named ReLU
   module.
3. During INT4 evaluation, replace or wrap those same ReLU modules so their
   outputs are fake-quantized with the calibrated `[0, max]` UINT4 parameters.
4. Log and write the observed post-ReLU site names.

Known limitation:

- In CIFAR ResNet blocks, one ReLU module can be called twice in the same
  forward pass. The module-level baseline will share one `max` for those two
  calls. This is acceptable for INT4-MinMax baseline if documented. It should
  not be described as call-site-specific layer-wise clipping.

If a later method needs stricter layer-wise call-site separation, that should be
implemented as a separate change and documented separately.

## Experiment Protocol

The INT4-MinMax experiment should follow this order:

1. Load `configs/int4_minmax_cifar10.yaml`.
2. Set the experiment seed.
3. Resolve the device.
4. Build the FP32 model using `src.models.build_model`.
5. Load the FP32 checkpoint using `src.utils.checkpoint.load_checkpoint`.
6. Verify checkpoint `model_name`, if present, matches the configured model.
7. Set the FP32 model to `eval()`.
8. Recompute FP32 top-1 accuracy on CIFAR-10 `train=False`.
9. Deep-copy the loaded FP32 model to create the INT4 simulated model.
10. Calibrate post-ReLU activation maxima on a deterministic CIFAR-10
    `train=True` subset.
11. Apply per-channel symmetric INT4 weight fake quantization to the INT4 copy.
12. Attach post-ReLU activation fake quantization using the calibrated UINT4
    parameters.
13. Evaluate INT4 top-1 accuracy on CIFAR-10 `train=False`.
14. Compute local post-ReLU `activation_mse` on CIFAR-10 `train=False`.
15. Compute `logit_mse` between FP32 logits and INT4 fake-quant logits on
    CIFAR-10 `train=False`.
16. Write `outputs/results/int4_minmax_result.csv`.

No step may call:

```text
model.train()
optimizer.step()
loss.backward()
save_checkpoint()
```

BatchNorm statistics must not be updated during PTQ evaluation or calibration.

## Data Protocol

### Calibration Set

Source:

- `torchvision.datasets.CIFAR10(train=True)`

Sampling:

- Deterministic random subset using the experiment seed.
- Default size: `1024`, configurable.

Transform:

- `ToTensor`
- CIFAR-10 normalization using the same mean and std as FP32 and INT8.

No training-time augmentation:

- No `RandomCrop`
- No `RandomHorizontalFlip`

Calibration is used only to collect post-ReLU activation maxima. It must not be
used for final accuracy.

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
final full-evaluation CSV unless a `--result-path` override is provided. The
preferred default is:

```text
outputs/results/int4_minmax_result_smoke.csv
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

- `top1_accuracy`: INT4-MinMax top-1 accuracy on CIFAR-10 test set, reported as
  a percentage in `[0, 100]`.
- `fp32_top1_accuracy`: FP32 checkpoint top-1 accuracy recomputed by the INT4
  script on the same test set.
- `int4_top1_accuracy`: same value as `top1_accuracy`, included for explicit
  method readability.
- `accuracy_drop`: `fp32_top1_accuracy - int4_top1_accuracy`, in percentage
  points.
- `activation_mse`: element-weighted local reconstruction MSE between FP32
  post-ReLU activation tensors and their calibrated UINT4 fake-quantized and
  dequantized tensors.
- `logit_mse`: element-weighted MSE between FP32 logits and INT4 fake-quant
  model logits.

`activation_mse` and `logit_mse` should be measured on the evaluation loader,
not the calibration loader, unless a dedicated diagnostic mode is explicitly
added later.

## Result CSV Schema

The final INT4 CSV must be saved to:

```text
outputs/results/int4_minmax_result.csv
```

The row should include at least these common table columns:

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
```

Recommended INT4-specific columns:

```text
int4_top1_accuracy
activation_site_type
activation_clip_min
activation_clip_source
activation_granularity
weight_granularity
weight_symmetry
```

For main-table comparison, the core columns should be enough:

```text
method
model
dataset
top1_accuracy
fp32_top1_accuracy
accuracy_drop
activation_mse
logit_mse
activation_quant_dtype
weight_qmin
weight_qmax
activation_qmin
activation_qmax
calibration_size
test_size
evaluated_test_size
is_smoke
```

This makes INT4-MinMax comparable with FP32 and INT8-MinMax even if method-
specific diagnostic columns differ.

## Config Draft

Recommended `configs/int4_minmax_cifar10.yaml`:

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
  method: INT4-MinMax
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
    clip_max_source: calibration_max

paths:
  checkpoint_path: checkpoints/fp32_best.pt
  result_path: outputs/results/int4_minmax_result.csv
  log_path: outputs/logs/int4_minmax_ptq.log

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
- Activation clipping uses real interval `[0, max]`.
- Activation zero-point is `0`.
- Scales are finite and positive.
- Degenerate all-zero activation max uses a finite fallback scale.
- Fake-quantized outputs are finite.

### Observer and Wrapper Tests

Verify:

- Observed activation site names are post-ReLU sites.
- The same site names are wrapped for fake quantization.
- Calibration processes at least one batch.
- Activation calibration uses only a loader supplied by the experiment script.
- ReLU wrapper output remains nonnegative and finite.

### MSE Tests

Verify:

- `activation_mse.num_batches > 0`.
- `activation_mse.num_elements > 0`.
- `activation_mse.mse >= 0`.
- `logit_mse >= 0`.
- MSE values are finite.

### Experiment Smoke Test

Run:

```bash
python -m src.experiments.run_int4_minmax_ptq --config configs/int4_minmax_cifar10.yaml --max-calibration-batches 1 --max-test-batches 1
```

Check:

- Required metrics are printed.
- Smoke result path is `outputs/results/int4_minmax_result_smoke.csv` unless
  overridden.
- `is_smoke = true`.
- Calibration source is `CIFAR10 train=True`.
- Evaluation source is `CIFAR10 train=False`.

### Full Evaluation

Run:

```bash
python -m src.experiments.run_int4_minmax_ptq --config configs/int4_minmax_cifar10.yaml
```

Check:

- `outputs/results/int4_minmax_result.csv` exists.
- `is_smoke = false`.
- `test_size = 10000`.
- `evaluated_test_size = 10000`.
- `accuracy_drop = fp32_top1_accuracy - int4_top1_accuracy` up to formatting
  precision.
- CSV includes `top1_accuracy`, `accuracy_drop`, `activation_mse`, and
  `logit_mse`.
- CSV common columns can be merged with FP32 and INT8-MinMax rows.

## Acceptance Checklist

Before considering INT4-MinMax complete:

1. Weight quantization is per-channel symmetric INT4 with range `[-7, 7]`.
2. Activation quantization is per-tensor affine UINT4 with range `[0, 15]`.
3. Activation fake quantization is applied only after ReLU.
4. Activation clipping interval is `[0, max]`.
5. `max` comes only from the calibration set.
6. No QAT, retraining, fine-tuning, optimizer step, or checkpoint update occurs.
7. The four required metrics are written.
8. Result path is `outputs/results/int4_minmax_result.csv`.
9. Final evaluation uses the full CIFAR-10 test set.
10. The CSV can be compared with FP32 and INT8-MinMax through common columns.

## Implementation Notes

This baseline should remain intentionally simple. It is a MinMax PTQ baseline,
not the proposed clipping method. It should not use percentile thresholds,
MSE-selected thresholds, or any threshold search. Those belong to later
INT4-P99.9 and INT4-MSE-Selected experiments.

If INT4-MinMax accuracy is poor, that is not automatically a bug. The immediate
debug signals should be:

- Post-ReLU activation scale range.
- Weight scale range.
- Number and names of observed activation sites.
- Activation MSE.
- Logit MSE.
- Accuracy drop against recomputed FP32 accuracy.

The paper should describe this baseline as simulated PTQ/fake quantization and
must not claim real INT4 hardware acceleration.
