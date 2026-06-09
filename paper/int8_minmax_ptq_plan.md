# INT8-MinMax PTQ Plan Review and Revised Implementation Plan

## Second Review Findings

The plan is suitable for implementing INT8-MinMax PTQ as a sanity check, but a
second review found a few details that should be made more explicit before any
code is written:

1. `activation_mse` must have a precise meaning. It should measure local
   activation fake-quant reconstruction error on FP32 activation tensors, not a
   vaguely matched hidden-state difference after cumulative INT8 errors.
2. The INT8 model should be created from a separate copy of the loaded FP32
   model. Weight fake quantization must not mutate the FP32 model used to
   recompute `fp32_top1_accuracy`.
3. Calibration and evaluation must run with `model.eval()` and `torch.no_grad()`.
   There should be no optimizer, backward pass, training loop, or BatchNorm
   statistics update.
4. Smoke-test runs should not silently overwrite the final full-evaluation CSV.
   If smoke output is written, it should either be clearly marked or go to a
   separate path.
5. Observer placement is acceptable only if calibration and fake quantization
   use the exact same named activation sites. Module names should be logged so
   this can be audited if INT8 accuracy collapses.
6. The failure gate before INT4 should require diagnostic values, not only
   conceptual review: qmin/qmax, scale range, zero-point range, observed module
   count, calibration source, test size, and checkpoint path.

This revised plan keeps the scope narrow: simulated PTQ only, no retraining, no
QAT, no INT4, no BatchNorm folding, no real hardware kernel, and no changes to
the FP32 baseline script.

## Objective

Implement INT8-MinMax post-training quantization as a sanity check for the
CIFAR-10 PTQ workflow.

The experiment must:

- Use the existing trained FP32 checkpoint.
- Avoid model retraining, fine-tuning, or QAT.
- Draw the calibration set from the CIFAR-10 training set.
- Evaluate classification accuracy on the complete CIFAR-10 test set.
- Use per-channel symmetric INT8 fake quantization for weights.
- Use per-tensor affine INT8 or UINT8 fake quantization for activations, with
  explicit bit ranges.
- Output `top1_accuracy`, `accuracy_drop`, `activation_mse`, and `logit_mse`.
- Save the final full-evaluation result to
  `outputs/results/int8_minmax_result.csv`.
- Leave the FP32 baseline script unchanged.

## Scope

### Planned File Changes

Future implementation should add:

```text
src/quant/int8_minmax.py
src/experiments/run_int8_minmax_ptq.py
configs/int8_minmax_cifar10.yaml
```

Optional, only if a lightweight smoke test is useful:

```text
tests/test_int8_minmax.py
```

### Files That Must Not Change

The implementation must not modify:

```text
src/experiments/train_fp32.py
outputs/results/fp32_result.csv
```

The INT8 script may reuse existing project utilities such as `build_model`,
`load_checkpoint`, `top1_accuracy`, and `write_single_row_csv`, but it must not
change the FP32 baseline script or retrain the FP32 model.

## Execution Protocol

The INT8 experiment should follow this order:

1. Load `configs/int8_minmax_cifar10.yaml`.
2. Set the experiment seed.
3. Resolve the device.
4. Build the FP32 model with `src.models.build_model`.
5. Load the checkpoint with `src.utils.checkpoint.load_checkpoint`.
6. Set the model to `eval()` and run all inference under `torch.no_grad()`.
7. Recompute FP32 accuracy on the full CIFAR-10 test set.
8. Deep-copy the loaded FP32 model to create the INT8 simulated model.
9. Calibrate activation min/max on a deterministic subset of CIFAR-10
   `train=True`.
10. Apply per-channel weight fake quantization to the INT8 model copy.
11. Attach activation fake quantization at the same named sites used during
   calibration.
12. Evaluate INT8 accuracy on the full CIFAR-10 test set.
13. Compute `activation_mse` and `logit_mse`.
14. Write the final CSV to `outputs/results/int8_minmax_result.csv`.

No step may call `model.train()`, create an optimizer, run `loss.backward()`, or
update checkpoint weights.

## Quantization Specification

### Weight Quantization

Target modules:

- `torch.nn.Conv2d`
- `torch.nn.Linear`

Method:

- Per-channel symmetric fake quantization.
- Channel axis: output channel dimension, `dim=0`.
- Integer dtype: signed INT8.
- Integer range: `[-127, 127]`.
- Zero-point: `0`.

The range intentionally excludes `-128` so the signed quantization range is
symmetric.

For each output channel `c`:

```text
qmin_w = -127
qmax_w = 127
max_abs_c = max(abs(weight_c))
scale_c = max_abs_c / qmax_w
```

If `max_abs_c == 0`, use a finite fallback scale such as `scale_c = 1.0`, so
the channel quantizes to all zeros.

Fake quantization:

```text
q_w = clamp(round(weight / scale_c), qmin_w, qmax_w)
dequant_w = q_w * scale_c
```

Expected broadcast scale shapes:

- `Conv2d`: `[out_channels, 1, 1, 1]`.
- `Linear`: `[out_features, 1]`.

Bias, if present, remains FP32 for this simulated PTQ sanity check.

Weight fake quantization must be applied only to the INT8 model copy. The FP32
model used for baseline accuracy and FP32 activations must remain untouched.

### Activation Quantization

Target tensors:

- Outputs of wrapped `Conv2d` and `Linear` modules.

Observer and fake-quant placement:

- During calibration, collect min/max from the output tensor of each target
  `Conv2d` and `Linear` module.
- During INT8 evaluation, apply activation fake quantization to the output
  tensor of the same named module.
- The collected observer site names and the fake-quant wrapper site names must
  match exactly.
- BatchNorm, ReLU, pooling, residual addition, flatten, and other non-target
  operations remain FP32 in this INT8 sanity-check implementation.

Default activation quantization:

- Method: per-tensor affine fake quantization.
- Integer dtype: `UINT8`.
- Integer range: `[0, 255]`.

Optional activation mode:

- Method: per-tensor affine fake quantization.
- Integer dtype: `INT8`.
- Integer range: `[-128, 127]`.
- Because this is affine INT8, zero-point may be nonzero and must be clamped to
  the configured signed range.

For each activation site:

```text
qmin_a, qmax_a = configured activation integer range
x_min = observed minimum over calibration batches
x_max = observed maximum over calibration batches
scale_a = (x_max - x_min) / (qmax_a - qmin_a)
zero_point_a = round(qmin_a - x_min / scale_a)
zero_point_a = clamp(zero_point_a, qmin_a, qmax_a)
```

If `x_min == x_max`, use a finite fallback scale such as `scale_a = 1.0` and a
valid zero-point in the configured range. This prevents NaN or Inf values during
calibration and fake quantization.

Fake quantization:

```text
q_x = clamp(round(x / scale_a + zero_point_a), qmin_a, qmax_a)
dequant_x = (q_x - zero_point_a) * scale_a
```

The bit ranges must be stored as constants or config values, not scattered as
magic numbers.

## Data Protocol

### Calibration Set

Source:

- `torchvision.datasets.CIFAR10(train=True)`.

Sampling:

- Deterministic random subset using the experiment seed.
- Default size: `1024`, configurable.
- The calibration subset must never draw from `CIFAR10(train=False)`.

Transform:

- Evaluation transform only:
  - `ToTensor`
  - CIFAR-10 normalization using the same mean/std constants as the FP32
    baseline.

No training-time augmentation:

- No `RandomCrop`.
- No `RandomHorizontalFlip`.

Purpose:

- Calibration is used only to collect activation min/max statistics.
- Calibration accuracy must not be reported as final model accuracy.

Audit fields or logs should include:

- `calibration_size`
- `calibration_seed`
- `calibration_source = CIFAR10 train=True`
- `calibration_num_batches`
- Optional first few sampled indices or an index checksum.

### Evaluation Set

Source:

- `torchvision.datasets.CIFAR10(train=False)`.

Transform:

- Same evaluation transform as calibration.

Requirement:

- Final reported metrics must be computed on the complete CIFAR-10 test set.
- For the final acceptance run:

```text
test_size = 10000
evaluated_test_size = 10000
is_smoke = false
```

Smoke-test limits may exist for debugging. Smoke output must not be confused
with the final result. Prefer either:

- Do not write `outputs/results/int8_minmax_result.csv` during smoke runs; or
- Write it with `is_smoke = true` and never use it as the accepted final result.

## Checkpoint Protocol

The INT8 experiment must load the same trained checkpoint used for the FP32
baseline. The default path should be:

```text
checkpoints/fp32_best.pt
```

The experiment should:

1. Build the model using the configured model name.
2. Load the checkpoint state dict into that model.
3. Verify that `checkpoint["model_name"]`, if present, matches the configured
   model name.
4. Log or write the checkpoint path.
5. Recompute FP32 test accuracy inside the INT8 script using the loaded
   checkpoint and full CIFAR-10 test set.
6. Compute `accuracy_drop` against this recomputed FP32 accuracy, not against a
   manually copied number from a previous CSV.

This keeps checkpoint identity auditable and avoids changing the FP32 baseline
script.

## Metrics

The final CSV must include these required fields:

```text
top1_accuracy
accuracy_drop
activation_mse
logit_mse
```

Metric definitions:

- `top1_accuracy`: INT8-MinMax top-1 accuracy on the full CIFAR-10 test set,
  reported as a percentage in `[0, 100]`.
- `fp32_top1_accuracy`: FP32 checkpoint top-1 accuracy recomputed on the full
  CIFAR-10 test set, reported as a percentage in `[0, 100]`.
- `int8_top1_accuracy`: same value as `top1_accuracy`, included for readability.
- `accuracy_drop`: `fp32_top1_accuracy - int8_top1_accuracy`, measured in
  percentage points.
- `activation_mse`: element-weighted mean squared reconstruction error between
  FP32 activation tensors at target sites and those same tensors after applying
  the calibrated activation fake-quant/dequant function. This should be measured
  on the full CIFAR-10 test set and uses the activation quantization parameters
  from calibration.
- `logit_mse`: element-weighted mean squared error between FP32 logits and INT8
  fake-quant model logits on the full CIFAR-10 test set.

`activation_mse` is a local fake-quant reconstruction metric. `logit_mse` is
the end-to-end model-output consistency metric. Keeping them separate makes the
sanity check easier to debug.

Recommended additional CSV fields:

```text
method
model
dataset
seed
checkpoint_path
checkpoint_model_name
calibration_size
calibration_source
calibration_num_batches
test_size
evaluated_test_size
fp32_top1_accuracy
int8_top1_accuracy
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
```

Required final result path:

```text
outputs/results/int8_minmax_result.csv
```

## Planned Module Responsibilities

### `src/quant/int8_minmax.py`

Responsibilities:

- Define INT8 and UINT8 quantization ranges.
- Implement per-channel symmetric weight fake quantization.
- Implement per-tensor affine activation fake quantization.
- Implement activation min/max observers.
- Provide calibration helpers for named target module outputs.
- Provide an INT8 fake-quant wrapper or conversion helper for `Conv2d` and
  `Linear`.
- Return activation site names and diagnostic quantization parameters.
- Provide utilities to compute local activation reconstruction MSE.

This module must not:

- Build CIFAR-10 dataloaders.
- Load checkpoints.
- Run complete experiments.
- Write result CSV files.
- Import from `src.experiments`.

### `src/experiments/run_int8_minmax_ptq.py`

Responsibilities:

- Load config and CLI overrides.
- Build calibration and test dataloaders.
- Build and load the FP32 checkpoint model.
- Recompute FP32 full-test accuracy.
- Deep-copy the loaded FP32 model for INT8 simulated PTQ.
- Run activation calibration on the calibration subset.
- Apply INT8-MinMax fake quantization to weights and activations.
- Evaluate INT8-MinMax on the full CIFAR-10 test set.
- Compute `top1_accuracy`, `accuracy_drop`, `activation_mse`, and `logit_mse`.
- Write the final result CSV.

This script must not:

- Train or fine-tune the model.
- Modify `src/experiments/train_fp32.py`.
- Write or overwrite `outputs/results/fp32_result.csv`.
- Treat smoke-test results as final full-test results.

### `configs/int8_minmax_cifar10.yaml`

Recommended contents:

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
  method: INT8-MinMax
  weight:
    dtype: int8
    qmin: -127
    qmax: 127
    granularity: per_channel
    symmetry: symmetric
    channel_axis: 0
  activation:
    dtype: uint8
    qmin: 0
    qmax: 255
    granularity: per_tensor
    symmetry: affine

paths:
  checkpoint_path: checkpoints/fp32_best.pt
  result_path: outputs/results/int8_minmax_result.csv
  log_path: outputs/logs/int8_minmax_ptq.log

smoke:
  max_calibration_batches:
  max_test_batches:
```

## Verification Plan

### Tensor-Level Smoke Check

Verify:

- Weight fake quantization preserves the original tensor shape.
- Per-channel weight scales have the expected broadcast shape.
- Weight integer values are clamped to `[-127, 127]`.
- Activation fake quantization uses the configured range, default `[0, 255]`.
- Activation scales are finite and positive.
- Activation zero-points are finite integers clamped to the configured range.
- Fake-quant/dequant outputs are finite.
- MSE values are finite and non-negative.

### Observer Placement Check

Verify:

- The number of observed activation sites equals the number of wrapped target
  `Conv2d` and `Linear` modules.
- The ordered list of observed site names matches the ordered list of
  fake-quantized site names.
- Calibration observers are attached to target module outputs.
- Fake quantization is applied to the same target module outputs.
- No test-set batch is used to update activation min/max statistics.

### Experiment Smoke Check

Run a small debugging pass, for example:

```bash
python -m src.experiments.run_int8_minmax_ptq --config configs/int8_minmax_cifar10.yaml --max-calibration-batches 1 --max-test-batches 1
```

Verify:

- The FP32 checkpoint is loaded.
- Both FP32 and INT8 models are in eval mode.
- No optimizer, backward pass, or training loop runs.
- Calibration uses CIFAR-10 `train=True`.
- Evaluation uses CIFAR-10 `train=False`.
- Required metrics are produced.
- Smoke output is clearly marked with `is_smoke = true` or written separately.

### Full Evaluation

Run:

```bash
python -m src.experiments.run_int8_minmax_ptq --config configs/int8_minmax_cifar10.yaml
```

Verify:

- The full CIFAR-10 test set is evaluated.
- `outputs/results/int8_minmax_result.csv` exists.
- The CSV includes `top1_accuracy`, `accuracy_drop`, `activation_mse`, and
  `logit_mse`.
- `evaluated_test_size == test_size == 10000`.
- `accuracy_drop == fp32_top1_accuracy - int8_top1_accuracy` up to formatting
  precision.
- `checkpoint_path` matches the FP32 baseline checkpoint path.
- `is_smoke = false`.

## Failure Review Gate Before INT4

If INT8 accuracy is much worse than FP32, do not continue to INT4 yet. First
review and fix the INT8 sanity-check pipeline.

Required review checklist:

1. Scale correctness:
   - Weight scale uses per-output-channel `max_abs / 127`.
   - Activation scale uses `(x_max - x_min) / (qmax - qmin)`.
   - Degenerate min/max cases use finite fallback scales.
   - Logged min/max scale values are finite and positive.
2. Zero-point correctness:
   - Symmetric weight zero-point is exactly `0`.
   - Activation zero-point uses `round(qmin - x_min / scale)`.
   - Activation zero-point is clamped to the configured bit range.
   - Logged zero-point min/max values fall within the configured qmin/qmax.
3. Observer placement:
   - Calibration observers are attached to the intended `Conv2d` and `Linear`
     outputs.
   - Fake quantization is applied at the same named sites.
   - The number and names of observed sites match the target module list.
4. Dataset separation:
   - Calibration samples come only from CIFAR-10 `train=True`.
   - Final evaluation uses only CIFAR-10 `train=False`.
   - No test sample is used for calibration or threshold selection.
   - `evaluated_test_size` is `10000` for the final result.
5. Checkpoint consistency:
   - The INT8 script loads the same checkpoint path as the FP32 baseline.
   - The checkpoint `model_name`, if present, matches the configured model.
   - FP32 accuracy is recomputed in the INT8 script from the same loaded model.
   - The FP32 model object is not mutated by INT8 weight fake quantization.

Only after this checklist passes should INT4-MinMax or clipping experiments be
started.

