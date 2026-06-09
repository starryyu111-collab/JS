# INT4-MSE-Selected Activation Clipping Proposed Method Plan

## Review Findings

The revised plan should keep the proposed method clearly separated from the
existing INT4-MinMax and INT4-P99.9 baselines. The current repository already
contains reusable INT4 fake-quantization logic and experiment utilities, so the
implementation should be a controlled extension instead of a broad refactor.

1. The proposed method needs its own quantization module.
   Core threshold-search logic should live in:

   ```text
   src/quant/clipping_search.py
   ```

   This file should implement layer-wise candidate evaluation and selected
   activation qparams. It should not write result files, create figures, or run
   CIFAR-10 experiments.

2. Experiment orchestration belongs in `src/experiments/`.
   CSV writing, figure generation, checkpoint loading, CIFAR-10 loaders,
   evaluation metrics, and output paths should be handled by a separate
   experiment entry point:

   ```text
   src/experiments/run_int4_mse_selected_ptq.py
   ```

3. The method must not be mixed with the fixed percentile baseline.
   `src/quant/int4_p999.py` should remain the INT4-P99.9 fixed clipping
   baseline. The proposed method may reuse generic INT4 utilities from
   `src/quant/int4_minmax.py`, but it should not reuse fixed-P99.9 calibration
   functions.

4. Result artifacts are part of the acceptance criteria.
   The implementation must generate:

   ```text
   outputs/results/int4_mse_selected_result.csv
   outputs/results/mse_selected_thresholds.csv
   outputs/figures/layerwise_mse.png
   ```

5. The name should stay conservative and precise.
   Use `INT4-MSE-Selected` or `Layer-wise MSE-Selected Activation Clipping`.
   The paper and code should describe the threshold as selected from a fixed
   candidate percentile set, not as an unrestricted global best threshold.

6. The existing ReLU site policy should remain consistent.
   Current INT4-MinMax and INT4-P99.9 logic treats each named `nn.ReLU` module
   as one post-ReLU activation site. In CIFAR ResNet blocks, the same ReLU
   module can be called twice, so this is module-level post-ReLU clipping rather
   than call-site-specific clipping. The proposed method should keep this policy
   unless a separate model rewrite is explicitly approved.

## Revised Objective

Implement the proposed method:

```text
Layer-wise MSE-Selected Activation Clipping for INT4 PTQ
```

For each observed post-ReLU activation layer, use the calibration set to select
one clipping threshold from the predefined candidate percentile set:

```text
C = {P99.0, P99.5, P99.9, P99.95, P100.0}
```

For each layer and each candidate percentile:

1. Collect FP32 post-ReLU activation values from the calibration set.
2. Convert the candidate percentile into a clipping threshold `alpha`.
3. Apply clip, UINT4 fake quantization, and dequantization.
4. Compute activation reconstruction MSE against the original FP32 activation.
5. Select the candidate with the smallest calibration reconstruction MSE.
6. Use the selected `alpha` for that layer's INT4 activation PTQ.

The final experiment must output:

```text
top1_accuracy
accuracy_drop
activation_mse
logit_mse
```

## Scope

### Planned File Changes

Add:

```text
src/quant/clipping_search.py
src/experiments/run_int4_mse_selected_ptq.py
configs/int4_mse_selected_cifar10.yaml
tests/test_clipping_search.py
tests/test_int4_mse_selected.py
paper/int4_mse_selected_ptq_plan.md
```

Modify:

```text
src/utils/csv_io.py
```

The `csv_io.py` change should be limited to adding a reusable multi-row CSV
writer for the layer-wise threshold table. Existing `write_single_row_csv`
behavior should remain unchanged.

### Files That Should Not Change

Do not modify the behavior of:

```text
src/quant/int4_minmax.py
src/quant/int4_p999.py
src/quant/int8_minmax.py
src/experiments/run_int4_minmax_ptq.py
src/experiments/run_int4_p999_ptq.py
src/experiments/run_int8_minmax_ptq.py
src/experiments/train_fp32.py
configs/int4_minmax_cifar10.yaml
configs/int4_p999_cifar10.yaml
configs/int8_minmax_cifar10.yaml
configs/fp32_cifar10.yaml
```

Existing baseline tests should continue to pass.

## Relationship To Baselines

| Setting | INT4-MinMax | INT4-P99.9 | INT4-MSE-Selected |
|---|---|---|---|
| Weight dtype | INT4 | INT4 | INT4 |
| Weight range | `[-7, 7]` | `[-7, 7]` | `[-7, 7]` |
| Weight granularity | per-channel | per-channel | per-channel |
| Activation dtype | UINT4 | UINT4 | UINT4 |
| Activation range | `[0, 15]` | `[0, 15]` | `[0, 15]` |
| Activation target | post-ReLU | post-ReLU | post-ReLU |
| Activation clip interval | `[0, max_l]` | `[0, P99.9_l]` | `[0, selected_alpha_l]` |
| Threshold source | calibration max | fixed percentile | MSE-selected candidate |
| Candidate search | no | no | yes |
| Selection metric | none | none | activation reconstruction MSE |

The proposed method should be implemented as its own experiment, not as a mode
inside the fixed percentile baseline.

## Quantization Specification

### Candidate Percentiles

Use one named constant for the candidate set:

```text
MSE_SELECTED_PERCENTILES = (99.0, 99.5, 99.9, 99.95, 100.0)
```

The corresponding quantiles are:

```text
0.9900
0.9950
0.9990
0.9995
1.0000
```

`P100.0` should be treated as the layer maximum. This keeps the max baseline
inside the candidate set.

### Weight Quantization

Use the same weight quantization as INT4-MinMax and INT4-P99.9:

- Target modules: `torch.nn.Conv2d` and `torch.nn.Linear`.
- Per-channel symmetric fake quantization.
- Channel axis: output channel dimension, `dim=0`.
- Integer range: `[-7, 7]`.
- Zero-point: `0`.
- Bias remains FP32.

Weight fake quantization is applied only to a deep copy of the loaded FP32
model.

### Activation Quantization

Target tensors:

- Outputs of named post-ReLU modules.

Method:

- Per-tensor affine fake quantization.
- Integer dtype: UINT4.
- Integer range: `[0, 15]`.
- Real clipping interval: `[0, selected_alpha_l]`.
- Zero-point: `0`.

For layer `l` and candidate percentile `p`:

```text
alpha_l,p = percentile(x_l, p)
scale_l,p = alpha_l,p / 15
x_hat_l,p = dequantize(quantize(clip(x_l, 0, alpha_l,p)))
mse_l,p = mean((x_l - x_hat_l,p)^2)
```

Then select:

```text
p_l = candidate percentile with minimum mse_l,p
selected_alpha_l = alpha_l,p_l
```

Tie-breaking should be deterministic. If multiple candidates have the same MSE,
choose the first one in `MSE_SELECTED_PERCENTILES`.

If `selected_alpha_l <= 0` or the scale is not finite, use the same finite
fallback policy as INT4-MinMax:

```text
clip_max_l = 0.0
scale_l = 1.0
zero_point_l = 0
```

## Calibration Policy

Calibration source:

- CIFAR-10 training split: `train=True`.

Calibration target:

- Post-ReLU activation outputs from the FP32 model or the unmodified INT4 copy
  before activation wrappers are attached.

Collection policy:

- Collect all activation elements for each named post-ReLU module across the
  calibration subset.
- Compute each percentile from the concatenated per-layer distribution.
- Compute each candidate MSE over the same per-layer calibration distribution.
- Do not average per-batch percentiles.
- Do not compute one global activation distribution shared across layers.
- Do not use the CIFAR-10 test split for threshold selection.

Exact collection is acceptable for the default calibration size. If memory
becomes a concern later, approximate percentile estimation should be a separate,
documented change with tests against an exact small case.

## `src/quant/clipping_search.py` Design

This module should expose small, testable APIs. Suggested data structures:

```text
CandidateMSE
LayerClippingSearchResult
MSESelectedActivationCalibrationResult
```

Suggested public function:

```text
calibrate_post_relu_activation_mse_selected(
    model,
    loader,
    device,
    candidate_percentiles=MSE_SELECTED_PERCENTILES,
    qmin=0,
    qmax=15,
    max_batches=None,
)
```

Return values should include:

- `qparams_by_name`
- `observed_site_names`
- `calibration_num_batches`
- `layer_results`
- `min_activation_scale`
- `max_activation_scale`
- `min_activation_zero_point`
- `max_activation_zero_point`

Each layer result should include:

- layer name
- selected percentile
- selected alpha
- selected MSE
- all candidate percentiles
- all candidate alphas
- all candidate MSE values
- number of activation elements used for that layer

The module may reuse these generic INT4 utilities from `src/quant/int4_minmax.py`:

```text
make_post_relu_activation_qparams
fake_quantize_post_relu_activation
iter_named_post_relu_modules
validate_integer_range
PostReluActivationQuantizationParams
ActivationCalibrationResult-compatible fields
```

It should not import from `src/experiments/` and should not write files.

## Experiment Script Design

Add:

```text
src/experiments/run_int4_mse_selected_ptq.py
```

The experiment should run in this order:

1. Load `configs/int4_mse_selected_cifar10.yaml`.
2. Apply CLI overrides.
3. Set the experiment seed.
4. Resolve the device.
5. Build CIFAR-10 calibration and test loaders.
6. Build the configured FP32 model.
7. Load the FP32 checkpoint.
8. Recompute FP32 top-1 accuracy on CIFAR-10 `train=False`.
9. Deep-copy the FP32 model for INT4 simulation.
10. Run MSE-selected activation threshold calibration on the calibration loader.
11. Write the layer-wise threshold CSV.
12. Generate the layer-wise MSE figure.
13. Apply INT4 per-channel weight fake quantization to the INT4 copy.
14. Attach post-ReLU activation fake quantization using selected qparams.
15. Evaluate INT4-MSE-Selected top-1 accuracy on CIFAR-10 `train=False`.
16. Compute `activation_mse` on CIFAR-10 `train=False`.
17. Compute `logit_mse` on CIFAR-10 `train=False`.
18. Write the final result CSV.

No step may call:

```text
model.train()
optimizer.step()
loss.backward()
save_checkpoint()
```

BatchNorm statistics must not be updated during calibration or evaluation.

## Config Draft

Add:

```text
configs/int4_mse_selected_cifar10.yaml
```

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
  method: INT4-MSE-Selected
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
    clip_method: mse_selected
    candidate_percentiles: [99.0, 99.5, 99.9, 99.95, 100.0]
    clip_max_source: calibration_mse_selected_percentile

paths:
  checkpoint_path: checkpoints/fp32_best.pt
  result_path: outputs/results/int4_mse_selected_result.csv
  threshold_result_path: outputs/results/mse_selected_thresholds.csv
  figure_path: outputs/figures/layerwise_mse.png
  log_path: outputs/logs/int4_mse_selected_ptq.log

smoke:
  max_calibration_batches:
  max_test_batches:
```

## Output Files

### Final Result CSV

Path:

```text
outputs/results/int4_mse_selected_result.csv
```

Required metric columns:

```text
top1_accuracy
accuracy_drop
activation_mse
logit_mse
```

Recommended common columns:

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
threshold_result_path
figure_path
log_path
device
batch_size
observed_activation_sites
```

Recommended method-specific columns:

```text
activation_clip_method
activation_clip_source
activation_clip_min
activation_granularity
activation_site_type
threshold_search
mse_selected
candidate_percentiles
min_selected_activation_alpha
max_selected_activation_alpha
min_selected_layer_mse
max_selected_layer_mse
mean_selected_layer_mse
weight_granularity
weight_symmetry
```

Recommended values:

```text
method = INT4-MSE-Selected
activation_clip_method = mse_selected
activation_clip_source = calibration_mse_selected_percentile
threshold_search = true
mse_selected = true
candidate_percentiles = 99.0;99.5;99.9;99.95;100.0
```

### Threshold Search CSV

Path:

```text
outputs/results/mse_selected_thresholds.csv
```

One row per observed post-ReLU activation layer.

Recommended columns:

```text
layer_name
layer_index
num_activation_elements
selected_percentile
selected_alpha
selected_mse
p99_0_alpha
p99_0_mse
p99_5_alpha
p99_5_mse
p99_9_alpha
p99_9_mse
p99_95_alpha
p99_95_mse
p100_0_alpha
p100_0_mse
activation_qmin
activation_qmax
activation_zero_point
activation_scale
```

If all candidate alphas and MSEs are easier to store as semicolon-separated
diagnostic fields, the CSV may additionally include:

```text
candidate_percentiles
candidate_alphas
candidate_mses
```

However, the selected percentile, selected alpha, and candidate MSE values must
be directly visible in the CSV.

### Layer-wise MSE Figure

Path:

```text
outputs/figures/layerwise_mse.png
```

The figure should show selected calibration reconstruction MSE per layer. The
x-axis can use layer index with layer names rotated or summarized. The selected
percentile should be visible through labels, annotations, or a compact legend.

The figure should be generated from the same layer search results that are saved
to `mse_selected_thresholds.csv`.

## Metrics

Required output metrics:

```text
top1_accuracy
accuracy_drop
activation_mse
logit_mse
```

Definitions:

- `top1_accuracy`: INT4-MSE-Selected top-1 accuracy on CIFAR-10 test set,
  reported as a percentage in `[0, 100]`.
- `fp32_top1_accuracy`: FP32 checkpoint top-1 accuracy recomputed by the same
  script on the same test set.
- `int4_top1_accuracy`: same value as `top1_accuracy`, included for explicit
  method readability.
- `accuracy_drop`: `fp32_top1_accuracy - int4_top1_accuracy`.
- `activation_mse`: element-weighted local reconstruction MSE on CIFAR-10
  `train=False`, using the selected activation thresholds.
- `logit_mse`: element-weighted MSE between FP32 logits and the simulated
  INT4-MSE-Selected model logits on CIFAR-10 `train=False`.

Threshold search MSE is computed on the calibration set. Final reported
`activation_mse` and `logit_mse` are computed on the test set.

## Testing Plan

### Unit Tests For `src/quant/clipping_search.py`

Verify:

- Candidate set is exactly `(99.0, 99.5, 99.9, 99.95, 100.0)`.
- `P100.0` equals the layer maximum on a small exact tensor.
- Candidate alphas are layer-specific.
- Candidate MSE values are finite and nonnegative.
- The selected candidate has the smallest candidate MSE.
- Tie-breaking is deterministic.
- Selected qparams use UINT4 range `[0, 15]`.
- Selected qparams have zero-point `0`.
- Calibration processes at least one batch.
- No global alpha is shared across layers.

### Experiment Smoke Test

Use a tiny model and mocked dataloaders to verify:

- The experiment prints `top1_accuracy`.
- The experiment prints `accuracy_drop`.
- The experiment prints `activation_mse`.
- The experiment prints `logit_mse`.
- The final result CSV is written.
- The threshold CSV is written.
- The layer-wise MSE figure is written.
- Smoke result paths do not overwrite full result paths unless overridden.

### Baseline Regression Tests

Run existing baseline tests to ensure this method does not change them:

```text
tests/test_int4_minmax.py
tests/test_int4_p999.py
tests/test_int8_minmax.py
```

## Verification Commands

Suggested test command:

```bash
pytest tests/test_clipping_search.py tests/test_int4_mse_selected.py tests/test_int4_p999.py tests/test_int4_minmax.py
```

Suggested smoke command:

```bash
python -m src.experiments.run_int4_mse_selected_ptq --config configs/int4_mse_selected_cifar10.yaml --device cpu --max-calibration-batches 1 --max-test-batches 1
```

Expected smoke artifacts:

```text
outputs/results/int4_mse_selected_result_smoke.csv
outputs/results/mse_selected_thresholds_smoke.csv
outputs/figures/layerwise_mse_smoke.png
```

Suggested full run:

```bash
python -m src.experiments.run_int4_mse_selected_ptq --config configs/int4_mse_selected_cifar10.yaml
```

Expected full artifacts:

```text
outputs/results/int4_mse_selected_result.csv
outputs/results/mse_selected_thresholds.csv
outputs/figures/layerwise_mse.png
```

## Acceptance Checklist

Before considering the proposed method complete:

1. Main threshold-search code is in `src/quant/clipping_search.py`.
2. The fixed percentile baseline remains separate.
3. Candidate set is exactly `{P99.0, P99.5, P99.9, P99.95, P100.0}`.
4. Calibration uses CIFAR-10 `train=True`.
5. Accuracy and final MSE evaluation use CIFAR-10 `train=False`.
6. Every observed post-ReLU activation layer gets its own selected percentile.
7. Every observed post-ReLU activation layer gets its own selected alpha.
8. Candidate MSE values are saved for every layer.
9. The final result CSV includes `top1_accuracy`, `accuracy_drop`,
   `activation_mse`, and `logit_mse`.
10. `outputs/results/int4_mse_selected_result.csv` is generated for a full run.
11. `outputs/results/mse_selected_thresholds.csv` is generated for a full run.
12. `outputs/figures/layerwise_mse.png` is generated for a full run.
13. No QAT, retraining, fine-tuning, optimizer step, or checkpoint update
    occurs.
14. The method is described as selected from a predefined candidate percentile
    set.
15. Existing INT4-MinMax and INT4-P99.9 tests still pass.

## Anti-Requirements

The implementation must not:

- Mix this method into `src/quant/int4_p999.py`.
- Use one global alpha for all activation layers.
- Select thresholds from values outside the predefined candidate set.
- Use the CIFAR-10 test set for threshold selection.
- Change baseline result semantics.
- Claim real INT4 hardware acceleration.
- Claim training, fine-tuning, or QAT.
- Describe module-level reused ReLU observation as call-site-specific clipping.

## Implementation Notes

This method is expected to improve activation reconstruction behavior more
directly than a fixed percentile rule, but the paper should report the measured
accuracy and MSE results honestly. If accuracy does not improve much, the
analysis should focus on activation MSE, logit consistency, layer sensitivity,
and error accumulation.

The strongest paper-safe statement is:

```text
The clipping threshold is selected per layer by minimizing calibration
activation reconstruction MSE within a predefined percentile candidate set.
```
