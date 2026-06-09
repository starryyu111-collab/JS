# INT4-P99.9 CUDA Device And Quantile Fix Plan

## 1. Review Summary

This plan separates three issues that were mixed together during the previous
run:

1. The old MX330 CUDA compatibility issue.
   - The current `pytorch_env` uses a PyTorch CUDA build that supports MX330
     (`sm_61`).
   - Therefore `device: auto` should resolve to `cuda` in the current
     environment.

2. The INT4-P99.9 full-run failure.
   - The failure is not caused by CUDA fallback.
   - The failure occurs during P99.9 calibration at:

   ```python
   torch.quantile(values, FIXED_ACTIVATION_QUANTILE)
   ```

   - In the full run, some collected activation tensors are too large for
     PyTorch's `torch.quantile` implementation and trigger:

   ```text
   RuntimeError: quantile() input tensor is too large
   ```

3. The main-table orchestration issue.
   - `run_all_experiments` can only finish after the missing full
     `INT4-P99.9` result CSV is generated.
   - Smoke results must not be used for the paper main table.

The corrected execution order is:

```text
fix device resolver -> fix P99.9 quantile path -> regenerate missing full result -> generate main table
```

## 2. Step 1: Fix CUDA Device Resolution

### Intended Changes

Modify:

```text
src/experiments/run_int4_minmax_ptq.py
tests/test_int4_minmax.py
```

`run_int4_p999_ptq.py` and `run_int4_mse_selected_ptq.py` already reuse the
device resolver from `run_int4_minmax_ptq.py`, so they should not be changed
directly unless a later review finds a concrete need.

### Required Behavior

For `device: auto`:

- Return `cpu` when CUDA is unavailable.
- Return `cuda` when CUDA is available and the runtime probe succeeds.
- In the current MX330 + supported PyTorch CUDA build environment, return
  `cuda`.

For explicit `--device cuda`:

- Raise a clear error if CUDA is unavailable.
- Raise a clear error if CUDA is visible but the runtime cannot execute a small
  CUDA tensor operation.
- Do not silently fall back to CPU.

### Must Not Change

- Do not modify `src/quant/`.
- Do not modify P99.9 quantile logic in this step.
- Do not modify configs.
- Do not modify result CSV files.

### Verification

Run:

```bash
D:\ProgramData\anaconda3\envs\pytorch_env\python.exe -c "from src.experiments.run_int4_minmax_ptq import resolve_device; print(resolve_device('auto'))"
```

Expected output:

```text
cuda
```

Run:

```bash
D:\ProgramData\anaconda3\envs\pytorch_env\python.exe -m pytest tests/test_int4_minmax.py
```

Run P99.9 smoke:

```bash
D:\ProgramData\anaconda3\envs\pytorch_env\python.exe -m src.experiments.run_int4_p999_ptq --max-calibration-batches 1 --max-test-batches 1
```

Check the log/output shows:

```text
device=cuda
smoke=True
```

Review this step before moving to Step 2.

## 3. Step 2: Fix INT4-P99.9 Large-Tensor Quantile

### Intended Changes

Modify:

```text
src/quant/int4_p999.py
tests/test_int4_p999.py
```

This step changes quantization logic and must be reviewed separately from the
device fix.

### Required Behavior

Keep the P99.9 baseline definition unchanged:

```text
alpha_l = empirical_quantile(activation_l, 0.999)
```

The implementation must:

- Keep `calibration_size=1024` for the main run.
- Keep per-observed-ReLU-module element-wise P99.9 calibration.
- Keep small tensors on the normal `torch.quantile` path.
- Use an equivalent deterministic quantile path when `torch.quantile` cannot
  handle the input tensor size.
- Avoid approximate sampling unless explicitly approved later.
- Avoid MSE threshold selection.
- Avoid candidate percentile search.
- Avoid changing INT4-MinMax or INT4-MSE-Selected behavior.

### Must Not Change

- Do not reduce calibration size to make the run pass.
- Do not use smoke results for the paper table.
- Do not alter the method label or CSV metric definitions.

### Verification

Run:

```bash
D:\ProgramData\anaconda3\envs\pytorch_env\python.exe -m pytest tests/test_int4_p999.py
```

Then run the full P99.9 experiment:

```bash
D:\ProgramData\anaconda3\envs\pytorch_env\python.exe -m src.experiments.run_int4_p999_ptq
```

Acceptance output:

```text
outputs/results/int4_p999_result.csv
```

The CSV row must satisfy:

```text
method = INT4-P99.9
is_smoke = false
test_size = 10000
evaluated_test_size = 10000
threshold_search = false
mse_selected = false
activation_percentile = 99.9
activation_quantile = 0.999
```

Review this step before moving to Step 3.

## 4. Step 3: Restore Main Table Generation

### Intended Changes

Inspect and, only if needed, modify:

```text
src/experiments/run_all_experiments.py
tests/test_run_all_experiments.py
```

### Required Behavior

The main-table script must:

- Read existing full result CSVs first.
- Run only missing methods in `auto` mode.
- Never use smoke CSVs for the paper main table.
- Write the final main table only after all five method rows are available.
- Preserve each quantized method's recorded `accuracy_drop`.
- Warn, but do not recompute quantized rows, when a quantized CSV's
  `fp32_top1_accuracy` differs from the FP32 result row.

### Verification

Run:

```bash
D:\ProgramData\anaconda3\envs\pytorch_env\python.exe -m pytest tests/test_run_all_experiments.py
```

Run:

```bash
D:\ProgramData\anaconda3\envs\pytorch_env\python.exe -m src.experiments.run_all_experiments --mode auto
```

Acceptance outputs:

```text
outputs/results/main_results.csv
outputs/results/main_results.md
```

The method order must be:

```text
FP32
INT8-MinMax
INT4-MinMax
INT4-P99.9
INT4-MSE-Selected
```

The main table columns must be exactly:

```text
method,weight_bits,activation_bits,top1_accuracy,accuracy_drop,activation_mse,logit_mse
```

## 5. Paper Impact

If implemented as above, the paper method definition does not change.

The paper can continue describing the baseline as:

```text
INT4-P99.9 uses a fixed 99.9 percentile clipping rule for each observed post-ReLU activation module.
```

If implementation details are mentioned for reproducibility, use wording such
as:

```text
Large activation distributions were handled with an equivalent deterministic quantile computation to avoid framework input-size limits.
```

Do not use the following for the paper main table:

- Smoke results.
- Results generated with reduced `calibration_size`.
- Approximate-sampling quantile results unless the approximation is documented.
- MSE-Selected results as a substitute for the P99.9 baseline.

## 6. Assumptions And Gates

- The active experiment environment is:

```text
D:\ProgramData\anaconda3\envs\pytorch_env\python.exe
```

- The current PyTorch CUDA build supports MX330 (`sm_61`).
- Each step should be reviewed before moving to the next one.
- Result CSV files should only be changed by explicitly running the
  corresponding experiment.
- This document is a plan file only; code implementation should be done in
  separate, reviewed steps.
