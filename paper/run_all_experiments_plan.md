# Plan: `src/experiments/run_all_experiments.py`

## Second Review

The original plan is directionally correct: it keeps the required method order,
uses the stable main-table columns requested by the paper workflow, and does not
touch quantization implementations.

The plan needs these corrections before implementation:

1. The source script, config file, and result CSV for each method must be named
   explicitly so the summary script does not rely on hidden assumptions.
2. The FP32 result CSV does not contain `accuracy_drop`, `activation_mse`, or
   `logit_mse`; those fields must be filled with numeric zeros in the main
   table.
3. Quantized result CSVs already contain `fp32_top1_accuracy` and
   `accuracy_drop`. The summary script should preserve each method's recorded
   `accuracy_drop` and warn if its recorded FP32 reference differs from the
   FP32 row.
4. Full result CSVs for `INT8-MinMax` and `INT4-P99.9` may be missing. The
   script needs a clear default policy: read existing full result CSVs first,
   then run only missing methods.
5. Smoke result CSVs must not be used for the paper main table unless a future
   user explicitly adds a separate smoke-output mode. The required outputs are
   the full main results.

## Objective

Implement `src/experiments/run_all_experiments.py` to collect the main results
for all required experiments and write a paper-ready table in both CSV and
Markdown formats.

Required outputs:

```text
outputs/results/main_results.csv
outputs/results/main_results.md
```

## Required Method Order

The summary script must process and emit methods in this exact order:

1. `FP32`
2. `INT8-MinMax`
3. `INT4-MinMax`
4. `INT4-P99.9`
5. `INT4-MSE-Selected`

## Method Source Mapping

| method | script/module | config | expected full result CSV |
|---|---|---|---|
| `FP32` | `src.experiments.train_fp32` | `configs/fp32_cifar10.yaml` | `outputs/results/fp32_result.csv` |
| `INT8-MinMax` | `src.experiments.run_int8_minmax_ptq` | `configs/int8_minmax_cifar10.yaml` | `outputs/results/int8_minmax_result.csv` |
| `INT4-MinMax` | `src.experiments.run_int4_minmax_ptq` | `configs/int4_minmax_cifar10.yaml` | `outputs/results/int4_minmax_result.csv` |
| `INT4-P99.9` | `src.experiments.run_int4_p999_ptq` | `configs/int4_p999_cifar10.yaml` | `outputs/results/int4_p999_result.csv` |
| `INT4-MSE-Selected` | `src.experiments.run_int4_mse_selected_ptq` | `configs/int4_mse_selected_cifar10.yaml` | `outputs/results/int4_mse_selected_result.csv` |

The implementation should read each result path from the config file when
possible, while keeping the table above as the expected default mapping.

## Files To Modify During Implementation

1. Add `src/experiments/run_all_experiments.py`
   - Orchestrates reading or running the five required methods.
   - Normalizes single-method result CSVs into one stable main table.
   - Writes `outputs/results/main_results.csv`.
   - Writes `outputs/results/main_results.md`.

2. Optionally add a focused smoke/unit test such as
   `tests/test_run_all_experiments.py`
   - Tests CSV normalization and Markdown generation without running CIFAR-10
     experiments.
   - This is optional but recommended if the implementation adds helper
     functions that can be tested cheaply.

## Files And Areas Not To Modify

The implementation must not change:

1. Any quantization method implementation in `src/quant/`
2. Existing FP32 training logic in `src/experiments/train_fp32.py`
3. Existing INT8 or INT4 single-method experiment logic
4. Model definitions in `src/models/`
5. Calibration, threshold search, fake quantization, or evaluation formulas
6. Existing result CSVs, except by explicitly running the corresponding
   existing experiment script when a required full result CSV is missing

`run_all_experiments.py` is an orchestration and summarization script only.

## Result Handling Strategy

Default behavior should be `auto`:

1. Load the method's YAML config.
2. Resolve the configured full result CSV path.
3. If the full result CSV already exists, read it.
4. If the full result CSV is missing, run the existing method script using its
   existing `run(...)` entry point and config.
5. Read the generated full result CSV.
6. Normalize exactly one row into the main table schema.

The script should write the final main table only after all five rows have been
collected successfully. If a required method fails or still does not produce its
full result CSV, the script should stop with a clear error instead of writing a
partial paper table.

## Main Table Schema

The CSV header must be exactly:

```text
method,weight_bits,activation_bits,top1_accuracy,accuracy_drop,activation_mse,logit_mse
```

Column meanings:

| column | meaning |
|---|---|
| `method` | Stable method label from the required method order |
| `weight_bits` | Effective simulated weight bit-width |
| `activation_bits` | Effective simulated activation bit-width |
| `top1_accuracy` | Top-1 accuracy in percentage points, using the existing result CSV value |
| `accuracy_drop` | Accuracy drop in percentage points relative to the FP32 reference used by that method |
| `activation_mse` | Activation reconstruction MSE from the existing result CSV |
| `logit_mse` | Logit MSE from the existing result CSV |

## Bit-Width Mapping

The bit-width values are fixed by method:

| method | weight_bits | activation_bits |
|---|---:|---:|
| `FP32` | 32 | 32 |
| `INT8-MinMax` | 8 | 8 |
| `INT4-MinMax` | 4 | 4 |
| `INT4-P99.9` | 4 | 4 |
| `INT4-MSE-Selected` | 4 | 4 |

## Metric Normalization Rules

For the FP32 row:

| field | value |
|---|---:|
| `top1_accuracy` | `top1_accuracy` from `fp32_result.csv` |
| `accuracy_drop` | `0.0000` |
| `activation_mse` | `0.00000000` |
| `logit_mse` | `0.00000000` |

For quantized rows:

1. `top1_accuracy` comes from the method result CSV.
2. `accuracy_drop` comes from the method result CSV.
3. `activation_mse` comes from the method result CSV.
4. `logit_mse` comes from the method result CSV.
5. All metric values must be parseable finite numbers.

The summary script should not recompute quantization metrics and should not
change the method-specific interpretation already recorded by the existing
single-method scripts.

## FP32 Reference Consistency

The summary script should compare:

1. `top1_accuracy` from the FP32 row
2. `fp32_top1_accuracy` from each quantized result CSV, when present

If these values differ beyond a small tolerance such as `1e-4`, the script
should print a warning that names the method and both values. This avoids
silently hiding a mismatch between the standalone FP32 result and the FP32
checkpoint evaluation used inside a PTQ script.

The main table should still preserve each quantized row's recorded
`accuracy_drop`, because that value was computed by the corresponding existing
experiment script against its own loaded FP32 checkpoint.

## Markdown Output

`outputs/results/main_results.md` must be generated from the same normalized
rows as the CSV output.

The Markdown table should:

1. Use the same column order as the CSV.
2. Use the same five method rows in the required order.
3. Preserve the same metric values as the CSV.
4. Be ready to paste into the paper.

## CLI Shape

The minimal command should be:

```text
python -m src.experiments.run_all_experiments
```

Recommended optional arguments:

| argument | purpose |
|---|---|
| `--mode auto` | Default. Read existing full CSVs and run missing methods. |
| `--mode read-only` | Read existing full CSVs only and fail if any are missing. |
| `--mode run-all` | Re-run all five existing method scripts before summarizing. |
| `--output-csv outputs/results/main_results.csv` | Override the CSV output path. |
| `--output-md outputs/results/main_results.md` | Override the Markdown output path. |
| `--strict-fp32-reference` | Treat FP32 reference mismatches as errors instead of warnings. |

These arguments keep the script reproducible without changing the underlying
experiment methods.

## Verification Plan

After implementation, verify with:

```text
python -m src.experiments.run_all_experiments --mode auto
```

Then check:

1. `outputs/results/main_results.csv` exists.
2. `outputs/results/main_results.md` exists.
3. The CSV header exactly matches:

```text
method,weight_bits,activation_bits,top1_accuracy,accuracy_drop,activation_mse,logit_mse
```

4. The method order is exactly:

```text
FP32
INT8-MinMax
INT4-MinMax
INT4-P99.9
INT4-MSE-Selected
```

5. The Markdown table contains the same five methods and the same metric values
   as the CSV.
6. No files under `src/quant/` were modified.

If a lightweight test is added, also run:

```text
python -m pytest tests/test_run_all_experiments.py
```

## Acceptance Criteria

The task is complete when both files exist:

```text
outputs/results/main_results.csv
outputs/results/main_results.md
```

and the generated main table uses exactly the required stable columns without
changing any quantization method implementation.
