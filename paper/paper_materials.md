# Paper Materials

> Scope note: this file is a structured material sheet for drafting the course paper. It is **not** a complete paper draft.
>
> Generation rule followed: only information that can be read from the current GitHub repository is used. Any requested result or artifact that could not be found is marked as **MISSING**.

## Source status checked

| Requested source | Status | Notes |
|---|---|---|
| `README.md` | FOUND | Project describes CIFAR-10 INT4 PTQ with layer-wise MSE-selected activation clipping. |
| `Claude.md` | FOUND | Provides paper title, scope, required methods, workflow, and claim restrictions. |
| `configs/cifar10.yaml` | MISSING | The exact requested config file was not found. Method-specific config files were found and used instead: `configs/fp32_cifar10.yaml`, `configs/int8_minmax_cifar10.yaml`, `configs/int4_minmax_cifar10.yaml`, `configs/int4_p999_cifar10.yaml`, and `configs/int4_mse_selected_cifar10.yaml`. |
| `src/run_all_experiments.py` | MISSING | The exact requested path was not found. The actual experiment orchestrator found in the repository is `src/experiments/run_all_experiments.py`. |
| `src/quant/clipping_search.py` | FOUND | Implements layer-wise MSE-selected post-ReLU activation clipping. |
| `outputs/tables/` | MISSING | No committed table file was found at checked paths such as `outputs/tables/main_results.md` or `outputs/tables/main_results.csv`. |
| `outputs/results/` | MISSING | Checked common result files referenced by the orchestrator/configs, including `main_results.md`, `main_results.csv`, and per-method CSVs; no committed result files were found. |
| `outputs/figures/` | MISSING | Checked expected figure files `accuracy_drop.png`, `error_metrics.png`, and `layerwise_mse.png`; no committed figure files were found. |

## 1. Paper title

**Research on Low-Bit (INT4) Post-Training Quantization for Lightweight CNNs Based on Activation Outlier Suppression**

## 2. Research problem

Low-bit post-training quantization can be sensitive to activation outliers. In INT4 PTQ, MinMax calibration may allocate too much quantization range to rare large activation values, which can increase activation reconstruction error and degrade classification accuracy.

This project studies whether calibration-time activation clipping can make INT4 PTQ more stable than ordinary INT4 MinMax PTQ on a CIFAR-10 CNN classification task.

## 3. Method name

**Layer-wise MSE-Selected Activation Clipping**

Implementation/result-table method name: **INT4-MSE-Selected**.

Method definition from the inspected code:

1. Collect post-ReLU activations on the calibration set.
2. For each observed activation layer/site, evaluate the candidate percentile set `{99.0, 99.5, 99.9, 99.95, 100.0}`.
3. Convert each candidate percentile into a clipping threshold.
4. Fake-quantize and dequantize the collected activation values under each threshold.
5. Compute activation reconstruction MSE between the original FP32 activations and the dequantized activations.
6. Select the candidate threshold with the lowest MSE for that layer.

Important wording constraint: the method is **MSE-selected within a predefined candidate percentile set**, not globally optimal clipping.

## 4. Dataset and model

- Dataset: **CIFAR-10**
- Model: **resnet18_cifar**
- Number of classes: **10**
- FP32 training config:
  - Epochs: **100**
  - Batch size: **128**
  - Seed: **42**
  - Optimizer: **SGD**
  - Learning rate: **0.1**
  - Momentum: **0.9**
  - Weight decay: **0.0005**
  - Scheduler: **MultiStepLR**, milestones `[50, 75]`, gamma `0.1`
- Validation size in FP32 config: **5000**
- Full test size required by the orchestrator: **10000**

## 5. Calibration set and evaluation set

### Calibration set

- Source: **CIFAR-10 training set** according to the project workflow.
- Calibration size in quantization configs: **1024 samples**.
- Calibration seed: **42** in quantization experiment configs.
- Calibration batch count from actual outputs: **MISSING**.
- Calibration index list/checksum from actual outputs: **MISSING**.

### Evaluation set

- Evaluation target: **full CIFAR-10 test set**.
- Required full test size in `src/experiments/run_all_experiments.py`: **10000 samples**.
- The orchestrator rejects smoke results and requires `is_smoke=false`, `test_size=10000`, and `evaluated_test_size=10000` for full result rows.
- Actual evaluated test size from committed output tables: **MISSING**.

## 6. Quantization settings

### FP32

- Weight bits: **32**
- Activation bits: **32**
- Activation site type: **none**
- Activation granularity: **none**
- Result path expected by config/orchestrator: `outputs/results/fp32_result.csv`
- Actual result file status: **MISSING**

### INT8-MinMax

- Weight dtype: **int8**
- Weight integer range: **[-127, 127]**
- Weight granularity: **per_channel**
- Weight symmetry: **symmetric**
- Weight channel axis: **0**
- Activation dtype: **uint8**
- Activation integer range: **[0, 255]**
- Activation granularity: **per_tensor**
- Activation symmetry: **affine**
- Activation site: **conv_linear_output**
- Activation clipping/calibration: **MinMax from calibration data**
- Result path expected by config/orchestrator: `outputs/results/int8_minmax_result.csv`
- Actual result file status: **MISSING**

### INT4-MinMax

- Weight dtype: **int4**
- Weight integer range: **[-7, 7]**
- Weight granularity: **per_channel**
- Weight symmetry: **symmetric**
- Weight channel axis: **0**
- Activation dtype: **uint4**
- Activation integer range: **[0, 15]**
- Activation granularity: **per_tensor_per_relu_module**
- Activation symmetry: **affine**
- Activation site: **post_relu**
- Activation clip minimum: **0**
- Activation clip maximum source: **calibration_max**
- Result path expected by config/orchestrator: `outputs/results/int4_minmax_result.csv`
- Actual result file status: **MISSING**

### INT4-P99.9 fixed clipping

- Weight dtype: **int4**
- Weight integer range: **[-7, 7]**
- Weight granularity: **per_channel**
- Weight symmetry: **symmetric**
- Activation dtype: **uint4**
- Activation integer range: **[0, 15]**
- Activation granularity: **per_tensor_per_relu_module**
- Activation symmetry: **affine**
- Activation site: **post_relu**
- Activation clip minimum: **0**
- Activation clip method: **fixed_percentile**
- Activation percentile: **99.9**
- Activation quantile: **0.999**
- Activation clip maximum source: **calibration_percentile**
- Result path expected by config/orchestrator: `outputs/results/int4_p999_result.csv`
- Actual result file status: **MISSING**

### INT4-MSE-Selected

- Weight dtype: **int4**
- Weight integer range: **[-7, 7]**
- Weight granularity: **per_channel**
- Weight symmetry: **symmetric**
- Activation dtype: **uint4**
- Activation integer range: **[0, 15]**
- Activation granularity: **per_tensor_per_relu_module**
- Activation symmetry: **affine**
- Activation site: **post_relu**
- Activation clip minimum: **0**
- Activation clip method: **mse_selected**
- Candidate percentiles: **99.0, 99.5, 99.9, 99.95, 100.0**
- Activation clip maximum source: **calibration_mse_selected_percentile**
- Threshold result path expected by config: `outputs/results/mse_selected_thresholds.csv`
- Main result path expected by config/orchestrator: `outputs/results/int4_mse_selected_result.csv`
- Actual result file status: **MISSING**

## 7. Baselines

1. **FP32**: floating-point reference.
2. **INT8-MinMax**: quantization sanity-check baseline.
3. **INT4-MinMax**: main low-bit PTQ baseline.
4. **INT4-P99.9 fixed clipping**: fixed percentile clipping baseline.

The proposed method is **INT4-MSE-Selected** and should be compared against the baselines above.

## 8. Main result table

No committed `outputs/tables/` or `outputs/results/` table/result CSV was found in the current repository. Therefore, the metric values below are marked as **MISSING**. Do not replace these fields with numbers unless they are copied from real experiment outputs.

| Method | Weight bits | Activation bits | Activation site type | Activation granularity | Top-1 accuracy (%) | Accuracy drop vs FP32 (pp) | Activation MSE | Logit MSE | Result source |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| FP32 | 32 | 32 | none | none | MISSING | MISSING | MISSING | MISSING | `outputs/results/fp32_result.csv` MISSING |
| INT8-MinMax | 8 | 8 | conv_linear_output | per_tensor | MISSING | MISSING | MISSING | MISSING | `outputs/results/int8_minmax_result.csv` MISSING |
| INT4-MinMax | 4 | 4 | post_relu | per_tensor_per_relu_module | MISSING | MISSING | MISSING | MISSING | `outputs/results/int4_minmax_result.csv` MISSING |
| INT4-P99.9 | 4 | 4 | post_relu | per_tensor_per_relu_module | MISSING | MISSING | MISSING | MISSING | `outputs/results/int4_p999_result.csv` MISSING |
| INT4-MSE-Selected | 4 | 4 | post_relu | per_tensor_per_relu_module | MISSING | MISSING | MISSING | MISSING | `outputs/results/int4_mse_selected_result.csv` MISSING |

## 9. Figure list and each figure's intended message

1. **`outputs/figures/accuracy_drop.png`** — **MISSING**
   - Intended message: compare Top-1 accuracy and accuracy drop across FP32, INT8-MinMax, INT4-MinMax, INT4-P99.9, and INT4-MSE-Selected.
   - Source in code: generated by `write_accuracy_drop_figure(...)` if real main rows exist.

2. **`outputs/figures/error_metrics.png`** — **MISSING**
   - Intended message: compare activation reconstruction MSE and logit MSE across quantized methods.
   - Source in code: generated by `write_error_metric_figure(...)` if real main rows exist.

3. **`outputs/figures/layerwise_mse.png`** — **MISSING**
   - Intended message: show layer-wise activation reconstruction MSE behavior for MSE-selected clipping.
   - Source in config: expected output path in `configs/int4_mse_selected_cifar10.yaml`.

4. **Activation histogram figure** — **MISSING**
   - Intended message if later added: visualize activation outliers and clipping thresholds.
   - Current status: no committed histogram figure path was found from the inspected files.

## 10. Three main observations

1. **The experiment design is coherent, but the committed result artifacts are missing.**  
   The repository defines a complete comparison among FP32, INT8-MinMax, INT4-MinMax, INT4-P99.9, and INT4-MSE-Selected. However, no committed output table/result CSV was found, so the paper cannot yet report numeric accuracy, accuracy drop, activation MSE, or logit MSE from the current repository.

2. **The proposed method is implemented as a calibration-time layer-wise clipping search.**  
   `src/quant/clipping_search.py` collects post-ReLU activation values, evaluates the candidate percentile set `{99.0, 99.5, 99.9, 99.95, 100.0}`, computes fake-quantization reconstruction MSE, and selects the lowest-MSE candidate for each activation site.

3. **The claim boundary must be conservative until real outputs are committed.**  
   The paper may describe the configured method, baselines, dataset, model, and validation rules. It must not claim that INT4-MSE-Selected improves accuracy or MSE unless real output rows are added under `outputs/tables/` or `outputs/results/`.

## 11. Limitations

- The exact requested file `configs/cifar10.yaml` is missing; only method-specific config files were found.
- The exact requested file `src/run_all_experiments.py` is missing; the actual orchestrator is `src/experiments/run_all_experiments.py`.
- No committed `outputs/tables/` files were found, so the main result table currently has no verified numeric results.
- No committed `outputs/results/` per-method CSVs were found, even though the configs/orchestrator define expected paths.
- No committed `outputs/figures/` files were found, so figure availability is currently **MISSING**.
- The project scope is limited to CIFAR-10 and `resnet18_cifar` in the inspected configs.
- The method is evaluated/configured as simulated PTQ / fake quantization, not a real INT4 inference kernel.
- No real hardware deployment, latency, throughput, energy, or memory-bandwidth measurement is available from the inspected files.
- No QAT comparison is included in the inspected project scope.
- No multi-seed, multi-dataset, multi-architecture, or calibration-size sensitivity result was found.
- The method selects from a small predefined percentile candidate set; it must not be described as globally optimal clipping.

## 12. Claims that are allowed

The paper may claim:

1. The project studies **INT4 post-training quantization** for a **CIFAR-10 / resnet18_cifar** classification setup.
2. The configured proposed method is **Layer-wise MSE-Selected Activation Clipping**.
3. The method selects each post-ReLU activation clipping threshold from `{99.0, 99.5, 99.9, 99.95, 100.0}` by minimizing calibration activation reconstruction MSE.
4. The configured comparison groups are FP32, INT8-MinMax, INT4-MinMax, INT4-P99.9, and INT4-MSE-Selected.
5. The quantization settings use per-channel symmetric signed weights and affine unsigned activations for the INT4 post-ReLU variants.
6. The calibration size configured for quantized methods is **1024** CIFAR-10 training samples.
7. The orchestrator requires full CIFAR-10 test evaluation with `test_size=10000` and `evaluated_test_size=10000` before accepting result rows.
8. The current repository, as inspected here, is missing committed numeric output tables/results; therefore, numeric result claims must wait until real output files are present.

## 13. Claims that are forbidden

The paper must not claim:

1. Any specific Top-1 accuracy, accuracy drop, activation MSE, or logit MSE value while the corresponding output table/result CSV is **MISSING**.
2. That INT4-MSE-Selected outperforms INT4-MinMax, INT4-P99.9, INT8-MinMax, or FP32 in accuracy unless supported by real output rows.
3. That INT4-MSE-Selected achieves the best INT4 accuracy unless supported by real output rows.
4. That the method reaches INT8-level accuracy.
5. That the method is SOTA.
6. That the method is globally optimal; it is only selected from a predefined candidate percentile set.
7. That the results generalize to ImageNet, CIFAR-100, detection, segmentation, transformers, MobileNetV2, or other models without additional experiments.
8. That the method is robust across random seeds, calibration sizes, datasets, or architectures without corresponding experiments.
9. That a real INT4 inference kernel was implemented.
10. That FPGA/RFSoC, TensorRT, ONNX Runtime, TVM, or other deployment results were measured.
11. That latency, throughput, energy, memory-bandwidth, or real storage/compression improvements were experimentally measured.
12. That activation outliers are fully removed rather than mitigated by clipping.
