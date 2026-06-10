# Paper Materials

> Scope note: this file is a structured material sheet for drafting the course paper. It is **not** a complete paper draft.
>
> Generation rule followed: only information that can be read from the current GitHub repository is used. Requested artifacts that could not be found are marked as **MISSING**. The main result table below uses committed experiment output values, not invented numbers.
>
> Regenerated on: 2026-06-10

## Source status checked

| Requested source | Status | Notes |
|---|---|---|
| `README.md` | FOUND | Project describes CIFAR-10 INT4 PTQ with layer-wise MSE-selected activation clipping. |
| `Claude.md` | FOUND | Provides paper title, scope, required methods, workflow, and claim restrictions. |
| `configs/cifar10.yaml` | MISSING | The exact requested single config file was not found. The repository uses method-specific configs instead: `configs/fp32_cifar10.yaml`, `configs/int8_minmax_cifar10.yaml`, `configs/int4_minmax_cifar10.yaml`, `configs/int4_p999_cifar10.yaml`, and `configs/int4_mse_selected_cifar10.yaml`. |
| `src/experiments/run_all_experiments.py` | FOUND | Defines the five compared methods, validates full non-smoke result rows, writes `outputs/results/main_results.csv` and `outputs/results/main_results.md`, and generates summary figures. |
| `src/quant/clipping_search.py` | FOUND | Implements layer-wise MSE-selected post-ReLU activation clipping. |
| `outputs/tables/` | MISSING | No committed main table was found at checked paths such as `outputs/tables/main_results.md` or `outputs/tables/main_results.csv`. The actual committed result tables are under `outputs/results/`. |
| `outputs/results/` | FOUND | `main_results.md`, `main_results.csv`, per-method result CSVs, and `mse_selected_thresholds.csv` were found. |
| `outputs/figures/` | FOUND | `accuracy_drop.png`, `error_metrics.png`, and `layerwise_mse.png` were found. No activation histogram figure was found from the inspected committed files. |

---

## 1. Paper title

**Research on Low-Bit (INT4) Post-Training Quantization for Lightweight CNNs Based on Activation Outlier Suppression**

---

## 2. Research problem

Low-bit post-training quantization is sensitive to activation outliers. In INT4 PTQ, ordinary MinMax activation calibration may allocate too much quantization range to rare large activation values, increasing activation reconstruction error and degrading classification accuracy.

This project studies whether calibration-time activation clipping can make INT4 PTQ more stable than ordinary INT4 MinMax PTQ on a CIFAR-10 CNN classification task.

---

## 3. Method name

**Layer-wise MSE-Selected Activation Clipping**

Implementation/result-table method name: **INT4-MSE-Selected**.

Method definition from the inspected code:

1. Collect post-ReLU activations on the calibration set.
2. For each observed activation layer/site, evaluate the candidate percentile set `{99.0, 99.5, 99.9, 99.95, 100.0}`.
3. Convert each candidate percentile into a clipping threshold.
4. Fake-quantize and dequantize the collected activation values under each threshold.
5. Compute activation reconstruction MSE between original FP32 activations and dequantized activations.
6. Select the candidate threshold with the lowest MSE for that layer.

Important wording constraint: the method is **MSE-selected within a predefined candidate percentile set**, not globally optimal clipping.

---

## 4. Dataset and model

- Dataset: **CIFAR-10**
- Model: **resnet18_cifar**
- Number of classes: **10**
- FP32 checkpoint path: `checkpoints/fp32_best.pt`
- FP32 training setup from config:
  - Seed: **42**
  - Deterministic: **true**
  - Epochs: **100**
  - Batch size: **128**
  - Optimizer: **SGD**
  - Learning rate: **0.1**
  - Momentum: **0.9**
  - Weight decay: **0.0005**
  - Scheduler: **MultiStepLR**, milestones `[50, 75]`, gamma `0.1`
- FP32 result metadata from committed output:
  - Train size: **45000**
  - Validation size: **5000**
  - Test size: **10000**
  - Best epoch: **100**
  - Validation Top-1 accuracy: **94.5600%**
  - Test Top-1 accuracy: **94.1300%**

---

## 5. Calibration set and evaluation set

### Calibration set

| Item | Value |
|---|---|
| Used by | Quantized methods: INT8-MinMax, INT4-MinMax, INT4-P99.9, INT4-MSE-Selected |
| Source | `CIFAR10 train=True` |
| Calibration size | **1024** samples |
| Calibration seed | **42** |
| Calibration batch count | **8** |
| Calibration index checksum | **827329294** |

### Evaluation set

| Item | Value |
|---|---|
| Source | CIFAR-10 test set / `train=False` |
| Test size | **10000** samples |
| Evaluated test size in committed quantized outputs | **10000** samples |
| FP32 evaluated test size | **10000** samples |
| Smoke setting | `is_smoke=false` for all committed main-result rows |

The experiment orchestrator rejects smoke rows and requires full CIFAR-10 test evaluation with `test_size=10000` and `evaluated_test_size=10000` before a row can be used in the main result table.

---

## 6. Quantization settings

| Method | Weight setting | Activation setting | Activation clipping/calibration | Result path |
|---|---|---|---|---|
| FP32 | 32-bit floating point | 32-bit floating point | none | `outputs/results/fp32_result.csv` |
| INT8-MinMax | int8, qmin=-127, qmax=127, per-channel, symmetric, channel_axis=0 | uint8, qmin=0, qmax=255, per-tensor, affine, site=`conv_linear_output` | MinMax from calibration data; clip source=`calibration_minmax` | `outputs/results/int8_minmax_result.csv` |
| INT4-MinMax | int4, qmin=-7, qmax=7, per-channel, symmetric, channel_axis=0 | uint4, qmin=0, qmax=15, per-tensor-per-ReLU-module, affine, site=`post_relu` | clip_min=0, clip source=`calibration_max` | `outputs/results/int4_minmax_result.csv` |
| INT4-P99.9 | int4, qmin=-7, qmax=7, per-channel, symmetric, channel_axis=0 | uint4, qmin=0, qmax=15, per-tensor-per-ReLU-module, affine, site=`post_relu` | fixed percentile clipping, percentile=99.9, quantile=0.999, clip_min=0, clip source=`calibration_percentile` | `outputs/results/int4_p999_result.csv` |
| INT4-MSE-Selected | int4, qmin=-7, qmax=7, per-channel, symmetric, channel_axis=0 | uint4, qmin=0, qmax=15, per-tensor-per-ReLU-module, affine, site=`post_relu` | MSE-selected clipping from candidate percentiles `[99.0, 99.5, 99.9, 99.95, 100.0]`, clip_min=0, clip source=`calibration_mse_selected_percentile` | `outputs/results/int4_mse_selected_result.csv` |

Additional committed INT4-MSE-Selected threshold artifact:

- Threshold CSV: `outputs/results/mse_selected_thresholds.csv`
- Layer-wise MSE figure: `outputs/figures/layerwise_mse.png`

---

## 7. Baselines

1. **FP32**: floating-point reference / upper-bound baseline.
2. **INT8-MinMax**: quantization sanity-check baseline.
3. **INT4-MinMax**: main low-bit PTQ baseline.
4. **INT4-P99.9 fixed clipping**: fixed percentile clipping baseline.

The proposed method is **INT4-MSE-Selected** and should be compared against the baselines above.

---

## 8. Main result table

Source: committed `outputs/results/main_results.md` and `outputs/results/main_results.csv`.

Note: `outputs/tables/` is **MISSING**, so the table below intentionally uses the actual committed result table under `outputs/results/`.

| Method | Weight bits | Activation bits | Activation site type | Activation granularity | Top-1 accuracy (%) | Accuracy drop vs FP32 (pp) | Activation MSE | Logit MSE |
|---|---:|---:|---|---|---:|---:|---:|---:|
| FP32 | 32 | 32 | none | none | 94.1300 | 0.0000 | 0.00000000 | 0.00000000 |
| INT8-MinMax | 8 | 8 | conv_linear_output | per_tensor | 94.1100 | 0.0200 | 0.00005792 | 0.00910544 |
| INT4-MinMax | 4 | 4 | post_relu | per_tensor_per_relu_module | 88.2000 | 5.9300 | 0.00273660 | 1.84912006 |
| INT4-P99.9 | 4 | 4 | post_relu | per_tensor_per_relu_module | 92.9700 | 1.1600 | 0.00035606 | 0.46350010 |
| INT4-MSE-Selected | 4 | 4 | post_relu | per_tensor_per_relu_module | 92.8600 | 1.2700 | 0.00034846 | 0.41728575 |

Useful comparisons derived from the real table:

- INT4-MSE-Selected improves Top-1 accuracy over INT4-MinMax by **4.6600 percentage points**: 92.8600% vs 88.2000%.
- INT4-P99.9 improves Top-1 accuracy over INT4-MinMax by **4.7700 percentage points**: 92.9700% vs 88.2000%.
- INT4-MSE-Selected has lower activation MSE than INT4-MinMax: 0.00034846 vs 0.00273660.
- INT4-MSE-Selected has lower logit MSE than INT4-MinMax: 0.41728575 vs 1.84912006.
- INT4-MSE-Selected has slightly lower activation MSE and logit MSE than INT4-P99.9, but **does not** have higher Top-1 accuracy than INT4-P99.9 in the committed result table.

---

## 9. Figure list and each figure's intended message

| Figure | Status | Intended message | Source/notes |
|---|---|---|---|
| `outputs/figures/accuracy_drop.png` | FOUND | Compare Top-1 accuracy and accuracy drop across FP32, INT8-MinMax, INT4-MinMax, INT4-P99.9, and INT4-MSE-Selected. | Generated by the experiment orchestrator from the main result rows. |
| `outputs/figures/error_metrics.png` | FOUND | Compare activation reconstruction MSE and logit MSE across quantized methods. | Generated by the experiment orchestrator from the main result rows. |
| `outputs/figures/layerwise_mse.png` | FOUND | Show per-layer selected calibration reconstruction MSE for INT4-MSE-Selected and annotate the selected percentile per layer. | Generated from INT4-MSE-Selected layer search results. |
| Activation histogram figure | MISSING | Intended message if later added: visualize activation outliers and show how clipping thresholds suppress rare large activation values. | No committed histogram figure path was found from the inspected files. |

---

## 10. Three main observations

1. **INT8-MinMax validates the basic quantization pipeline.**  
   INT8-MinMax reaches **94.1100%** Top-1 accuracy, only **0.0200 pp** below the FP32 result of **94.1300%**. This indicates that the evaluation pipeline and checkpoint are reasonable before moving to INT4.

2. **Plain INT4-MinMax is the weakest INT4 setting, and clipping substantially reduces the degradation.**  
   INT4-MinMax drops to **88.2000%** Top-1 accuracy with a **5.9300 pp** accuracy drop. Both clipping variants recover most of this loss: INT4-P99.9 reaches **92.9700%**, and INT4-MSE-Selected reaches **92.8600%**.

3. **INT4-MSE-Selected gives the lowest INT4 error metrics, but not the highest INT4 accuracy.**  
   INT4-MSE-Selected has the lowest INT4 activation MSE (**0.00034846**) and logit MSE (**0.41728575**), slightly better than INT4-P99.9 on those error metrics. However, INT4-P99.9 has slightly higher Top-1 accuracy (**92.9700%** vs **92.8600%**). Therefore, the paper should claim improved error behavior and robustness relative to INT4-MinMax, not unconditional accuracy superiority over all clipping baselines.

---

## 11. Limitations

- The exact requested file `configs/cifar10.yaml` is missing; the repository uses method-specific CIFAR-10 config files instead.
- The exact requested directory `outputs/tables/` is missing or has no checked main-result table; the committed result tables are under `outputs/results/`.
- The project is limited to **CIFAR-10** and **resnet18_cifar** in the inspected committed experiment setup.
- The committed results appear to be from a single seed (**42**) and one checkpoint (`checkpoints/fp32_best.pt`). No multi-seed statistics were found.
- No ImageNet, COCO, MobileNetV2, or cross-architecture generalization result was found.
- No calibration-size sensitivity experiment was found.
- No Quantization-Aware Training (QAT) baseline was found.
- The implementation uses simulated PTQ / fake quantization; no real INT4 inference kernel is implemented.
- No real hardware latency, throughput, energy, or memory-bandwidth measurement is available.
- The method selects thresholds from a predefined candidate percentile set; it is not a globally optimal clipping method.
- INT4-MSE-Selected is slightly worse than INT4-P99.9 in Top-1 accuracy in the committed result table, even though it has lower activation MSE and logit MSE.
- The activation histogram figure is missing, so any paper discussion of activation histograms must either omit the figure or generate/commit it later from real data.

---

## 12. Claims that are allowed

The paper may claim:

1. The project studies **INT4 post-training quantization** for a **CIFAR-10 / resnet18_cifar** classification setup.
2. The proposed method is **Layer-wise MSE-Selected Activation Clipping**, implemented as **INT4-MSE-Selected**.
3. The method selects each post-ReLU activation clipping threshold from `{99.0, 99.5, 99.9, 99.95, 100.0}` by minimizing calibration activation reconstruction MSE.
4. The configured comparison groups are FP32, INT8-MinMax, INT4-MinMax, INT4-P99.9, and INT4-MSE-Selected.
5. The quantized methods use **1024** CIFAR-10 training samples for calibration and evaluate on the full **10000-sample** CIFAR-10 test set.
6. In the committed result table, INT8-MinMax has only **0.0200 pp** accuracy drop from FP32.
7. In the committed result table, INT4-MinMax has **88.2000%** Top-1 accuracy and **5.9300 pp** accuracy drop.
8. In the committed result table, INT4-MSE-Selected improves Top-1 accuracy over INT4-MinMax by **4.6600 pp**.
9. In the committed result table, INT4-MSE-Selected reduces activation MSE compared with INT4-MinMax: **0.00034846** vs **0.00273660**.
10. In the committed result table, INT4-MSE-Selected reduces logit MSE compared with INT4-MinMax: **0.41728575** vs **1.84912006**.
11. In the committed result table, INT4-MSE-Selected has slightly lower activation MSE and logit MSE than INT4-P99.9.
12. The results suggest that activation clipping can mitigate INT4 MinMax sensitivity to activation outliers in this CIFAR-10 / resnet18_cifar experiment.
13. The work is a small, controlled course-paper experiment, not a SOTA quantization framework.

---

## 13. Claims that are forbidden

The paper must **not** claim:

1. That INT4-MSE-Selected is SOTA.
2. That INT4-MSE-Selected is globally optimal clipping.
3. That the method always outperforms fixed percentile clipping.
4. That INT4-MSE-Selected has the best Top-1 accuracy among all INT4 methods, because INT4-P99.9 is slightly higher in the committed table.
5. That the method reaches INT8-level or FP32-level accuracy.
6. That the method was evaluated on ImageNet, COCO, MobileNetV2, or multiple architectures.
7. That the result is statistically robust across multiple seeds, because no multi-seed result was found.
8. That the method uses QAT.
9. That the method implements true INT4 hardware kernels.
10. That the method provides measured latency, throughput, energy, or hardware speedup.
11. That compression or memory reduction was empirically measured on hardware.
12. That `outputs/tables/` contains the main result table; the found table is under `outputs/results/`.
13. That activation histogram analysis is backed by a committed histogram figure, because that figure is currently **MISSING**.
