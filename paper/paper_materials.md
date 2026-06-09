# Paper Materials

> Scope note: this file is a structured material sheet for drafting the course paper. It is not a complete paper draft.
>
> Source status checked during preparation:
> - `README.md`: found and inspected.
> - `Claude.md`: found and inspected.
> - `configs/cifar10.yaml`: MISSING. The experiment orchestrator instead references method-specific config files: `configs/fp32_cifar10.yaml`, `configs/int8_minmax_cifar10.yaml`, `configs/int4_minmax_cifar10.yaml`, `configs/int4_p999_cifar10.yaml`, and `configs/int4_mse_selected_cifar10.yaml`.
> - `src/run_all_experiments.py`: MISSING. The actual orchestrator is `src/experiments/run_all_experiments.py`.
> - `src/quant/clipping_search.py`: found and inspected.
> - `outputs/autodl_results.zip`: found and parsed for real experiment results.

## 1. Paper title

**Research on Low-Bit (INT4) Post-Training Quantization for Lightweight CNNs Based on Activation Outlier Suppression**

## 2. Research problem

Low-bit post-training quantization can strongly amplify activation outliers in CNNs. In the INT4 setting, MinMax activation calibration may allocate too much quantization range to rare large activation values, increasing reconstruction error and causing classification accuracy degradation.

This project studies whether activation clipping selected from calibration data can make INT4 PTQ more stable than ordinary INT4 MinMax PTQ on CIFAR-10.

## 3. Method name

**Layer-wise MSE-Selected Activation Clipping**

Implementation name in the result table: **INT4-MSE-Selected**.

For each observed post-ReLU activation site, the method searches the candidate percentile set:

`{99.0, 99.5, 99.9, 99.95, 100.0}`

For each candidate percentile, it computes the clipping threshold, fake-quantizes/dequantizes the collected calibration activations, computes the reconstruction MSE against the original FP32 activations, and selects the threshold with the minimum activation reconstruction MSE for that layer.

Important wording constraint: this is **MSE-selected within a predefined candidate percentile set**, not a globally optimal clipping method.

## 4. Dataset and model

- Dataset: **CIFAR-10**
- Model: **resnet18_cifar**
- Number of classes: **10**
- FP32 training split recorded in config/result metadata:
  - Train size: **45,000**
  - Validation size: **5,000**
  - Test size: **10,000**
- FP32 training setting:
  - Epochs: **100**
  - Seed: **42**
  - Batch size: **128**
  - Optimizer: **SGD**
  - Learning rate: **0.1**
  - Weight decay: **0.0005**
  - Scheduler: **MultiStepLR**
  - Device used in recorded full results: **cuda**

## 5. Calibration set and evaluation set

### Calibration set

- Source: **CIFAR-10 training set** (`CIFAR10 train=True`)
- Calibration size: **1024 samples**
- Calibration seed: **42**
- Calibration batches: **8**
- Calibration index checksum: **827329294**

### Evaluation set

- Evaluation set: **full CIFAR-10 test set**
- Test size: **10,000 samples**
- Evaluated test size in all full results: **10,000 samples**
- The main table must not use smoke-test results.

## 6. Quantization settings

### FP32

- Weight bits: **32**
- Activation bits: **32**
- Activation site type: **none**
- Activation granularity: **none**

### INT8-MinMax

- Weight dtype: **int8**
- Weight integer range: **[-127, 127]**
- Weight granularity: **per-channel**
- Weight symmetry: **symmetric**
- Activation dtype: **uint8**
- Activation integer range: **[0, 255]**
- Activation site: **conv_linear_output**
- Activation granularity: **per_tensor**
- Activation clipping/calibration: **MinMax from calibration data**
- Role: quantization sanity-check baseline.

### INT4-MinMax

- Weight dtype: **int4**
- Weight integer range: **[-7, 7]**
- Weight granularity: **per-channel**
- Weight symmetry: **symmetric**
- Activation dtype: **uint4**
- Activation integer range: **[0, 15]**
- Activation site: **post_relu**
- Activation granularity: **per_tensor_per_relu_module**
- Activation clip minimum: **0**
- Activation clip maximum source: **calibration_max**
- Role: main INT4 baseline.

### INT4-P99.9

- Same INT4 weight and activation integer ranges as INT4-MinMax.
- Activation site: **post_relu**
- Activation granularity: **per_tensor_per_relu_module**
- Activation clip minimum: **0**
- Activation clip method: **fixed_percentile**
- Activation percentile: **99.9**
- Activation quantile: **0.999**
- Activation clip maximum source: **calibration_percentile**
- Role: fixed clipping baseline.

### INT4-MSE-Selected

- Same INT4 weight and activation integer ranges as INT4-MinMax.
- Activation site: **post_relu**
- Activation granularity: **per_tensor_per_relu_module**
- Activation clip minimum: **0**
- Activation clip method: **mse_selected**
- Candidate percentiles: **99.0, 99.5, 99.9, 99.95, 100.0**
- Activation clip maximum source: **calibration_mse_selected_percentile**
- Role: proposed method.

## 7. Baselines

1. **FP32**: floating-point reference upper bound.
2. **INT8-MinMax**: sanity-check PTQ baseline; expected to be close to FP32.
3. **INT4-MinMax**: main low-bit PTQ baseline.
4. **INT4-P99.9**: fixed activation clipping baseline.
5. **INT4-MSE-Selected**: proposed layer-wise MSE-selected activation clipping method.

## 8. Main result table

The following table uses real results parsed from `outputs/autodl_results.zip`. Do not change these numbers unless new experiments are run and the result artifact is updated.

| Method | Weight bits | Activation bits | Activation site type | Activation granularity | Top-1 accuracy (%) | Accuracy drop vs FP32 (pp) | Activation MSE | Logit MSE |
|---|---:|---:|---|---|---:|---:|---:|---:|
| FP32 | 32 | 32 | none | none | 94.1300 | 0.0000 | 0.00000000 | 0.00000000 |
| INT8-MinMax | 8 | 8 | conv_linear_output | per_tensor | 94.1100 | 0.0200 | 0.00005792 | 0.00910544 |
| INT4-MinMax | 4 | 4 | post_relu | per_tensor_per_relu_module | 88.2000 | 5.9300 | 0.00273660 | 1.84912006 |
| INT4-P99.9 | 4 | 4 | post_relu | per_tensor_per_relu_module | 92.9700 | 1.1600 | 0.00035606 | 0.46350010 |
| INT4-MSE-Selected | 4 | 4 | post_relu | per_tensor_per_relu_module | 92.8600 | 1.2700 | 0.00034846 | 0.41728575 |

## 9. Figure list and each figure's intended message

1. **`outputs/figures/accuracy_drop.png`**
   - Intended message: compare Top-1 accuracy and accuracy drop across FP32, INT8-MinMax, INT4-MinMax, INT4-P99.9, and INT4-MSE-Selected.
   - Expected paper use: main visual summary showing that INT4 MinMax suffers a large accuracy drop, while activation clipping recovers much of the lost accuracy.

2. **`outputs/figures/error_metrics.png`**
   - Intended message: compare activation reconstruction MSE and logit MSE across quantized methods.
   - Expected paper use: explain why activation clipping helps: it sharply reduces activation and logit errors compared with INT4-MinMax.

3. **`outputs/figures/layerwise_mse.png`**
   - Intended message: show layer-wise activation reconstruction MSE behavior under the MSE-selected clipping search.
   - Expected paper use: support the claim that clipping decisions are layer-dependent rather than one uniform threshold being optimal for every layer.

4. **Activation histogram figure**: **MISSING**
   - Intended message if later added: visualize activation outliers and the effect of clipping thresholds.
   - Current status: not found in the parsed result artifact or the inspected config paths.

## 10. Three main observations

1. **INT8-MinMax is a valid sanity-check baseline.**  
   INT8-MinMax achieves **94.1100%** Top-1 accuracy, only **0.0200 percentage points** below the FP32 baseline of **94.1300%**. This indicates that the evaluation and simulated quantization pipeline is not generally broken.

2. **Naive INT4-MinMax causes a large accuracy drop.**  
   INT4-MinMax drops from **94.1300%** to **88.2000%**, with an accuracy drop of **5.9300 percentage points**. Its activation MSE (**0.00273660**) and logit MSE (**1.84912006**) are also much larger than the clipped INT4 variants.

3. **Activation clipping substantially improves INT4 PTQ, but the proposed method is not the top-accuracy method in this run.**  
   INT4-P99.9 reaches **92.9700%**, while INT4-MSE-Selected reaches **92.8600%**. Therefore, the paper may claim that MSE-selected clipping reduces activation/logit error compared with INT4-MinMax and slightly improves MSE metrics compared with P99.9, but it must not claim the best Top-1 accuracy. Specifically, INT4-MSE-Selected has lower activation MSE (**0.00034846 vs 0.00035606**) and lower logit MSE (**0.41728575 vs 0.46350010**) than INT4-P99.9, but slightly lower Top-1 accuracy (**92.8600% vs 92.9700%**).

## 11. Limitations

- Only one dataset is evaluated: **CIFAR-10**.
- Only one model is evaluated in the main full experiment: **resnet18_cifar**.
- Only one recorded random seed is used: **42**.
- Calibration uses a fixed subset of **1024** CIFAR-10 training samples; no calibration-size sensitivity study is included.
- The implementation uses simulated PTQ / fake quantization, not a real INT4 inference kernel.
- No real hardware latency, throughput, energy, or memory-bandwidth measurement is reported.
- No QAT comparison is included.
- INT8 and INT4 use different activation observation sites/granularities in the recorded setup, so INT8-MinMax should be treated as a sanity check rather than a strictly controlled ablation against the INT4 post-ReLU setup.
- The proposed threshold is selected only from a small predefined percentile candidate set; it is not globally optimal.
- The main accuracy improvement claim is strongest against INT4-MinMax, not against INT4-P99.9.

## 12. Claims that are allowed

The paper may claim:

1. On **CIFAR-10 / resnet18_cifar**, naive **INT4-MinMax PTQ** causes a substantial accuracy drop compared with FP32.
2. On this experiment, **activation clipping** strongly improves INT4 PTQ compared with INT4-MinMax.
3. **INT4-P99.9** improves Top-1 accuracy from **88.2000%** to **92.9700%** compared with INT4-MinMax.
4. **INT4-MSE-Selected** improves Top-1 accuracy from **88.2000%** to **92.8600%** compared with INT4-MinMax.
5. **INT4-MSE-Selected** reduces activation reconstruction MSE from **0.00273660** to **0.00034846** compared with INT4-MinMax.
6. **INT4-MSE-Selected** reduces logit MSE from **1.84912006** to **0.41728575** compared with INT4-MinMax.
7. Compared with INT4-P99.9, **INT4-MSE-Selected** has slightly lower activation MSE and logit MSE in this run.
8. The method is a simple calibration-time PTQ method and does not require retraining.
9. The method is **MSE-selected within the candidate set** `{99.0, 99.5, 99.9, 99.95, 100.0}`.
10. The results suggest that suppressing activation outliers can stabilize INT4 PTQ on this CIFAR-10 experiment.

## 13. Claims that are forbidden

The paper must not claim:

1. That INT4-MSE-Selected achieves the best Top-1 accuracy among all INT4 methods in the recorded experiment.
2. That INT4-MSE-Selected outperforms INT4-P99.9 in classification accuracy.
3. That the method is globally optimal; it is only selected from a predefined candidate percentile set.
4. That the method reaches INT8-level accuracy.
5. That the method is SOTA.
6. That the method generalizes to ImageNet, detection, segmentation, transformers, MobileNetV2, or other models without additional experiments.
7. That real INT4 hardware acceleration was implemented.
8. That latency, throughput, energy, or memory-bandwidth improvements were measured.
9. That the project includes a real INT4 kernel, TensorRT deployment, ONNX Runtime deployment, TVM deployment, or FPGA/RFSoC deployment.
10. That compression or storage reduction is an experimentally measured deployment result. At most, bit-width reduction can be discussed as a theoretical estimate.
11. That the method removes all activation outlier effects.
12. That the method is robust across seeds, calibration sizes, or datasets; these are not tested here.
