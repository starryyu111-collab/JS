# Experiment Notes

## Data Sources

- Main table: `outputs/results/main_results.csv`
- MSE-selected threshold table: `outputs/results/mse_selected_thresholds.csv`
- Available figure: `outputs/figures/layerwise_mse.png`
- Note: no full activation histogram figure was found under `outputs/figures` in the current outputs.

## Captions

### Main Results Table

Table X. CIFAR-10 post-training quantization results for the FP32 baseline, INT8-MinMax sanity check, INT4-MinMax baseline, fixed P99.9 activation clipping, and layer-wise MSE-selected activation clipping. Top-1 accuracy is reported on the full CIFAR-10 test set, while activation MSE and logit MSE quantify reconstruction and output consistency after simulated quantization.

### Activation Histogram

Figure X. Distribution of post-ReLU activation values collected from the CIFAR-10 calibration set. The long-tailed activation distribution indicates the presence of outliers that enlarge the MinMax quantization range, motivating percentile-based clipping before low-bit UINT4 activation fake quantization.

### Layer-Wise MSE Figure

Figure X. Layer-wise activation reconstruction MSE for the INT4-MSE-Selected clipping method. Each bar reports the selected calibration MSE for one observed post-ReLU activation site, with the selected percentile annotated above the bar. The larger errors in later layers suggest that quantization sensitivity is not uniform across the network.

## Observations

1. INT4-MinMax shows the strongest degradation among the INT4 methods, with top-1 accuracy decreasing to 10.6000 and activation MSE increasing to 0.01979739. This supports the expectation that using the full MinMax activation range is vulnerable to activation outliers under 4-bit quantization.

2. Activation clipping substantially improves reconstruction metrics compared with INT4-MinMax. The INT4-MSE-Selected result reduces activation MSE from 0.01979739 to 0.00559793 and logit MSE from 7.63254050 to 1.14619427, indicating better local activation reconstruction and output consistency.

3. The accuracy improvement from MSE-selected clipping is limited and should not be overstated. INT4-MSE-Selected reaches 11.6700 top-1 accuracy, which is close to INT4-P99.9 at 11.6900 and only modestly better than INT4-MinMax. Since all observed layers selected P99.9 and later layers still have larger MSE values, the results suggest that lower local MSE alone does not guarantee a clear accuracy gain; layer sensitivity and accumulated quantization error remain important limitations.
