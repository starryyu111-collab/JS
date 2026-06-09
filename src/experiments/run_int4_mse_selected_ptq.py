from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.experiments.run_int4_minmax_ptq import (
    apply_cli_overrides as apply_base_cli_overrides,
    build_cifar10_ptq_loaders,
    compute_index_checksum,
    compute_logit_mse,
    evaluate_top1,
    load_config,
    resolve_activation_range,
    resolve_device,
    resolve_result_path,
    _validate_int4_post_relu_activation_config,
    _validate_int4_weight_config,
)
from src.models import build_model
from src.quant.clipping_search import (
    MSE_SELECTED_PERCENTILES,
    LayerClippingSearchResult,
    calibrate_post_relu_activation_mse_selected,
)
from src.quant.int4_minmax import (
    ACTIVATION_UINT4_QMAX,
    ACTIVATION_UINT4_QMIN,
    WEIGHT_INT4_QMAX,
    WEIGHT_INT4_QMIN,
    apply_weight_fake_quantization,
    attach_post_relu_activation_fake_quantization,
    compute_activation_reconstruction_mse,
)
from src.utils.checkpoint import load_checkpoint
from src.utils.csv_io import write_rows_csv, write_single_row_csv
from src.utils.logging import configure_logger
from src.utils.seed import set_seed


RESULT_COLUMNS = [
    "method",
    "model",
    "dataset",
    "seed",
    "checkpoint_path",
    "checkpoint_model_name",
    "calibration_size",
    "calibration_seed",
    "calibration_source",
    "calibration_num_batches",
    "calibration_index_checksum",
    "test_size",
    "evaluated_test_size",
    "top1_accuracy",
    "fp32_top1_accuracy",
    "int4_top1_accuracy",
    "accuracy_drop",
    "activation_mse",
    "logit_mse",
    "activation_quant_dtype",
    "activation_qmin",
    "activation_qmax",
    "weight_qmin",
    "weight_qmax",
    "num_observed_activation_sites",
    "num_quantized_modules",
    "min_activation_scale",
    "max_activation_scale",
    "min_activation_zero_point",
    "max_activation_zero_point",
    "min_weight_scale",
    "max_weight_scale",
    "is_smoke",
    "result_path",
    "threshold_result_path",
    "figure_path",
    "log_path",
    "device",
    "batch_size",
    "observed_activation_sites",
    "activation_clip_method",
    "activation_clip_source",
    "activation_clip_min",
    "activation_granularity",
    "activation_site_type",
    "threshold_search",
    "mse_selected",
    "candidate_percentiles",
    "min_selected_activation_alpha",
    "max_selected_activation_alpha",
    "min_selected_layer_mse",
    "max_selected_layer_mse",
    "mean_selected_layer_mse",
    "weight_granularity",
    "weight_symmetry",
]

THRESHOLD_COLUMNS = [
    "layer_name",
    "layer_index",
    "num_activation_elements",
    "selected_percentile",
    "selected_alpha",
    "selected_mse",
    "p99_0_alpha",
    "p99_0_mse",
    "p99_5_alpha",
    "p99_5_mse",
    "p99_9_alpha",
    "p99_9_mse",
    "p99_95_alpha",
    "p99_95_mse",
    "p100_0_alpha",
    "p100_0_mse",
    "activation_qmin",
    "activation_qmax",
    "activation_zero_point",
    "activation_scale",
    "candidate_percentiles",
    "candidate_alphas",
    "candidate_mses",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run INT4-MSE-Selected activation clipping PTQ for CIFAR-10."
    )
    parser.add_argument(
        "--config",
        default="configs/int4_mse_selected_cifar10.yaml",
        help="Path to the INT4-MSE-Selected PTQ YAML config.",
    )
    parser.add_argument("--model", choices=["resnet18_cifar", "compact_cnn"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="Use auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--calibration-size", type=int)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--result-path")
    parser.add_argument("--threshold-result-path")
    parser.add_argument("--figure-path")
    parser.add_argument("--max-calibration-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    return parser.parse_args()


def apply_cli_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bool, bool, bool]:
    resolved, result_path_overridden = apply_base_cli_overrides(config, args)
    threshold_result_path_overridden = args.threshold_result_path is not None
    figure_path_overridden = args.figure_path is not None

    if args.threshold_result_path is not None:
        resolved["paths"]["threshold_result_path"] = args.threshold_result_path
    if args.figure_path is not None:
        resolved["paths"]["figure_path"] = args.figure_path

    return (
        resolved,
        result_path_overridden,
        threshold_result_path_overridden,
        figure_path_overridden,
    )


def run(
    config: dict[str, Any],
    result_path_overridden: bool = False,
    threshold_result_path_overridden: bool = False,
    figure_path_overridden: bool = False,
) -> None:
    paths = config["paths"]
    experiment_config = config["experiment"]
    model_config = config["model"]
    dataset_config = config["dataset"]
    quant_config = config["quantization"]
    smoke_config = config.get("smoke", {})

    max_calibration_batches = smoke_config.get("max_calibration_batches")
    max_test_batches = smoke_config.get("max_test_batches")
    is_smoke = max_calibration_batches is not None or max_test_batches is not None

    configured_result_path = Path(paths["result_path"])
    configured_threshold_path = Path(paths["threshold_result_path"])
    configured_figure_path = Path(paths["figure_path"])
    result_path = resolve_result_path(
        configured_result_path,
        is_smoke,
        result_path_overridden,
    )
    threshold_result_path = resolve_result_path(
        configured_threshold_path,
        is_smoke,
        threshold_result_path_overridden,
    )
    figure_path = resolve_result_path(
        configured_figure_path,
        is_smoke,
        figure_path_overridden,
    )
    log_path = Path(paths["log_path"])
    checkpoint_path = Path(paths["checkpoint_path"])

    logger = configure_logger("int4_mse_selected_ptq", log_path)

    seed = int(experiment_config["seed"])
    set_seed(seed, deterministic=bool(experiment_config.get("deterministic", True)))
    device = resolve_device(str(experiment_config.get("device", "auto")))

    activation_config = quant_config["activation"]
    weight_config = quant_config["weight"]
    _validate_int4_post_relu_activation_config(activation_config)
    _validate_mse_selected_config(activation_config)
    _validate_int4_weight_config(weight_config)

    activation_dtype = str(activation_config.get("dtype", "uint4")).lower()
    activation_qmin, activation_qmax = resolve_activation_range(activation_config)
    weight_qmin = int(weight_config.get("qmin", WEIGHT_INT4_QMIN))
    weight_qmax = int(weight_config.get("qmax", WEIGHT_INT4_QMAX))
    candidate_percentiles = MSE_SELECTED_PERCENTILES

    logger.info(
        "Starting INT4-MSE-Selected PTQ | model=%s | seed=%s | device=%s | smoke=%s",
        model_config["name"],
        seed,
        device,
        is_smoke,
    )
    logger.info(
        (
            "Quant ranges | weight=[%s,%s] | activation_dtype=%s | "
            "activation=[%s,%s] | candidates=%s"
        ),
        weight_qmin,
        weight_qmax,
        activation_dtype,
        activation_qmin,
        activation_qmax,
        _format_percentile_list(candidate_percentiles),
    )

    loaders, sizes, calibration_indices = build_cifar10_ptq_loaders(
        config,
        seed=seed,
        device=device,
    )
    calibration_index_checksum = compute_index_checksum(calibration_indices)
    logger.info(
        "Dataset | calibration_source=CIFAR10 train=True | calibration_size=%s | "
        "test_source=CIFAR10 train=False | test_size=%s | calibration_index_checksum=%s",
        sizes["calibration_size"],
        sizes["test_size"],
        calibration_index_checksum,
    )

    fp32_model = build_model(
        str(model_config["name"]),
        num_classes=int(model_config.get("num_classes", 10)),
    ).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model=fp32_model, map_location=device)
    checkpoint_model_name = str(checkpoint.get("model_name", ""))
    if checkpoint_model_name and checkpoint_model_name != str(model_config["name"]):
        raise ValueError(
            "Checkpoint model_name mismatch: "
            f"checkpoint={checkpoint_model_name}, config={model_config['name']}."
        )
    fp32_model.eval()
    logger.info(
        "Loaded checkpoint | path=%s | checkpoint_model_name=%s",
        checkpoint_path,
        checkpoint_model_name,
    )

    fp32_metrics = evaluate_top1(
        fp32_model,
        loaders["test"],
        device,
        max_batches=max_test_batches,
        split_name="fp32_test",
    )
    logger.info("Recomputed FP32 top1_accuracy=%.4f", fp32_metrics["top1_accuracy"])

    int4_model = copy.deepcopy(fp32_model)
    int4_model.eval()

    calibration = calibrate_post_relu_activation_mse_selected(
        int4_model,
        loaders["calibration"],
        device,
        candidate_percentiles=candidate_percentiles,
        qmin=activation_qmin,
        qmax=activation_qmax,
        max_batches=max_calibration_batches,
    )
    logger.info(
        "Observed post-ReLU activation sites | count=%s | names=%s",
        len(calibration.observed_site_names),
        ", ".join(calibration.observed_site_names),
    )

    write_threshold_results_csv(
        threshold_result_path,
        calibration.layer_results,
        activation_qmin=activation_qmin,
        activation_qmax=activation_qmax,
    )
    write_layerwise_mse_figure(figure_path, calibration.layer_results)
    logger.info("Wrote threshold CSV | path=%s", threshold_result_path)
    logger.info("Wrote layer-wise MSE figure | path=%s", figure_path)

    weight_result = apply_weight_fake_quantization(
        int4_model,
        qmin=weight_qmin,
        qmax=weight_qmax,
    )
    wrapped_site_names = attach_post_relu_activation_fake_quantization(
        int4_model,
        calibration.qparams_by_name,
    )
    if wrapped_site_names != calibration.observed_site_names:
        raise RuntimeError("Observed activation site names do not match wrapped sites.")

    int4_metrics = evaluate_top1(
        int4_model,
        loaders["test"],
        device,
        max_batches=max_test_batches,
        split_name="int4_mse_selected_test",
    )
    activation_mse = compute_activation_reconstruction_mse(
        fp32_model,
        loaders["test"],
        calibration.qparams_by_name,
        device,
        max_batches=max_test_batches,
    )
    logit_mse = compute_logit_mse(
        fp32_model,
        int4_model,
        loaders["test"],
        device,
        max_batches=max_test_batches,
    )

    if not is_smoke:
        expected_test_size = int(sizes["test_size"])
        fp32_evaluated_size = int(fp32_metrics["evaluated_size"])
        int4_evaluated_size = int(int4_metrics["evaluated_size"])
        if expected_test_size != 10000:
            raise RuntimeError(
                f"Expected full CIFAR-10 test_size=10000, got {expected_test_size}."
            )
        if fp32_evaluated_size != expected_test_size:
            raise RuntimeError(
                "FP32 evaluation did not cover the full CIFAR-10 test set: "
                f"{fp32_evaluated_size}/{expected_test_size}."
            )
        if int4_evaluated_size != expected_test_size:
            raise RuntimeError(
                "INT4-MSE-Selected evaluation did not cover the full CIFAR-10 "
                f"test set: {int4_evaluated_size}/{expected_test_size}."
            )

    accuracy_drop = fp32_metrics["top1_accuracy"] - int4_metrics["top1_accuracy"]
    if not math.isfinite(activation_mse.mse) or activation_mse.mse < 0.0:
        raise FloatingPointError(f"Invalid activation_mse: {activation_mse.mse}.")
    if not math.isfinite(logit_mse) or logit_mse < 0.0:
        raise FloatingPointError(f"Invalid logit_mse: {logit_mse}.")

    selected_alphas = [result.selected_alpha for result in calibration.layer_results]
    selected_mses = [result.selected_mse for result in calibration.layer_results]
    min_selected_alpha = min(selected_alphas)
    max_selected_alpha = max(selected_alphas)
    min_selected_mse = min(selected_mses)
    max_selected_mse = max(selected_mses)
    mean_selected_mse = sum(selected_mses) / len(selected_mses)

    logger.info(
        (
            "INT4-MSE-Selected result | fp32_top1=%.4f | int4_top1=%.4f | "
            "accuracy_drop=%.4f | activation_mse=%.8f | logit_mse=%.8f | "
            "selected_alpha_range=[%.8g, %.8g]"
        ),
        fp32_metrics["top1_accuracy"],
        int4_metrics["top1_accuracy"],
        accuracy_drop,
        activation_mse.mse,
        logit_mse,
        min_selected_alpha,
        max_selected_alpha,
    )
    print(f"top1_accuracy={int4_metrics['top1_accuracy']:.4f}")
    print(f"accuracy_drop={accuracy_drop:.4f}")
    print(f"activation_mse={activation_mse.mse:.8f}")
    print(f"logit_mse={logit_mse:.8f}")

    result_row = {
        "method": quant_config.get("method", "INT4-MSE-Selected"),
        "model": model_config["name"],
        "dataset": dataset_config["name"],
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_model_name": checkpoint_model_name,
        "calibration_size": sizes["calibration_size"],
        "calibration_seed": seed,
        "calibration_source": "CIFAR10 train=True",
        "calibration_num_batches": calibration.calibration_num_batches,
        "calibration_index_checksum": calibration_index_checksum,
        "test_size": sizes["test_size"],
        "evaluated_test_size": int(int4_metrics["evaluated_size"]),
        "top1_accuracy": f"{int4_metrics['top1_accuracy']:.4f}",
        "fp32_top1_accuracy": f"{fp32_metrics['top1_accuracy']:.4f}",
        "int4_top1_accuracy": f"{int4_metrics['top1_accuracy']:.4f}",
        "accuracy_drop": f"{accuracy_drop:.4f}",
        "activation_mse": f"{activation_mse.mse:.8f}",
        "logit_mse": f"{logit_mse:.8f}",
        "activation_quant_dtype": activation_dtype,
        "activation_qmin": activation_qmin,
        "activation_qmax": activation_qmax,
        "weight_qmin": weight_qmin,
        "weight_qmax": weight_qmax,
        "num_observed_activation_sites": len(calibration.observed_site_names),
        "num_quantized_modules": weight_result.num_quantized_modules,
        "min_activation_scale": f"{calibration.min_activation_scale:.10g}",
        "max_activation_scale": f"{calibration.max_activation_scale:.10g}",
        "min_activation_zero_point": calibration.min_activation_zero_point,
        "max_activation_zero_point": calibration.max_activation_zero_point,
        "min_weight_scale": f"{weight_result.min_weight_scale:.10g}",
        "max_weight_scale": f"{weight_result.max_weight_scale:.10g}",
        "is_smoke": str(is_smoke).lower(),
        "result_path": str(result_path),
        "threshold_result_path": str(threshold_result_path),
        "figure_path": str(figure_path),
        "log_path": str(log_path),
        "device": str(device),
        "batch_size": experiment_config["batch_size"],
        "observed_activation_sites": ";".join(calibration.observed_site_names),
        "activation_clip_method": activation_config.get("clip_method", "mse_selected"),
        "activation_clip_source": activation_config.get(
            "clip_max_source",
            "calibration_mse_selected_percentile",
        ),
        "activation_clip_min": activation_config.get("clip_min", 0),
        "activation_granularity": "per_tensor_per_relu_module",
        "activation_site_type": activation_config.get("site", "post_relu"),
        "threshold_search": "true",
        "mse_selected": "true",
        "candidate_percentiles": _format_percentile_list(candidate_percentiles),
        "min_selected_activation_alpha": f"{min_selected_alpha:.10g}",
        "max_selected_activation_alpha": f"{max_selected_alpha:.10g}",
        "min_selected_layer_mse": f"{min_selected_mse:.10g}",
        "max_selected_layer_mse": f"{max_selected_mse:.10g}",
        "mean_selected_layer_mse": f"{mean_selected_mse:.10g}",
        "weight_granularity": weight_config.get("granularity", "per_channel"),
        "weight_symmetry": weight_config.get("symmetry", "symmetric"),
    }
    write_single_row_csv(result_path, result_row, RESULT_COLUMNS)
    logger.info("Wrote result CSV | path=%s", result_path)


def write_threshold_results_csv(
    path: str | Path,
    layer_results: tuple[LayerClippingSearchResult, ...],
    activation_qmin: int,
    activation_qmax: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for layer_index, result in enumerate(layer_results):
        candidate_by_percentile = {
            candidate.percentile: candidate for candidate in result.candidate_mses
        }
        row = {
            "layer_name": result.layer_name,
            "layer_index": layer_index,
            "num_activation_elements": result.num_activation_elements,
            "selected_percentile": _format_percentile(result.selected_percentile),
            "selected_alpha": f"{result.selected_alpha:.10g}",
            "selected_mse": f"{result.selected_mse:.10g}",
            "activation_qmin": activation_qmin,
            "activation_qmax": activation_qmax,
            "activation_zero_point": result.qparams.zero_point,
            "activation_scale": f"{result.qparams.scale:.10g}",
            "candidate_percentiles": _format_percentile_list(MSE_SELECTED_PERCENTILES),
            "candidate_alphas": ";".join(
                f"{candidate.alpha:.10g}" for candidate in result.candidate_mses
            ),
            "candidate_mses": ";".join(
                f"{candidate.mse:.10g}" for candidate in result.candidate_mses
            ),
        }
        for percentile in MSE_SELECTED_PERCENTILES:
            column_prefix = _percentile_column_prefix(percentile)
            candidate = candidate_by_percentile[percentile]
            row[f"{column_prefix}_alpha"] = f"{candidate.alpha:.10g}"
            row[f"{column_prefix}_mse"] = f"{candidate.mse:.10g}"
        rows.append(row)
    write_rows_csv(path, rows, THRESHOLD_COLUMNS)


def write_layerwise_mse_figure(
    path: str | Path,
    layer_results: tuple[LayerClippingSearchResult, ...],
) -> None:
    figure_path = Path(path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions = list(range(len(layer_results)))
    selected_mses = [result.selected_mse for result in layer_results]
    selected_percentiles = [
        _format_percentile(result.selected_percentile) for result in layer_results
    ]
    layer_labels = [result.layer_name for result in layer_results]
    width = max(8.0, min(18.0, 0.45 * max(1, len(layer_results)) + 4.0))
    fig, ax = plt.subplots(figsize=(width, 4.8))
    ax.bar(x_positions, selected_mses, color="#2f6f9f")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Selected calibration reconstruction MSE")
    ax.set_title("INT4-MSE-Selected layer-wise activation MSE")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    for x_position, mse, percentile in zip(
        x_positions,
        selected_mses,
        selected_percentiles,
    ):
        ax.annotate(
            f"P{percentile}",
            xy=(x_position, mse),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def _validate_mse_selected_config(activation_config: dict[str, Any]) -> None:
    clip_method = str(activation_config.get("clip_method", "mse_selected"))
    if clip_method != "mse_selected":
        raise ValueError(
            f"INT4-MSE-Selected requires clip_method=mse_selected, got {clip_method}."
        )
    configured_percentiles = tuple(
        float(percentile)
        for percentile in activation_config.get(
            "candidate_percentiles",
            MSE_SELECTED_PERCENTILES,
        )
    )
    if configured_percentiles != MSE_SELECTED_PERCENTILES:
        raise ValueError(
            "INT4-MSE-Selected candidate_percentiles must be exactly "
            f"{MSE_SELECTED_PERCENTILES}."
        )


def _format_percentile_list(percentiles: tuple[float, ...]) -> str:
    return ";".join(_format_percentile(percentile) for percentile in percentiles)


def _format_percentile(percentile: float) -> str:
    text = f"{percentile:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def _percentile_column_prefix(percentile: float) -> str:
    return f"p{_format_percentile(percentile).replace('.', '_')}"


def main() -> None:
    args = parse_args()
    (
        config,
        result_path_overridden,
        threshold_result_path_overridden,
        figure_path_overridden,
    ) = apply_cli_overrides(load_config(args.config), args)
    run(
        config,
        result_path_overridden=result_path_overridden,
        threshold_result_path_overridden=threshold_result_path_overridden,
        figure_path_overridden=figure_path_overridden,
    )


if __name__ == "__main__":
    main()
