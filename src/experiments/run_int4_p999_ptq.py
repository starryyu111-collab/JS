from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import torch

from src.experiments.run_int4_minmax_ptq import (
    apply_cli_overrides,
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
from src.quant.int4_p999 import (
    ACTIVATION_UINT4_QMAX,
    ACTIVATION_UINT4_QMIN,
    FIXED_ACTIVATION_PERCENTILE,
    FIXED_ACTIVATION_QUANTILE,
    WEIGHT_INT4_QMAX,
    WEIGHT_INT4_QMIN,
    apply_weight_fake_quantization,
    attach_post_relu_activation_fake_quantization,
    calibrate_post_relu_activation_p999,
    compute_activation_reconstruction_mse,
)
from src.utils.checkpoint import load_checkpoint
from src.utils.csv_io import write_single_row_csv
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
    "log_path",
    "device",
    "batch_size",
    "observed_activation_sites",
    "activation_clip_method",
    "activation_percentile",
    "activation_quantile",
    "activation_clip_source",
    "activation_clip_min",
    "activation_granularity",
    "activation_site_type",
    "threshold_search",
    "mse_selected",
    "candidate_percentiles",
    "min_activation_alpha",
    "max_activation_alpha",
    "weight_granularity",
    "weight_symmetry",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run INT4-P99.9 fixed-percentile PTQ for a CIFAR-10 checkpoint."
    )
    parser.add_argument(
        "--config",
        default="configs/int4_p999_cifar10.yaml",
        help="Path to the INT4-P99.9 PTQ YAML config.",
    )
    parser.add_argument("--model", choices=["resnet18_cifar", "compact_cnn"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="Use auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--calibration-size", type=int)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--result-path")
    parser.add_argument("--max-calibration-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    return parser.parse_args()


def run(config: dict[str, Any], result_path_overridden: bool = False) -> None:
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
    result_path = resolve_result_path(configured_result_path, is_smoke, result_path_overridden)
    log_path = Path(paths["log_path"])
    checkpoint_path = Path(paths["checkpoint_path"])

    logger = configure_logger("int4_p999_ptq", log_path)

    seed = int(experiment_config["seed"])
    set_seed(seed, deterministic=bool(experiment_config.get("deterministic", True)))
    device = resolve_device(str(experiment_config.get("device", "auto")))

    activation_config = quant_config["activation"]
    weight_config = quant_config["weight"]
    _validate_int4_post_relu_activation_config(activation_config)
    _validate_fixed_p999_config(activation_config)
    _validate_int4_weight_config(weight_config)

    activation_dtype = str(activation_config.get("dtype", "uint4")).lower()
    activation_qmin, activation_qmax = resolve_activation_range(activation_config)
    weight_qmin = int(weight_config.get("qmin", WEIGHT_INT4_QMIN))
    weight_qmax = int(weight_config.get("qmax", WEIGHT_INT4_QMAX))

    logger.info(
        "Starting INT4-P99.9 PTQ | model=%s | seed=%s | device=%s | smoke=%s",
        model_config["name"],
        seed,
        device,
        is_smoke,
    )
    logger.info(
        (
            "Quant ranges | weight=[%s,%s] | activation_dtype=%s | "
            "activation=[%s,%s] | percentile=%.1f"
        ),
        weight_qmin,
        weight_qmax,
        activation_dtype,
        activation_qmin,
        activation_qmax,
        FIXED_ACTIVATION_PERCENTILE,
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

    calibration = calibrate_post_relu_activation_p999(
        int4_model,
        loaders["calibration"],
        device,
        qmin=activation_qmin,
        qmax=activation_qmax,
        max_batches=max_calibration_batches,
    )
    logger.info(
        "Observed post-ReLU activation sites | count=%s | names=%s",
        len(calibration.observed_site_names),
        ", ".join(calibration.observed_site_names),
    )

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
        split_name="int4_p999_test",
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
                "INT4-P99.9 evaluation did not cover the full CIFAR-10 test set: "
                f"{int4_evaluated_size}/{expected_test_size}."
            )

    accuracy_drop = fp32_metrics["top1_accuracy"] - int4_metrics["top1_accuracy"]
    if not math.isfinite(activation_mse.mse) or activation_mse.mse < 0.0:
        raise FloatingPointError(f"Invalid activation_mse: {activation_mse.mse}.")
    if not math.isfinite(logit_mse) or logit_mse < 0.0:
        raise FloatingPointError(f"Invalid logit_mse: {logit_mse}.")

    activation_alphas = [params.clip_max for params in calibration.qparams_by_name.values()]
    min_activation_alpha = min(activation_alphas)
    max_activation_alpha = max(activation_alphas)

    logger.info(
        (
            "INT4-P99.9 result | fp32_top1=%.4f | int4_top1=%.4f | "
            "accuracy_drop=%.4f | activation_mse=%.8f | logit_mse=%.8f | "
            "alpha_range=[%.8g, %.8g]"
        ),
        fp32_metrics["top1_accuracy"],
        int4_metrics["top1_accuracy"],
        accuracy_drop,
        activation_mse.mse,
        logit_mse,
        min_activation_alpha,
        max_activation_alpha,
    )
    print(f"top1_accuracy={int4_metrics['top1_accuracy']:.4f}")
    print(f"accuracy_drop={accuracy_drop:.4f}")
    print(f"activation_mse={activation_mse.mse:.8f}")
    print(f"logit_mse={logit_mse:.8f}")

    result_row = {
        "method": quant_config.get("method", "INT4-P99.9"),
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
        "log_path": str(log_path),
        "device": str(device),
        "batch_size": experiment_config["batch_size"],
        "observed_activation_sites": ";".join(calibration.observed_site_names),
        "activation_clip_method": activation_config.get("clip_method", "fixed_percentile"),
        "activation_percentile": f"{FIXED_ACTIVATION_PERCENTILE:.1f}",
        "activation_quantile": f"{FIXED_ACTIVATION_QUANTILE:.3f}",
        "activation_clip_source": activation_config.get(
            "clip_max_source",
            "calibration_percentile",
        ),
        "activation_clip_min": activation_config.get("clip_min", 0),
        "activation_granularity": "per_tensor_per_relu_module",
        "activation_site_type": activation_config.get("site", "post_relu"),
        "threshold_search": "false",
        "mse_selected": "false",
        "candidate_percentiles": "",
        "min_activation_alpha": f"{min_activation_alpha:.10g}",
        "max_activation_alpha": f"{max_activation_alpha:.10g}",
        "weight_granularity": weight_config.get("granularity", "per_channel"),
        "weight_symmetry": weight_config.get("symmetry", "symmetric"),
    }
    write_single_row_csv(result_path, result_row, RESULT_COLUMNS)
    logger.info("Wrote result CSV | path=%s", result_path)


def _validate_fixed_p999_config(activation_config: dict[str, Any]) -> None:
    clip_method = str(activation_config.get("clip_method", "fixed_percentile"))
    if clip_method != "fixed_percentile":
        raise ValueError(f"INT4-P99.9 requires clip_method=fixed_percentile, got {clip_method}.")
    percentile = float(activation_config.get("percentile", FIXED_ACTIVATION_PERCENTILE))
    quantile = float(activation_config.get("quantile", FIXED_ACTIVATION_QUANTILE))
    if percentile != FIXED_ACTIVATION_PERCENTILE:
        raise ValueError("INT4-P99.9 requires activation percentile fixed at 99.9.")
    if quantile != FIXED_ACTIVATION_QUANTILE:
        raise ValueError("INT4-P99.9 requires activation quantile fixed at 0.999.")


def main() -> None:
    args = parse_args()
    config, result_path_overridden = apply_cli_overrides(load_config(args.config), args)
    run(config, result_path_overridden=result_path_overridden)


if __name__ == "__main__":
    main()
