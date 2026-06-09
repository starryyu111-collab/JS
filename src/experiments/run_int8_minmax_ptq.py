from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models import build_model
from src.quant.int8_minmax import (
    ACTIVATION_INT8_QMAX,
    ACTIVATION_INT8_QMIN,
    ACTIVATION_UINT8_QMAX,
    ACTIVATION_UINT8_QMIN,
    WEIGHT_INT8_QMAX,
    WEIGHT_INT8_QMIN,
    apply_weight_fake_quantization,
    attach_activation_fake_quantization,
    calibrate_activation_ranges,
    compute_activation_reconstruction_mse,
)
from src.utils.checkpoint import load_checkpoint
from src.utils.csv_io import write_single_row_csv
from src.utils.logging import configure_logger
from src.utils.metrics import AverageMeter, top1_accuracy
from src.utils.seed import make_generator, make_worker_init_fn, set_seed


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

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
    "int8_top1_accuracy",
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
    "activation_site_type",
    "activation_clip_method",
    "activation_clip_source",
    "activation_granularity",
    "weight_granularity",
    "weight_symmetry",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run INT8-MinMax PTQ for a CIFAR-10 checkpoint."
    )
    parser.add_argument(
        "--config",
        default="configs/int8_minmax_cifar10.yaml",
        help="Path to the INT8-MinMax PTQ YAML config.",
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


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must contain a YAML mapping.")
    return config


def apply_cli_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bool]:
    resolved = copy.deepcopy(config)
    result_path_overridden = args.result_path is not None

    if args.model is not None:
        resolved["model"]["name"] = args.model
    if args.batch_size is not None:
        resolved["experiment"]["batch_size"] = args.batch_size
    if args.seed is not None:
        resolved["experiment"]["seed"] = args.seed
    if args.device is not None:
        resolved["experiment"]["device"] = args.device
    if args.num_workers is not None:
        resolved["dataset"]["num_workers"] = args.num_workers
    if args.calibration_size is not None:
        resolved["dataset"]["calibration_size"] = args.calibration_size
    if args.checkpoint_path is not None:
        resolved["paths"]["checkpoint_path"] = args.checkpoint_path
    if args.result_path is not None:
        resolved["paths"]["result_path"] = args.result_path

    smoke_config = resolved.setdefault("smoke", {})
    if args.max_calibration_batches is not None:
        smoke_config["max_calibration_batches"] = args.max_calibration_batches
    if args.max_test_batches is not None:
        smoke_config["max_test_batches"] = args.max_test_batches

    return resolved, result_path_overridden


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is not available.")
    return device


def build_cifar10_ptq_loaders(
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, DataLoader], dict[str, int], list[int]]:
    dataset_config = config["dataset"]
    experiment_config = config["experiment"]
    data_dir = dataset_config["data_dir"]
    batch_size = int(experiment_config["batch_size"])
    num_workers = int(dataset_config.get("num_workers", 0))
    calibration_size = int(dataset_config.get("calibration_size", 1024))

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    calibration_source = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    total_train = len(calibration_source)
    if calibration_size <= 0 or calibration_size > total_train:
        raise ValueError(
            f"calibration_size must be in [1, {total_train}], got {calibration_size}."
        )

    indices = torch.randperm(total_train, generator=make_generator(seed)).tolist()
    calibration_indices = indices[:calibration_size]
    calibration_dataset = Subset(calibration_source, calibration_indices)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": make_worker_init_fn(seed),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    calibration_loader = DataLoader(
        calibration_dataset,
        shuffle=False,
        generator=make_generator(seed + 1),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        generator=make_generator(seed + 2),
        **loader_kwargs,
    )

    sizes = {
        "calibration_size": len(calibration_dataset),
        "test_size": len(test_dataset),
    }
    return {"calibration": calibration_loader, "test": test_loader}, sizes, calibration_indices


@torch.no_grad()
def evaluate_top1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    split_name: str,
) -> dict[str, float]:
    model.eval()
    accuracy_meter = AverageMeter()
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)

        batch_size = targets.size(0)
        accuracy_meter.update(top1_accuracy(logits, targets), batch_size)
        num_batches += 1

    if accuracy_meter.count == 0:
        raise ValueError(f"{split_name} loop processed zero batches.")

    return {
        "top1_accuracy": accuracy_meter.average,
        "evaluated_size": float(accuracy_meter.count),
        "num_batches": float(num_batches),
    }


@torch.no_grad()
def compute_logit_mse(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    fp32_model.eval()
    int8_model.eval()
    total_squared_error = 0.0
    total_elements = 0
    num_batches = 0

    for batch_idx, (inputs, _targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        fp32_logits = fp32_model(inputs)
        int8_logits = int8_model(inputs)
        diff = (fp32_logits - int8_logits).float()
        total_squared_error += float(torch.sum(diff * diff).item())
        total_elements += diff.numel()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Logit MSE processed zero batches.")
    if total_elements == 0:
        raise ValueError("Logit MSE saw zero logit elements.")

    return total_squared_error / total_elements


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

    logger = configure_logger("int8_minmax_ptq", log_path)

    seed = int(experiment_config["seed"])
    set_seed(seed, deterministic=bool(experiment_config.get("deterministic", True)))
    device = resolve_device(str(experiment_config.get("device", "auto")))

    activation_config = quant_config["activation"]
    weight_config = quant_config["weight"]
    _validate_int8_activation_config(activation_config)
    _validate_int8_weight_config(weight_config)
    activation_dtype = str(activation_config.get("dtype", "uint8")).lower()
    activation_qmin, activation_qmax = resolve_activation_range(activation_config)
    weight_qmin = int(weight_config.get("qmin", WEIGHT_INT8_QMIN))
    weight_qmax = int(weight_config.get("qmax", WEIGHT_INT8_QMAX))

    logger.info(
        "Starting INT8-MinMax PTQ | model=%s | seed=%s | device=%s | smoke=%s",
        model_config["name"],
        seed,
        device,
        is_smoke,
    )
    logger.info(
        "Quant ranges | weight=[%s,%s] | activation_dtype=%s | activation=[%s,%s]",
        weight_qmin,
        weight_qmax,
        activation_dtype,
        activation_qmin,
        activation_qmax,
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

    int8_model = copy.deepcopy(fp32_model)
    int8_model.eval()

    calibration = calibrate_activation_ranges(
        int8_model,
        loaders["calibration"],
        device,
        qmin=activation_qmin,
        qmax=activation_qmax,
        max_batches=max_calibration_batches,
    )
    logger.info(
        "Observed activation sites | count=%s | names=%s",
        len(calibration.observed_site_names),
        ", ".join(calibration.observed_site_names),
    )

    weight_result = apply_weight_fake_quantization(
        int8_model,
        qmin=weight_qmin,
        qmax=weight_qmax,
    )
    wrapped_site_names = attach_activation_fake_quantization(
        int8_model,
        calibration.qparams_by_name,
    )
    if wrapped_site_names != calibration.observed_site_names:
        raise RuntimeError("Observed activation site names do not match wrapped sites.")
    if weight_result.num_quantized_modules != len(calibration.observed_site_names):
        raise RuntimeError("Observed activation site count does not match weight targets.")

    int8_metrics = evaluate_top1(
        int8_model,
        loaders["test"],
        device,
        max_batches=max_test_batches,
        split_name="int8_test",
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
        int8_model,
        loaders["test"],
        device,
        max_batches=max_test_batches,
    )

    if not is_smoke:
        expected_test_size = int(sizes["test_size"])
        fp32_evaluated_size = int(fp32_metrics["evaluated_size"])
        int8_evaluated_size = int(int8_metrics["evaluated_size"])
        if expected_test_size != 10000:
            raise RuntimeError(
                f"Expected full CIFAR-10 test_size=10000, got {expected_test_size}."
            )
        if fp32_evaluated_size != expected_test_size:
            raise RuntimeError(
                "FP32 evaluation did not cover the full CIFAR-10 test set: "
                f"{fp32_evaluated_size}/{expected_test_size}."
            )
        if int8_evaluated_size != expected_test_size:
            raise RuntimeError(
                "INT8 evaluation did not cover the full CIFAR-10 test set: "
                f"{int8_evaluated_size}/{expected_test_size}."
            )

    accuracy_drop = fp32_metrics["top1_accuracy"] - int8_metrics["top1_accuracy"]
    if not math.isfinite(activation_mse.mse) or activation_mse.mse < 0.0:
        raise FloatingPointError(f"Invalid activation_mse: {activation_mse.mse}.")
    if not math.isfinite(logit_mse) or logit_mse < 0.0:
        raise FloatingPointError(f"Invalid logit_mse: {logit_mse}.")

    logger.info(
        (
            "INT8 result | fp32_top1=%.4f | int8_top1=%.4f | accuracy_drop=%.4f | "
            "activation_mse=%.8f | logit_mse=%.8f"
        ),
        fp32_metrics["top1_accuracy"],
        int8_metrics["top1_accuracy"],
        accuracy_drop,
        activation_mse.mse,
        logit_mse,
    )
    print(f"top1_accuracy={int8_metrics['top1_accuracy']:.4f}")
    print(f"accuracy_drop={accuracy_drop:.4f}")
    print(f"activation_mse={activation_mse.mse:.8f}")
    print(f"logit_mse={logit_mse:.8f}")

    result_row = {
        "method": quant_config.get("method", "INT8-MinMax"),
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
        "evaluated_test_size": int(int8_metrics["evaluated_size"]),
        "top1_accuracy": f"{int8_metrics['top1_accuracy']:.4f}",
        "fp32_top1_accuracy": f"{fp32_metrics['top1_accuracy']:.4f}",
        "int8_top1_accuracy": f"{int8_metrics['top1_accuracy']:.4f}",
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
        "activation_site_type": activation_config.get("site", "conv_linear_output"),
        "activation_clip_method": activation_config.get("clip_method", "minmax"),
        "activation_clip_source": activation_config.get(
            "clip_max_source",
            "calibration_minmax",
        ),
        "activation_granularity": activation_config.get("granularity", "per_tensor"),
        "weight_granularity": weight_config.get("granularity", "per_channel"),
        "weight_symmetry": weight_config.get("symmetry", "symmetric"),
    }
    write_single_row_csv(result_path, result_row, RESULT_COLUMNS)
    logger.info("Wrote result CSV | path=%s", result_path)


def resolve_activation_range(config: dict[str, Any]) -> tuple[int, int]:
    dtype = str(config.get("dtype", "uint8")).lower()
    if "qmin" in config and "qmax" in config:
        return int(config["qmin"]), int(config["qmax"])
    if dtype == "uint8":
        return ACTIVATION_UINT8_QMIN, ACTIVATION_UINT8_QMAX
    if dtype == "int8":
        return ACTIVATION_INT8_QMIN, ACTIVATION_INT8_QMAX
    raise ValueError(f"Unsupported activation dtype '{dtype}'.")


def _validate_int8_activation_config(config: dict[str, Any]) -> None:
    site = str(config.get("site", "conv_linear_output"))
    if site != "conv_linear_output":
        raise ValueError(
            "INT8-MinMax activation hooks are attached to Conv2d/Linear outputs; "
            f"expected site=conv_linear_output, got {site}."
        )

    clip_method = str(config.get("clip_method", "minmax"))
    if clip_method != "minmax":
        raise ValueError(
            f"INT8-MinMax requires clip_method=minmax, got {clip_method}."
        )


def _validate_int8_weight_config(config: dict[str, Any]) -> None:
    dtype = str(config.get("dtype", "int8")).lower()
    if dtype != "int8":
        raise ValueError(f"INT8 PTQ requires weight dtype=int8, got {dtype}.")

    qmin = int(config.get("qmin", WEIGHT_INT8_QMIN))
    qmax = int(config.get("qmax", WEIGHT_INT8_QMAX))
    if qmin != WEIGHT_INT8_QMIN or qmax != WEIGHT_INT8_QMAX:
        raise ValueError(
            "INT8 weight quantization range must be symmetric signed "
            f"[{WEIGHT_INT8_QMIN}, {WEIGHT_INT8_QMAX}], got [{qmin}, {qmax}]."
        )

    granularity = str(config.get("granularity", "per_channel"))
    if granularity != "per_channel":
        raise ValueError(
            f"INT8 weight quantization requires granularity=per_channel, got {granularity}."
        )

    symmetry = str(config.get("symmetry", "symmetric"))
    if symmetry != "symmetric":
        raise ValueError(
            f"INT8 weight quantization requires symmetry=symmetric, got {symmetry}."
        )

    channel_axis = int(config.get("channel_axis", 0))
    if channel_axis != 0:
        raise ValueError(
            "INT8 per-channel weight quantization uses output channel axis 0, "
            f"got channel_axis={channel_axis}."
        )


def resolve_result_path(
    configured_result_path: Path,
    is_smoke: bool,
    result_path_overridden: bool,
) -> Path:
    if not is_smoke or result_path_overridden:
        return configured_result_path
    suffix = configured_result_path.suffix or ".csv"
    return configured_result_path.with_name(f"{configured_result_path.stem}_smoke{suffix}")


def compute_index_checksum(indices: list[int]) -> int:
    checksum = 0
    for position, index in enumerate(indices, start=1):
        checksum = (checksum + position * int(index)) % 1_000_000_007
    return checksum


def main() -> None:
    args = parse_args()
    config, result_path_overridden = apply_cli_overrides(load_config(args.config), args)
    run(config, result_path_overridden=result_path_overridden)


if __name__ == "__main__":
    main()
