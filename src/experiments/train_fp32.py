from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models import build_model
from src.utils.checkpoint import load_checkpoint, save_checkpoint
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
    "epochs",
    "best_epoch",
    "val_top1_accuracy",
    "top1_accuracy",
    "checkpoint_path",
    "result_path",
    "log_path",
    "status",
    "fallback_reason",
    "device",
    "batch_size",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "train_size",
    "val_size",
    "test_size",
    "evaluated_test_size",
    "test_num_batches",
    "is_smoke",
    "max_train_batches",
    "max_val_batches",
    "max_test_batches",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an FP32 CIFAR-10 baseline.")
    parser.add_argument(
        "--config",
        default="configs/fp32_cifar10.yaml",
        help="Path to the FP32 YAML config.",
    )
    parser.add_argument("--model", choices=["resnet18_cifar", "compact_cnn"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="Use auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--result-path")
    parser.add_argument("--log-path")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    parser.add_argument("--status", choices=["ok", "fallback"], default="ok")
    parser.add_argument("--fallback-reason", default="")
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
) -> tuple[dict[str, Any], bool, bool, bool]:
    resolved = copy.deepcopy(config)
    checkpoint_path_overridden = args.checkpoint_path is not None
    result_path_overridden = args.result_path is not None
    log_path_overridden = args.log_path is not None

    if args.model is not None:
        resolved["model"]["name"] = args.model
    if args.epochs is not None:
        resolved["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        resolved["training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        resolved["training"]["seed"] = args.seed
    if args.device is not None:
        resolved["training"]["device"] = args.device
    if args.num_workers is not None:
        resolved["dataset"]["num_workers"] = args.num_workers
    if args.learning_rate is not None:
        resolved["training"]["optimizer"]["learning_rate"] = args.learning_rate
    if args.checkpoint_path is not None:
        resolved["paths"]["checkpoint_path"] = args.checkpoint_path
    if args.result_path is not None:
        resolved["paths"]["result_path"] = args.result_path
    if args.log_path is not None:
        resolved["paths"]["log_path"] = args.log_path

    smoke_config = resolved.setdefault("smoke", {})
    if args.max_train_batches is not None:
        smoke_config["max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        smoke_config["max_val_batches"] = args.max_val_batches
    if args.max_test_batches is not None:
        smoke_config["max_test_batches"] = args.max_test_batches

    return (
        resolved,
        checkpoint_path_overridden,
        result_path_overridden,
        log_path_overridden,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is not available.")
    return device


def build_cifar10_dataloaders(
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    dataset_config = config["dataset"]
    training_config = config["training"]
    data_dir = dataset_config["data_dir"]
    batch_size = int(training_config["batch_size"])
    num_workers = int(dataset_config.get("num_workers", 0))
    val_size = int(dataset_config.get("val_size", 5000))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_source = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_source = datasets.CIFAR10(
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

    total_train = len(train_source)
    if val_size <= 0 or val_size >= total_train:
        raise ValueError(f"val_size must be in [1, {total_train - 1}], got {val_size}.")

    split_generator = make_generator(seed)
    indices = torch.randperm(total_train, generator=split_generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": make_worker_init_fn(seed),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=make_generator(seed + 1),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        generator=make_generator(seed + 2),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        generator=make_generator(seed + 3),
        **loader_kwargs,
    )

    sizes = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
    return {"train": train_loader, "val": val_loader, "test": test_loader}, sizes


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> Optimizer:
    optimizer_config = config["training"]["optimizer"]
    name = optimizer_config.get("name", "SGD").lower()
    learning_rate = float(
        optimizer_config.get("learning_rate", optimizer_config.get("lr", 0.1))
    )
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(optimizer_config.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_config.get('name')}'.")


def build_scheduler(
    optimizer: Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_config = config["training"].get("scheduler", {"name": "none"})
    name = str(scheduler_config.get("name", "none")).lower()
    if name in {"none", "null"}:
        return None
    if name == "multisteplr":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(scheduler_config.get("milestones", [])),
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config.get("t_max", config["training"]["epochs"])),
        )
    raise ValueError(f"Unsupported scheduler '{scheduler_config.get('name')}'.")


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss: {loss_value}")

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_meter.update(loss_value, batch_size)
        accuracy_meter.update(top1_accuracy(logits.detach(), targets), batch_size)

    if loss_meter.count == 0:
        raise ValueError("Training loop processed zero batches.")

    return {
        "loss": loss_meter.average,
        "top1_accuracy": accuracy_meter.average,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
    split_name: str,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite {split_name} loss: {loss_value}")

        batch_size = targets.size(0)
        loss_meter.update(loss_value, batch_size)
        accuracy_meter.update(top1_accuracy(logits, targets), batch_size)
        num_batches += 1

    if loss_meter.count == 0:
        raise ValueError(f"{split_name} loop processed zero batches.")

    return {
        "loss": loss_meter.average,
        "top1_accuracy": accuracy_meter.average,
        "evaluated_size": float(accuracy_meter.count),
        "num_batches": float(num_batches),
    }


def run(
    config: dict[str, Any],
    status: str,
    fallback_reason: str,
    checkpoint_path_overridden: bool = False,
    result_path_overridden: bool = False,
    log_path_overridden: bool = False,
) -> None:
    if status == "fallback" and not fallback_reason.strip():
        raise ValueError("--fallback-reason is required when --status fallback is used.")

    training_config = config["training"]
    model_config = config["model"]
    dataset_config = config["dataset"]
    optimizer_config = training_config["optimizer"]
    scheduler_config = training_config.get("scheduler", {"name": "none"})
    smoke_config = config.get("smoke", {})

    max_train_batches = smoke_config.get("max_train_batches")
    max_val_batches = smoke_config.get("max_val_batches")
    max_test_batches = smoke_config.get("max_test_batches")
    is_smoke = (
        max_train_batches is not None
        or max_val_batches is not None
        or max_test_batches is not None
    )

    paths = config["paths"]
    log_path = resolve_artifact_path(
        Path(paths["log_path"]),
        is_smoke=is_smoke,
        path_overridden=log_path_overridden,
    )
    checkpoint_path = resolve_artifact_path(
        Path(paths["checkpoint_path"]),
        is_smoke=is_smoke,
        path_overridden=checkpoint_path_overridden,
    )
    result_path = resolve_artifact_path(
        Path(paths["result_path"]),
        is_smoke=is_smoke,
        path_overridden=result_path_overridden,
    )

    logger = configure_logger("fp32_cifar10", log_path)

    seed = int(training_config["seed"])
    epochs = int(training_config["epochs"])
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}.")

    set_seed(seed, deterministic=bool(training_config.get("deterministic", True)))
    device = resolve_device(str(training_config.get("device", "auto")))

    logger.info(
        "Starting FP32 CIFAR-10 run | model=%s | seed=%s | device=%s | epochs=%s",
        model_config["name"],
        seed,
        device,
        epochs,
    )
    logger.info(
        "Smoke controls | is_smoke=%s | max_train_batches=%s | max_val_batches=%s | "
        "max_test_batches=%s",
        is_smoke,
        max_train_batches,
        max_val_batches,
        max_test_batches,
    )

    loaders, sizes = build_cifar10_dataloaders(config, seed=seed, device=device)
    logger.info(
        "Dataset split | train=%s | val=%s | test=%s",
        sizes["train_size"],
        sizes["val_size"],
        sizes["test_size"],
    )

    model = build_model(
        str(model_config["name"]),
        num_classes=int(model_config.get("num_classes", 10)),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    best_val_top1 = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            max_batches=max_train_batches,
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            max_batches=max_val_batches,
            split_name="validation",
        )
        current_lr = float(optimizer.param_groups[0]["lr"])

        logger.info(
            (
                "epoch=%d | train_loss=%.6f | train_top1_accuracy=%.4f | "
                "val_loss=%.6f | val_top1_accuracy=%.4f | lr=%.8f"
            ),
            epoch,
            train_metrics["loss"],
            train_metrics["top1_accuracy"],
            val_metrics["loss"],
            val_metrics["top1_accuracy"],
            current_lr,
        )

        if val_metrics["top1_accuracy"] > best_val_top1:
            best_val_top1 = val_metrics["top1_accuracy"]
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_top1_accuracy=best_val_top1,
                seed=seed,
                model_name=str(model_config["name"]),
                config=config,
            )
            logger.info(
                "Saved best checkpoint | epoch=%d | val_top1_accuracy=%.4f | path=%s",
                epoch,
                best_val_top1,
                checkpoint_path,
            )

        if scheduler is not None:
            scheduler.step()

    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    best_epoch = int(checkpoint["epoch"])
    best_val_top1 = float(checkpoint["val_top1_accuracy"])
    logger.info(
        "Reloaded best checkpoint | epoch=%d | val_top1_accuracy=%.4f",
        best_epoch,
        best_val_top1,
    )

    test_metrics = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        max_batches=max_test_batches,
        split_name="test",
    )
    logger.info("Final test top1_accuracy=%.4f", test_metrics["top1_accuracy"])
    print(f"top1_accuracy={test_metrics['top1_accuracy']:.4f}")

    learning_rate = float(
        optimizer_config.get("learning_rate", optimizer_config.get("lr", 0.1))
    )
    result_row = {
        "method": "FP32",
        "model": model_config["name"],
        "dataset": dataset_config["name"],
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "val_top1_accuracy": f"{best_val_top1:.4f}",
        "top1_accuracy": f"{test_metrics['top1_accuracy']:.4f}",
        "checkpoint_path": str(checkpoint_path),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "status": status,
        "fallback_reason": fallback_reason if status == "fallback" else "",
        "device": str(device),
        "batch_size": training_config["batch_size"],
        "optimizer": optimizer_config.get("name", "SGD"),
        "learning_rate": learning_rate,
        "weight_decay": optimizer_config.get("weight_decay", 0.0),
        "scheduler": scheduler_config.get("name", "none"),
        "train_size": sizes["train_size"],
        "val_size": sizes["val_size"],
        "test_size": sizes["test_size"],
        "evaluated_test_size": int(test_metrics["evaluated_size"]),
        "test_num_batches": int(test_metrics["num_batches"]),
        "is_smoke": str(is_smoke).lower(),
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "max_test_batches": max_test_batches,
    }
    write_single_row_csv(result_path, result_row, RESULT_COLUMNS)
    logger.info("Wrote result CSV | path=%s", result_path)


def resolve_artifact_path(
    configured_path: Path,
    is_smoke: bool,
    path_overridden: bool,
) -> Path:
    if not is_smoke or path_overridden:
        return configured_path
    suffix = configured_path.suffix or ".out"
    return configured_path.with_name(f"{configured_path.stem}_smoke{suffix}")


def main() -> None:
    args = parse_args()
    (
        config,
        checkpoint_path_overridden,
        result_path_overridden,
        log_path_overridden,
    ) = apply_cli_overrides(load_config(args.config), args)
    run(
        config,
        status=args.status,
        fallback_reason=args.fallback_reason,
        checkpoint_path_overridden=checkpoint_path_overridden,
        result_path_overridden=result_path_overridden,
        log_path_overridden=log_path_overridden,
    )


if __name__ == "__main__":
    main()
