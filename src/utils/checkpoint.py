from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    val_top1_accuracy: float,
    seed: int,
    model_name: str,
    config: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_top1_accuracy": val_top1_accuracy,
            "seed": seed,
            "model_name": model_name,
            "config": config,
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
