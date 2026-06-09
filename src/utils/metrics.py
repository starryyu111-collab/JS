from dataclasses import dataclass

import torch


@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return top-1 accuracy as a percentage in [0, 100]."""
    if targets.numel() == 0:
        return 0.0
    predictions = logits.argmax(dim=1)
    correct = predictions.eq(targets).sum().item()
    return 100.0 * correct / targets.numel()


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count
