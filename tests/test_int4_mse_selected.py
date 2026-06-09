import csv
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments import run_int4_mse_selected_ptq
from src.quant.clipping_search import MSE_SELECTED_PERCENTILES


def test_validate_mse_selected_config_rejects_wrong_method_or_candidates() -> None:
    valid_config = {
        "clip_method": "mse_selected",
        "candidate_percentiles": [99.0, 99.5, 99.9, 99.95, 100.0],
    }
    run_int4_mse_selected_ptq._validate_mse_selected_config(valid_config)

    with pytest.raises(ValueError, match="mse_selected"):
        run_int4_mse_selected_ptq._validate_mse_selected_config(
            {**valid_config, "clip_method": "fixed_percentile"}
        )
    with pytest.raises(ValueError, match="candidate_percentiles"):
        run_int4_mse_selected_ptq._validate_mse_selected_config(
            {**valid_config, "candidate_percentiles": [99.9, 100.0]}
        )


def test_mse_selected_smoke_run_writes_results_thresholds_and_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_path = tmp_path / "int4_mse_selected_result.csv"
    threshold_path = tmp_path / "mse_selected_thresholds.csv"
    figure_path = tmp_path / "layerwise_mse.png"
    log_path = tmp_path / "int4_mse_selected_ptq.log"
    config = _build_smoke_config(result_path, threshold_path, figure_path, log_path)

    def build_tiny_model(_name: str, num_classes: int = 10) -> nn.Module:
        return _TinyClassifier(num_classes=num_classes).eval()

    def load_fake_checkpoint(
        _path: Path,
        model: nn.Module,
        map_location: torch.device,
    ) -> dict[str, str]:
        model.to(map_location)
        return {"model_name": "compact_cnn"}

    calibration_loader = _build_loader(torch.rand(4, 3, 8, 8), batch_size=2)
    test_loader = _build_loader(torch.rand(4, 3, 8, 8), batch_size=2)

    def build_fake_loaders(
        _config: dict[str, object],
        seed: int,
        device: torch.device,
    ) -> tuple[dict[str, DataLoader], dict[str, int], list[int]]:
        assert seed == 7
        assert device == torch.device("cpu")
        return (
            {"calibration": calibration_loader, "test": test_loader},
            {"calibration_size": 4, "test_size": 4},
            [3, 1, 2, 0],
        )

    monkeypatch.setattr(run_int4_mse_selected_ptq, "build_model", build_tiny_model)
    monkeypatch.setattr(run_int4_mse_selected_ptq, "load_checkpoint", load_fake_checkpoint)
    monkeypatch.setattr(
        run_int4_mse_selected_ptq,
        "build_cifar10_ptq_loaders",
        build_fake_loaders,
    )

    run_int4_mse_selected_ptq.run(config)
    captured = capsys.readouterr()

    smoke_result_path = tmp_path / "int4_mse_selected_result_smoke.csv"
    smoke_threshold_path = tmp_path / "mse_selected_thresholds_smoke.csv"
    smoke_figure_path = tmp_path / "layerwise_mse_smoke.png"

    assert "top1_accuracy=" in captured.out
    assert "accuracy_drop=" in captured.out
    assert "activation_mse=" in captured.out
    assert "logit_mse=" in captured.out
    assert smoke_result_path.exists()
    assert smoke_threshold_path.exists()
    assert smoke_figure_path.exists()
    assert smoke_figure_path.stat().st_size > 0

    with smoke_result_path.open("r", newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    with smoke_threshold_path.open("r", newline="", encoding="utf-8") as handle:
        threshold_rows = list(csv.DictReader(handle))

    assert len(result_rows) == 1
    result_row = result_rows[0]
    assert result_row["method"] == "INT4-MSE-Selected"
    assert result_row["activation_clip_method"] == "mse_selected"
    assert result_row["activation_clip_source"] == "calibration_mse_selected_percentile"
    assert result_row["threshold_search"] == "true"
    assert result_row["mse_selected"] == "true"
    assert result_row["candidate_percentiles"] == "99.0;99.5;99.9;99.95;100.0"
    assert result_row["calibration_source"] == "CIFAR10 train=True"
    assert result_row["is_smoke"] == "true"
    assert result_row["result_path"] == str(smoke_result_path)
    assert result_row["threshold_result_path"] == str(smoke_threshold_path)
    assert result_row["figure_path"] == str(smoke_figure_path)
    assert float(result_row["activation_mse"]) >= 0.0
    assert float(result_row["logit_mse"]) >= 0.0
    assert float(result_row["max_selected_activation_alpha"]) >= float(
        result_row["min_selected_activation_alpha"]
    )

    assert len(threshold_rows) == 1
    threshold_row = threshold_rows[0]
    assert threshold_row["layer_name"] == "features.2"
    assert threshold_row["selected_percentile"] in {
        _format_percentile(percentile) for percentile in MSE_SELECTED_PERCENTILES
    }
    assert threshold_row["selected_alpha"] != ""
    assert threshold_row["selected_mse"] != ""
    assert threshold_row["p99_0_mse"] != ""
    assert threshold_row["p99_5_mse"] != ""
    assert threshold_row["p99_9_mse"] != ""
    assert threshold_row["p99_95_mse"] != ""
    assert threshold_row["p100_0_mse"] != ""
    assert threshold_row["activation_qmin"] == "0"
    assert threshold_row["activation_qmax"] == "15"
    assert threshold_row["activation_zero_point"] == "0"


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(4 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _build_loader(inputs: torch.Tensor, batch_size: int = 2) -> DataLoader:
    targets = torch.zeros(inputs.size(0), dtype=torch.long)
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)


def _build_smoke_config(
    result_path: Path,
    threshold_path: Path,
    figure_path: Path,
    log_path: Path,
) -> dict[str, object]:
    return {
        "dataset": {
            "name": "CIFAR-10",
            "data_dir": "data",
            "calibration_size": 4,
            "num_workers": 0,
        },
        "model": {"name": "compact_cnn", "num_classes": 10},
        "experiment": {
            "seed": 7,
            "deterministic": True,
            "device": "cpu",
            "batch_size": 2,
        },
        "quantization": {
            "method": "INT4-MSE-Selected",
            "weight": {
                "dtype": "int4",
                "qmin": -7,
                "qmax": 7,
                "granularity": "per_channel",
                "symmetry": "symmetric",
                "channel_axis": 0,
            },
            "activation": {
                "dtype": "uint4",
                "qmin": 0,
                "qmax": 15,
                "granularity": "per_tensor_per_relu_module",
                "symmetry": "affine",
                "site": "post_relu",
                "clip_min": 0,
                "clip_method": "mse_selected",
                "candidate_percentiles": [99.0, 99.5, 99.9, 99.95, 100.0],
                "clip_max_source": "calibration_mse_selected_percentile",
            },
        },
        "paths": {
            "checkpoint_path": "checkpoints/fp32_best.pt",
            "result_path": str(result_path),
            "threshold_result_path": str(threshold_path),
            "figure_path": str(figure_path),
            "log_path": str(log_path),
        },
        "smoke": {"max_calibration_batches": 1, "max_test_batches": 1},
    }


def _format_percentile(percentile: float) -> str:
    text = f"{percentile:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text
