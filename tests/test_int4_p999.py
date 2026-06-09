import csv
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments import run_int4_p999_ptq
from src.quant.int4_p999 import (
    ACTIVATION_UINT4_QMAX,
    ACTIVATION_UINT4_QMIN,
    FIXED_ACTIVATION_PERCENTILE,
    FIXED_ACTIVATION_QUANTILE,
    attach_post_relu_activation_fake_quantization,
    calibrate_post_relu_activation_p999,
    compute_empirical_quantile,
    fake_quantize_post_relu_activation,
    make_post_relu_activation_p999_qparams,
)


def test_fixed_percentile_constants_are_p999() -> None:
    assert FIXED_ACTIVATION_PERCENTILE == 99.9
    assert FIXED_ACTIVATION_QUANTILE == 0.999


def test_p999_qparams_use_uint4_post_relu_interval() -> None:
    qparams = make_post_relu_activation_p999_qparams(
        alpha=2.0,
        qmin=ACTIVATION_UINT4_QMIN,
        qmax=ACTIVATION_UINT4_QMAX,
    )
    tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0, 4.0])

    dequantized = fake_quantize_post_relu_activation(tensor, qparams)

    assert qparams.scale == 2.0 / 15.0
    assert qparams.zero_point == 0
    assert qparams.clip_min == 0.0
    assert qparams.clip_max == 2.0
    assert torch.isfinite(dequantized).all()
    assert dequantized.min().item() >= 0.0
    assert dequantized.max().item() <= 2.0


def test_p999_calibration_uses_layer_specific_elementwise_distribution() -> None:
    model = _TwoReluModel().eval()
    loader = _build_loader(
        torch.tensor(
            [
                [[[[0.0, 1.0], [2.0, 3.0]]]],
                [[[[4.0, 5.0], [6.0, 7.0]]]],
            ]
        ).reshape(2, 1, 2, 2)
    )

    calibration = calibrate_post_relu_activation_p999(
        model,
        loader,
        torch.device("cpu"),
    )

    first_alpha = calibration.qparams_by_name["relu_low"].clip_max
    second_alpha = calibration.qparams_by_name["relu_high"].clip_max
    expected_first = torch.quantile(torch.arange(8, dtype=torch.float32), 0.999).item()
    expected_second = torch.quantile(torch.arange(8, dtype=torch.float32) * 10.0, 0.999).item()

    assert calibration.observed_site_names == ("relu_low", "relu_high")
    assert first_alpha == pytest.approx(expected_first)
    assert second_alpha == pytest.approx(expected_second)
    assert first_alpha != second_alpha


def test_p999_calibration_does_not_average_batch_percentiles() -> None:
    model = nn.Sequential(nn.ReLU()).eval()
    inputs = torch.tensor([[[[0.0, 1.0]]], [[[1000.0, 1001.0]]]])
    loader = _build_loader(inputs, batch_size=1)

    calibration = calibrate_post_relu_activation_p999(
        model,
        loader,
        torch.device("cpu"),
    )

    alpha = calibration.qparams_by_name["0"].clip_max
    expected_global = torch.quantile(inputs.reshape(-1), 0.999).item()
    averaged_batch_percentiles = torch.stack(
        [torch.quantile(batch.reshape(-1), 0.999) for batch, _target in loader]
    ).mean()

    assert alpha == pytest.approx(expected_global)
    assert alpha != pytest.approx(float(averaged_batch_percentiles.item()))


def test_empirical_quantile_matches_torch_quantile_for_small_tensor() -> None:
    values = torch.tensor([5.0, 1.0, 3.0, 100.0, 8.0, 13.0, 21.0])

    quantile = compute_empirical_quantile(values, FIXED_ACTIVATION_QUANTILE)

    assert quantile.item() == pytest.approx(
        torch.quantile(values, FIXED_ACTIVATION_QUANTILE).item()
    )


def test_empirical_quantile_uses_exact_fallback_for_torch_size_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = torch.tensor([0.0, 10.0, 20.0, 30.0])

    def raise_size_limit(_values: torch.Tensor, _quantile: float) -> torch.Tensor:
        raise RuntimeError("quantile() input tensor is too large")

    monkeypatch.setattr(torch, "quantile", raise_size_limit)

    quantile = compute_empirical_quantile(values, FIXED_ACTIVATION_QUANTILE)

    assert quantile.item() == pytest.approx(29.97)


def test_empirical_quantile_bypasses_torch_quantile_for_known_large_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = torch.tensor([0.0, 10.0, 20.0, 30.0])

    def fail_if_called(_values: torch.Tensor, _quantile: float) -> torch.Tensor:
        raise AssertionError("large tensors should bypass torch.quantile")

    monkeypatch.setattr("src.quant.int4_p999.TORCH_QUANTILE_MAX_INPUT_ELEMENTS", 3)
    monkeypatch.setattr(torch, "quantile", fail_if_called)

    quantile = compute_empirical_quantile(values, FIXED_ACTIVATION_QUANTILE)

    assert quantile.item() == pytest.approx(29.97)


def test_empirical_quantile_reraises_unrelated_torch_quantile_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = torch.tensor([0.0, 1.0])

    def raise_unrelated(_values: torch.Tensor, _quantile: float) -> torch.Tensor:
        raise RuntimeError("some unrelated quantile failure")

    monkeypatch.setattr(torch, "quantile", raise_unrelated)

    with pytest.raises(RuntimeError, match="unrelated"):
        compute_empirical_quantile(values, FIXED_ACTIVATION_QUANTILE)


def test_p999_calibration_uses_exact_quantile_fallback_when_size_limit_is_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Sequential(nn.ReLU()).eval()
    inputs = torch.tensor([[[[0.0, 10.0]]], [[[20.0, 30.0]]]])
    loader = _build_loader(inputs, batch_size=2)

    def raise_size_limit(_values: torch.Tensor, _quantile: float) -> torch.Tensor:
        raise RuntimeError("quantile() input tensor is too large")

    monkeypatch.setattr(torch, "quantile", raise_size_limit)

    calibration = calibrate_post_relu_activation_p999(
        model,
        loader,
        torch.device("cpu"),
    )

    assert calibration.qparams_by_name["0"].clip_max == pytest.approx(29.97)


def test_p999_wrapper_targets_observed_relu_sites() -> None:
    model = _TwoReluModel().eval()
    loader = _build_loader(torch.rand(4, 1, 2, 2))
    calibration = calibrate_post_relu_activation_p999(
        model,
        loader,
        torch.device("cpu"),
    )

    wrapped_names = attach_post_relu_activation_fake_quantization(
        model,
        calibration.qparams_by_name,
    )

    assert wrapped_names == calibration.observed_site_names
    with torch.no_grad():
        output = model(torch.randn(2, 1, 2, 2))
    assert torch.isfinite(output).all()
    assert output.min().item() >= 0.0


def test_validate_fixed_p999_config_rejects_threshold_search_variants() -> None:
    valid_config = {
        "clip_method": "fixed_percentile",
        "percentile": 99.9,
        "quantile": 0.999,
    }
    run_int4_p999_ptq._validate_fixed_p999_config(valid_config)

    with pytest.raises(ValueError, match="fixed at 99.9"):
        run_int4_p999_ptq._validate_fixed_p999_config(
            {**valid_config, "percentile": 99.5}
        )
    with pytest.raises(ValueError, match="fixed_percentile"):
        run_int4_p999_ptq._validate_fixed_p999_config(
            {**valid_config, "clip_method": "mse_selected"}
        )


def test_p999_smoke_run_writes_fixed_percentile_result_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path = tmp_path / "int4_p999_result.csv"
    log_path = tmp_path / "int4_p999_ptq.log"
    config = _build_smoke_config(result_path, log_path)

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

    monkeypatch.setattr(run_int4_p999_ptq, "build_model", build_tiny_model)
    monkeypatch.setattr(run_int4_p999_ptq, "load_checkpoint", load_fake_checkpoint)
    monkeypatch.setattr(run_int4_p999_ptq, "build_cifar10_ptq_loaders", build_fake_loaders)

    run_int4_p999_ptq.run(config)

    smoke_path = tmp_path / "int4_p999_result_smoke.csv"
    assert smoke_path.exists()
    with smoke_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["method"] == "INT4-P99.9"
    assert row["activation_clip_method"] == "fixed_percentile"
    assert row["activation_percentile"] == "99.9"
    assert row["activation_quantile"] == "0.999"
    assert row["activation_clip_source"] == "calibration_percentile"
    assert row["threshold_search"] == "false"
    assert row["mse_selected"] == "false"
    assert row["candidate_percentiles"] == ""
    assert row["calibration_source"] == "CIFAR10 train=True"
    assert row["is_smoke"] == "true"
    assert row["result_path"] == str(smoke_path)
    assert float(row["activation_mse"]) >= 0.0
    assert float(row["logit_mse"]) >= 0.0
    assert float(row["max_activation_alpha"]) >= float(row["min_activation_alpha"])


class _TwoReluModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu_low = nn.ReLU()
        self.relu_high = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.relu_low(x)
        high = self.relu_high(x * 10.0)
        return low + high


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


def _build_smoke_config(result_path: Path, log_path: Path) -> dict[str, object]:
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
            "method": "INT4-P99.9",
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
                "clip_method": "fixed_percentile",
                "percentile": 99.9,
                "quantile": 0.999,
                "clip_max_source": "calibration_percentile",
            },
        },
        "paths": {
            "checkpoint_path": "checkpoints/fp32_best.pt",
            "result_path": str(result_path),
            "log_path": str(log_path),
        },
        "smoke": {"max_calibration_batches": 1, "max_test_batches": 1},
    }
