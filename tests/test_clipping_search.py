import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.quant.clipping_search import (
    MSE_SELECTED_PERCENTILES,
    calibrate_post_relu_activation_mse_selected,
    select_mse_minimizing_activation_qparams,
)


def test_mse_selected_candidate_set_is_exact() -> None:
    assert MSE_SELECTED_PERCENTILES == (99.0, 99.5, 99.9, 99.95, 100.0)


def test_p100_candidate_equals_layer_max() -> None:
    values = torch.tensor([0.0, 1.0, 3.0, 7.0])

    result = select_mse_minimizing_activation_qparams("relu", values)

    p100_candidate = _candidate_by_percentile(result, 100.0)
    assert p100_candidate.alpha == pytest.approx(7.0)


def test_candidate_alpha_matches_torch_quantile_on_small_tensor() -> None:
    values = torch.linspace(0.0, 100.0, steps=101)

    result = select_mse_minimizing_activation_qparams("relu", values)

    p99_5_candidate = _candidate_by_percentile(result, 99.5)
    expected = torch.quantile(values, 0.995).item()
    assert p99_5_candidate.alpha == pytest.approx(expected)


def test_mse_selected_calibration_uses_layer_specific_distributions() -> None:
    model = _TwoReluModel().eval()
    inputs = torch.tensor(
        [
            [[[[0.0, 1.0], [2.0, 3.0]]]],
            [[[[4.0, 5.0], [6.0, 7.0]]]],
        ]
    ).reshape(2, 1, 2, 2)
    loader = _build_loader(inputs)

    calibration = calibrate_post_relu_activation_mse_selected(
        model,
        loader,
        torch.device("cpu"),
    )

    first_result, second_result = calibration.layer_results
    first_p100 = _candidate_by_percentile(first_result, 100.0)
    second_p100 = _candidate_by_percentile(second_result, 100.0)

    assert calibration.observed_site_names == ("relu_low", "relu_high")
    assert first_p100.alpha == pytest.approx(7.0)
    assert second_p100.alpha == pytest.approx(70.0)
    assert first_result.selected_alpha != second_result.selected_alpha
    assert tuple(calibration.qparams_by_name) == calibration.observed_site_names


def test_candidate_mses_are_finite_nonnegative_and_selected_minimum() -> None:
    values = torch.linspace(0.0, 10.0, steps=101)

    result = select_mse_minimizing_activation_qparams("relu", values)
    candidate_mses = [candidate.mse for candidate in result.candidate_mses]

    assert len(candidate_mses) == len(MSE_SELECTED_PERCENTILES)
    assert all(torch.isfinite(torch.tensor(mse)) for mse in candidate_mses)
    assert all(mse >= 0.0 for mse in candidate_mses)
    assert result.selected_mse == min(candidate_mses)


def test_tie_breaking_selects_first_candidate() -> None:
    values = torch.zeros(8)

    result = select_mse_minimizing_activation_qparams("relu", values)

    assert all(candidate.mse == 0.0 for candidate in result.candidate_mses)
    assert result.selected_percentile == MSE_SELECTED_PERCENTILES[0]
    assert result.qparams.clip_max == 0.0
    assert result.qparams.scale == 1.0


def test_selected_qparams_use_uint4_zero_point_and_batch_count() -> None:
    model = nn.Sequential(nn.ReLU()).eval()
    loader = _build_loader(torch.rand(4, 1, 2, 2), batch_size=2)

    calibration = calibrate_post_relu_activation_mse_selected(
        model,
        loader,
        torch.device("cpu"),
        qmin=0,
        qmax=15,
        max_batches=1,
    )

    params = calibration.qparams_by_name["0"]
    assert calibration.calibration_num_batches == 1
    assert calibration.layer_results[0].num_activation_elements == 8
    assert params.qmin == 0
    assert params.qmax == 15
    assert params.zero_point == 0
    assert calibration.min_activation_zero_point == 0
    assert calibration.max_activation_zero_point == 0


def test_reused_relu_module_remains_one_module_level_site() -> None:
    model = _ReusedReluModule().eval()
    loader = _build_loader(torch.rand(4, 1, 2, 2), batch_size=2)

    calibration = calibrate_post_relu_activation_mse_selected(
        model,
        loader,
        torch.device("cpu"),
    )

    assert calibration.observed_site_names == ("relu",)
    assert len(calibration.layer_results) == 1
    assert calibration.layer_results[0].num_activation_elements == 32


class _TwoReluModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu_low = nn.ReLU()
        self.relu_high = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.relu_low(x)
        high = self.relu_high(x * 10.0)
        return low + high


class _ReusedReluModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        return self.relu(x - 0.5)


def _candidate_by_percentile(result: object, percentile: float) -> object:
    for candidate in result.candidate_mses:
        if candidate.percentile == percentile:
            return candidate
    raise AssertionError(f"Missing percentile {percentile}.")


def _build_loader(inputs: torch.Tensor, batch_size: int = 2) -> DataLoader:
    targets = torch.zeros(inputs.size(0), dtype=torch.long)
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)
