from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.quant.int4_minmax import (
    ACTIVATION_UINT4_QMAX,
    ACTIVATION_UINT4_QMIN,
    PostReluActivationQuantizationParams,
    fake_quantize_post_relu_activation,
    iter_named_post_relu_modules,
    make_post_relu_activation_qparams,
    validate_integer_range,
)


MSE_SELECTED_PERCENTILES = (99.0, 99.5, 99.9, 99.95, 100.0)


@dataclass(frozen=True)
class CandidateMSE:
    percentile: float
    alpha: float
    mse: float


@dataclass(frozen=True)
class LayerClippingSearchResult:
    layer_name: str
    selected_percentile: float
    selected_alpha: float
    selected_mse: float
    candidate_mses: tuple[CandidateMSE, ...]
    num_activation_elements: int
    qparams: PostReluActivationQuantizationParams


@dataclass(frozen=True)
class MSESelectedActivationCalibrationResult:
    qparams_by_name: OrderedDict[str, PostReluActivationQuantizationParams]
    observed_site_names: tuple[str, ...]
    calibration_num_batches: int
    layer_results: tuple[LayerClippingSearchResult, ...]
    min_activation_scale: float
    max_activation_scale: float
    min_activation_zero_point: int
    max_activation_zero_point: int


class _PostReluActivationCollector:
    def __init__(self) -> None:
        self._values: OrderedDict[str, list[torch.Tensor]] = OrderedDict()

    def update(self, name: str, output: torch.Tensor) -> None:
        tensor = output.detach().float().reshape(-1).cpu().clone()
        if name not in self._values:
            self._values[name] = []
        self._values[name].append(tensor)

    def to_layer_results(
        self,
        site_names: Iterable[str],
        candidate_percentiles: tuple[float, ...],
        qmin: int,
        qmax: int,
    ) -> tuple[LayerClippingSearchResult, ...]:
        layer_results: list[LayerClippingSearchResult] = []
        for name in site_names:
            if name not in self._values:
                raise ValueError(f"Missing activation statistics for '{name}'.")
            values = torch.cat(self._values[name])
            layer_results.append(
                select_mse_minimizing_activation_qparams(
                    layer_name=name,
                    values=values,
                    candidate_percentiles=candidate_percentiles,
                    qmin=qmin,
                    qmax=qmax,
                )
            )
        return tuple(layer_results)


def calibrate_post_relu_activation_mse_selected(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    candidate_percentiles: Iterable[float] = MSE_SELECTED_PERCENTILES,
    qmin: int = ACTIVATION_UINT4_QMIN,
    qmax: int = ACTIVATION_UINT4_QMAX,
    max_batches: int | None = None,
) -> MSESelectedActivationCalibrationResult:
    validate_integer_range(qmin, qmax)
    resolved_percentiles = _validate_candidate_percentiles(candidate_percentiles)
    model.eval()
    target_modules = iter_named_post_relu_modules(model)
    if not target_modules:
        raise ValueError(
            "No nn.ReLU activation modules found for INT4-MSE-Selected calibration."
        )
    site_names = tuple(name for name, _ in target_modules)
    collector = _PostReluActivationCollector()

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in target_modules:
        handles.append(module.register_forward_hook(_make_collector_hook(name, collector)))

    num_batches = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                inputs = _extract_inputs(batch).to(device, non_blocking=True)
                model(inputs)
                num_batches += 1
    finally:
        for handle in handles:
            handle.remove()

    if num_batches == 0:
        raise ValueError("Calibration processed zero batches.")

    layer_results = collector.to_layer_results(
        site_names,
        candidate_percentiles=resolved_percentiles,
        qmin=qmin,
        qmax=qmax,
    )
    qparams_by_name = OrderedDict(
        (result.layer_name, result.qparams) for result in layer_results
    )
    scales = [params.scale for params in qparams_by_name.values()]
    zero_points = [params.zero_point for params in qparams_by_name.values()]

    return MSESelectedActivationCalibrationResult(
        qparams_by_name=qparams_by_name,
        observed_site_names=site_names,
        calibration_num_batches=num_batches,
        layer_results=layer_results,
        min_activation_scale=min(scales),
        max_activation_scale=max(scales),
        min_activation_zero_point=min(zero_points),
        max_activation_zero_point=max(zero_points),
    )


def select_mse_minimizing_activation_qparams(
    layer_name: str,
    values: torch.Tensor,
    candidate_percentiles: Iterable[float] = MSE_SELECTED_PERCENTILES,
    qmin: int = ACTIVATION_UINT4_QMIN,
    qmax: int = ACTIVATION_UINT4_QMAX,
) -> LayerClippingSearchResult:
    validate_integer_range(qmin, qmax)
    resolved_percentiles = _validate_candidate_percentiles(candidate_percentiles)
    flattened = values.detach().float().reshape(-1).cpu()
    if flattened.numel() == 0:
        raise ValueError(f"Activation observer saw zero elements for '{layer_name}'.")
    if not torch.isfinite(flattened).all():
        raise FloatingPointError(f"Non-finite activation values observed for '{layer_name}'.")
    candidate_results: list[CandidateMSE] = []
    best_candidate: CandidateMSE | None = None
    best_qparams: PostReluActivationQuantizationParams | None = None
    order_statistic_cache: dict[int, float] = {}
    for percentile in resolved_percentiles:
        alpha = _percentile_to_alpha(
            flattened,
            percentile,
            order_statistic_cache,
        )
        qparams = make_post_relu_activation_qparams(
            clip_max=alpha,
            qmin=qmin,
            qmax=qmax,
        )
        dequantized = fake_quantize_post_relu_activation(flattened, qparams)
        diff = flattened - dequantized.float()
        mse = float(torch.mean(diff * diff).item())
        if not math.isfinite(mse) or mse < 0.0:
            raise FloatingPointError(
                f"Invalid candidate MSE for layer '{layer_name}', "
                f"percentile={percentile}: {mse}."
            )
        candidate = CandidateMSE(percentile=percentile, alpha=alpha, mse=mse)
        candidate_results.append(candidate)
        if best_candidate is None or candidate.mse < best_candidate.mse:
            best_candidate = candidate
            best_qparams = qparams

    if best_candidate is None or best_qparams is None:
        raise ValueError("At least one candidate percentile is required.")

    return LayerClippingSearchResult(
        layer_name=layer_name,
        selected_percentile=best_candidate.percentile,
        selected_alpha=best_candidate.alpha,
        selected_mse=best_candidate.mse,
        candidate_mses=tuple(candidate_results),
        num_activation_elements=flattened.numel(),
        qparams=best_qparams,
    )


def _percentile_to_alpha(
    values: torch.Tensor,
    percentile: float,
    order_statistic_cache: dict[int, float],
) -> float:
    if percentile == 100.0:
        return float(torch.amax(values).item())
    quantile = percentile / 100.0
    rank = quantile * float(values.numel() - 1)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    lower_value = _order_statistic(values, lower_index, order_statistic_cache)
    upper_value = _order_statistic(values, upper_index, order_statistic_cache)
    if lower_index == upper_index:
        return lower_value
    fraction = rank - float(lower_index)
    return lower_value + (upper_value - lower_value) * fraction


def _order_statistic(
    values: torch.Tensor,
    zero_based_index: int,
    cache: dict[int, float],
) -> float:
    if zero_based_index not in cache:
        kth_value = torch.kthvalue(values, zero_based_index + 1).values
        cache[zero_based_index] = float(kth_value.item())
    return cache[zero_based_index]


def _validate_candidate_percentiles(
    candidate_percentiles: Iterable[float],
) -> tuple[float, ...]:
    percentiles = tuple(float(percentile) for percentile in candidate_percentiles)
    if not percentiles:
        raise ValueError("candidate_percentiles must not be empty.")
    for percentile in percentiles:
        if not math.isfinite(percentile) or percentile <= 0.0 or percentile > 100.0:
            raise ValueError(
                "Each candidate percentile must be finite and in (0, 100], "
                f"got {percentile}."
            )
    return percentiles


def _make_collector_hook(
    name: str,
    collector: _PostReluActivationCollector,
) -> Any:
    def hook(
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor output for activation site '{name}'.")
        collector.update(name, output)

    return hook


def _extract_inputs(batch: Any) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch
