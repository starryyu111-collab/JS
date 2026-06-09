from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn

from src.quant.int4_minmax import (
    ACTIVATION_UINT4_QMAX,
    ACTIVATION_UINT4_QMIN,
    TARGET_ACTIVATION_MODULE_TYPES,
    WEIGHT_INT4_QMAX,
    WEIGHT_INT4_QMIN,
    ActivationCalibrationResult,
    ActivationMSE,
    PostReluActivationQuantizationParams,
    WeightQuantizationResult,
    apply_weight_fake_quantization,
    attach_post_relu_activation_fake_quantization,
    compute_activation_reconstruction_mse,
    fake_quantize_per_channel_symmetric,
    fake_quantize_post_relu_activation,
    iter_named_post_relu_modules,
    make_post_relu_activation_qparams,
    validate_integer_range,
)


FIXED_ACTIVATION_PERCENTILE = 99.9
FIXED_ACTIVATION_QUANTILE = 0.999
TORCH_QUANTILE_MAX_INPUT_ELEMENTS = 2**24


class _PostReluPercentileObserver:
    def __init__(self) -> None:
        self._values: OrderedDict[str, list[torch.Tensor]] = OrderedDict()

    def update(self, name: str, output: torch.Tensor) -> None:
        tensor = output.detach().float().reshape(-1).cpu()
        if name not in self._values:
            self._values[name] = []
        self._values[name].append(tensor)

    def to_qparams(
        self,
        site_names: Iterable[str],
        qmin: int,
        qmax: int,
    ) -> OrderedDict[str, PostReluActivationQuantizationParams]:
        validate_integer_range(qmin, qmax)
        qparams_by_name: OrderedDict[str, PostReluActivationQuantizationParams] = (
            OrderedDict()
        )
        for name in site_names:
            if name not in self._values:
                raise ValueError(f"Missing activation observer statistics for '{name}'.")
            values = torch.cat(self._values[name])
            if values.numel() == 0:
                raise ValueError(f"Activation observer saw zero elements for '{name}'.")
            alpha = float(
                compute_empirical_quantile(values, FIXED_ACTIVATION_QUANTILE).item()
            )
            qparams_by_name[name] = make_post_relu_activation_p999_qparams(
                alpha=alpha,
                qmin=qmin,
                qmax=qmax,
            )
        return qparams_by_name


def make_post_relu_activation_p999_qparams(
    alpha: float,
    qmin: int = ACTIVATION_UINT4_QMIN,
    qmax: int = ACTIVATION_UINT4_QMAX,
) -> PostReluActivationQuantizationParams:
    return make_post_relu_activation_qparams(
        clip_max=alpha,
        qmin=qmin,
        qmax=qmax,
    )


def compute_empirical_quantile(values: torch.Tensor, quantile: float) -> torch.Tensor:
    flat_values = values.detach().float().reshape(-1).cpu()
    if flat_values.numel() == 0:
        raise ValueError("Cannot compute a quantile for an empty tensor.")
    if flat_values.numel() > TORCH_QUANTILE_MAX_INPUT_ELEMENTS:
        return _compute_quantile_by_order_statistics(flat_values, quantile)
    try:
        return torch.quantile(flat_values, quantile)
    except RuntimeError as exc:
        if not _is_torch_quantile_size_limit_error(exc):
            raise
    return _compute_quantile_by_order_statistics(flat_values, quantile)


def calibrate_post_relu_activation_p999(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    qmin: int = ACTIVATION_UINT4_QMIN,
    qmax: int = ACTIVATION_UINT4_QMAX,
    max_batches: int | None = None,
) -> ActivationCalibrationResult:
    validate_integer_range(qmin, qmax)
    _validate_fixed_percentile_constants()
    model.eval()
    target_modules = iter_named_post_relu_modules(model)
    if not target_modules:
        raise ValueError("No nn.ReLU activation modules found for INT4-P99.9 calibration.")
    site_names = tuple(name for name, _ in target_modules)
    observer = _PostReluPercentileObserver()

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in target_modules:
        handles.append(module.register_forward_hook(_make_observer_hook(name, observer)))

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

    qparams_by_name = observer.to_qparams(site_names, qmin=qmin, qmax=qmax)
    scales = [params.scale for params in qparams_by_name.values()]
    zero_points = [params.zero_point for params in qparams_by_name.values()]

    return ActivationCalibrationResult(
        qparams_by_name=qparams_by_name,
        observed_site_names=site_names,
        calibration_num_batches=num_batches,
        min_activation_scale=min(scales),
        max_activation_scale=max(scales),
        min_activation_zero_point=min(zero_points),
        max_activation_zero_point=max(zero_points),
    )


def _validate_fixed_percentile_constants() -> None:
    if not math.isclose(FIXED_ACTIVATION_PERCENTILE, 99.9, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("INT4-P99.9 percentile must be fixed at 99.9.")
    if not math.isclose(FIXED_ACTIVATION_QUANTILE, 0.999, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("INT4-P99.9 quantile must be fixed at 0.999.")


def _is_torch_quantile_size_limit_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "quantile" in message and "input tensor is too large" in message


def _compute_quantile_by_order_statistics(
    flat_values: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}.")

    numel = flat_values.numel()
    if numel == 1:
        return flat_values[0]

    fractional_index = quantile * (numel - 1)
    lower_index = math.floor(fractional_index)
    upper_index = math.ceil(fractional_index)
    if lower_index == upper_index:
        partitioned = np.partition(flat_values.numpy(), lower_index)
        lower_value = float(partitioned[lower_index])
        return torch.tensor(lower_value, dtype=flat_values.dtype)

    partitioned = np.partition(flat_values.numpy(), (lower_index, upper_index))
    lower_value = float(partitioned[lower_index])
    upper_value = float(partitioned[upper_index])
    interpolation_weight = fractional_index - lower_index
    quantile_value = lower_value + (upper_value - lower_value) * interpolation_weight
    return torch.tensor(quantile_value, dtype=flat_values.dtype)


def _make_observer_hook(
    name: str,
    observer: _PostReluPercentileObserver,
) -> Any:
    def hook(
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor output for activation site '{name}'.")
        observer.update(name, output)

    return hook


def _extract_inputs(batch: Any) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch
