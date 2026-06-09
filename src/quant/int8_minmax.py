from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


# Signed symmetric weight quantization uses [-127, 127] so zero is exactly centered.
WEIGHT_INT8_QMIN = -127
WEIGHT_INT8_QMAX = 127
# UINT8 activation quantization uses the full affine range.
ACTIVATION_UINT8_QMIN = 0
ACTIVATION_UINT8_QMAX = 255
# Optional signed activation range is kept explicit for configs that request int8.
ACTIVATION_INT8_QMIN = -128
ACTIVATION_INT8_QMAX = 127

TARGET_MODULE_TYPES = (nn.Conv2d, nn.Linear)


@dataclass(frozen=True)
class AffineQuantizationParams:
    scale: float
    zero_point: int
    qmin: int
    qmax: int


@dataclass(frozen=True)
class ActivationCalibrationResult:
    qparams_by_name: OrderedDict[str, AffineQuantizationParams]
    observed_site_names: tuple[str, ...]
    calibration_num_batches: int
    min_activation_scale: float
    max_activation_scale: float
    min_activation_zero_point: int
    max_activation_zero_point: int


@dataclass(frozen=True)
class WeightQuantizationResult:
    quantized_module_names: tuple[str, ...]
    num_quantized_modules: int
    min_weight_scale: float
    max_weight_scale: float
    qmin: int
    qmax: int


@dataclass(frozen=True)
class ActivationMSE:
    mse: float
    num_elements: int
    num_batches: int


class _ActivationMinMaxObserver:
    def __init__(self) -> None:
        self._mins: OrderedDict[str, float] = OrderedDict()
        self._maxs: OrderedDict[str, float] = OrderedDict()

    def update(self, name: str, output: torch.Tensor) -> None:
        tensor = output.detach()
        current_min = float(torch.amin(tensor).item())
        current_max = float(torch.amax(tensor).item())
        if name not in self._mins:
            self._mins[name] = current_min
            self._maxs[name] = current_max
            return
        self._mins[name] = min(self._mins[name], current_min)
        self._maxs[name] = max(self._maxs[name], current_max)

    def to_qparams(
        self,
        site_names: Iterable[str],
        qmin: int,
        qmax: int,
    ) -> OrderedDict[str, AffineQuantizationParams]:
        validate_integer_range(qmin, qmax)
        qparams_by_name: OrderedDict[str, AffineQuantizationParams] = OrderedDict()
        for name in site_names:
            if name not in self._mins or name not in self._maxs:
                raise ValueError(f"Missing activation observer statistics for '{name}'.")

            x_min = self._mins[name]
            x_max = self._maxs[name]
            scale = (x_max - x_min) / float(qmax - qmin)
            if not math.isfinite(scale) or scale <= 0.0:
                scale = 1.0

            zero_point = int(round(qmin - x_min / scale))
            zero_point = min(max(zero_point, qmin), qmax)
            qparams_by_name[name] = AffineQuantizationParams(
                scale=scale,
                zero_point=zero_point,
                qmin=qmin,
                qmax=qmax,
            )
        return qparams_by_name


class ActivationFakeQuantWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        qparams: AffineQuantizationParams,
    ) -> None:
        super().__init__()
        self.module = module
        self.qparams = qparams

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        output = self.module(*args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Activation fake quantization expects tensor module outputs, "
                f"got {type(output)!r}."
            )
        return fake_quantize_per_tensor_affine(
            output,
            scale=self.qparams.scale,
            zero_point=self.qparams.zero_point,
            qmin=self.qparams.qmin,
            qmax=self.qparams.qmax,
        )


def validate_integer_range(qmin: int, qmax: int) -> None:
    if qmin >= qmax:
        raise ValueError(f"qmin must be less than qmax, got {qmin} >= {qmax}.")


def iter_named_target_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, TARGET_MODULE_TYPES)
    ]


def fake_quantize_per_tensor_affine(
    tensor: torch.Tensor,
    scale: float,
    zero_point: int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    validate_integer_range(qmin, qmax)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Activation scale must be finite and positive, got {scale}.")
    zero_point = min(max(int(zero_point), qmin), qmax)
    scale_tensor = torch.as_tensor(scale, dtype=tensor.dtype, device=tensor.device)
    zero_point_tensor = torch.as_tensor(
        zero_point,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    quantized = torch.clamp(
        torch.round(tensor / scale_tensor + zero_point_tensor),
        qmin,
        qmax,
    )
    return (quantized - zero_point_tensor) * scale_tensor


def fake_quantize_per_channel_symmetric(
    weight: torch.Tensor,
    qmin: int = WEIGHT_INT8_QMIN,
    qmax: int = WEIGHT_INT8_QMAX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_integer_range(qmin, qmax)
    if qmin != -qmax:
        raise ValueError(f"Symmetric weight quantization requires qmin == -qmax.")
    if weight.ndim < 2:
        raise ValueError("Per-channel weight quantization requires at least 2 dims.")

    reduce_dims = tuple(dim for dim in range(weight.ndim) if dim != 0)
    max_abs = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = max_abs / float(qmax)
    scale = torch.where(max_abs == 0, torch.ones_like(scale), scale)
    quantized = torch.clamp(torch.round(weight / scale), qmin, qmax)
    dequantized = quantized * scale
    return dequantized, quantized, scale


def apply_weight_fake_quantization(
    model: nn.Module,
    qmin: int = WEIGHT_INT8_QMIN,
    qmax: int = WEIGHT_INT8_QMAX,
) -> WeightQuantizationResult:
    module_names: list[str] = []
    min_scale = math.inf
    max_scale = -math.inf

    with torch.no_grad():
        for name, module in iter_named_target_modules(model):
            dequantized, _, scale = fake_quantize_per_channel_symmetric(
                module.weight,
                qmin=qmin,
                qmax=qmax,
            )
            module.weight.copy_(dequantized)
            module_names.append(name)
            min_scale = min(min_scale, float(torch.amin(scale).item()))
            max_scale = max(max_scale, float(torch.amax(scale).item()))

    if not module_names:
        min_scale = 0.0
        max_scale = 0.0

    return WeightQuantizationResult(
        quantized_module_names=tuple(module_names),
        num_quantized_modules=len(module_names),
        min_weight_scale=min_scale,
        max_weight_scale=max_scale,
        qmin=qmin,
        qmax=qmax,
    )


def calibrate_activation_ranges(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    qmin: int = ACTIVATION_UINT8_QMIN,
    qmax: int = ACTIVATION_UINT8_QMAX,
    max_batches: int | None = None,
) -> ActivationCalibrationResult:
    validate_integer_range(qmin, qmax)
    model.eval()
    target_modules = iter_named_target_modules(model)
    site_names = tuple(name for name, _ in target_modules)
    observer = _ActivationMinMaxObserver()

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
        min_activation_scale=min(scales) if scales else 0.0,
        max_activation_scale=max(scales) if scales else 0.0,
        min_activation_zero_point=min(zero_points) if zero_points else qmin,
        max_activation_zero_point=max(zero_points) if zero_points else qmin,
    )


def attach_activation_fake_quantization(
    model: nn.Module,
    qparams_by_name: OrderedDict[str, AffineQuantizationParams],
) -> tuple[str, ...]:
    wrapped_names: list[str] = []
    for name, qparams in qparams_by_name.items():
        parent, child_name, module = _resolve_child_module(model, name)
        if not isinstance(module, TARGET_MODULE_TYPES):
            raise TypeError(f"Module '{name}' is not a Conv2d or Linear target.")
        parent._modules[child_name] = ActivationFakeQuantWrapper(module, qparams)
        wrapped_names.append(name)
    return tuple(wrapped_names)


def compute_activation_reconstruction_mse(
    model: nn.Module,
    loader: Iterable[Any],
    qparams_by_name: OrderedDict[str, AffineQuantizationParams],
    device: torch.device,
    max_batches: int | None = None,
) -> ActivationMSE:
    model.eval()
    target_modules = dict(iter_named_target_modules(model))
    missing_names = [name for name in qparams_by_name if name not in target_modules]
    if missing_names:
        raise ValueError(f"Activation MSE target sites are missing: {missing_names}.")

    total_squared_error = 0.0
    total_elements = 0

    def make_hook(name: str) -> Any:
        qparams = qparams_by_name[name]

        def hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            nonlocal total_squared_error, total_elements
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Expected tensor output for activation site '{name}'.")
            tensor = output.detach()
            dequantized = fake_quantize_per_tensor_affine(
                tensor,
                scale=qparams.scale,
                zero_point=qparams.zero_point,
                qmin=qparams.qmin,
                qmax=qparams.qmax,
            )
            diff = (tensor - dequantized).float()
            total_squared_error += float(torch.sum(diff * diff).item())
            total_elements += tensor.numel()

        return hook

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name in qparams_by_name:
        handles.append(target_modules[name].register_forward_hook(make_hook(name)))

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
        raise ValueError("Activation MSE processed zero batches.")
    if total_elements == 0:
        raise ValueError("Activation MSE saw zero activation elements.")

    return ActivationMSE(
        mse=total_squared_error / total_elements,
        num_elements=total_elements,
        num_batches=num_batches,
    )


def _make_observer_hook(
    name: str,
    observer: _ActivationMinMaxObserver,
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


def _resolve_child_module(
    model: nn.Module,
    dotted_name: str,
) -> tuple[nn.Module, str, nn.Module]:
    parts = dotted_name.split(".")
    if not parts:
        raise ValueError("Module name must not be empty.")

    parent = model
    for part in parts[:-1]:
        if part not in parent._modules:
            raise ValueError(f"Unknown module path '{dotted_name}'.")
        parent = parent._modules[part]

    child_name = parts[-1]
    if child_name not in parent._modules:
        raise ValueError(f"Unknown module path '{dotted_name}'.")
    return parent, child_name, parent._modules[child_name]
