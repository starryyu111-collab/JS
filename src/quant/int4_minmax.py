from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


# Signed symmetric weight quantization uses [-7, 7] so zero is exactly centered.
WEIGHT_INT4_QMIN = -7
WEIGHT_INT4_QMAX = 7
# Post-ReLU activations are non-negative and use the full unsigned INT4 range.
ACTIVATION_UINT4_QMIN = 0
ACTIVATION_UINT4_QMAX = 15

TARGET_WEIGHT_MODULE_TYPES = (nn.Conv2d, nn.Linear)
TARGET_ACTIVATION_MODULE_TYPES = (nn.ReLU,)


@dataclass(frozen=True)
class PostReluActivationQuantizationParams:
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    clip_min: float
    clip_max: float


@dataclass(frozen=True)
class ActivationCalibrationResult:
    qparams_by_name: OrderedDict[str, PostReluActivationQuantizationParams]
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


class _PostReluMaxObserver:
    def __init__(self) -> None:
        self._maxs: OrderedDict[str, float] = OrderedDict()

    def update(self, name: str, output: torch.Tensor) -> None:
        tensor = output.detach()
        current_max = float(torch.amax(tensor).item())
        if name not in self._maxs:
            self._maxs[name] = current_max
            return
        self._maxs[name] = max(self._maxs[name], current_max)

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
            if name not in self._maxs:
                raise ValueError(f"Missing activation observer statistics for '{name}'.")
            qparams_by_name[name] = make_post_relu_activation_qparams(
                clip_max=self._maxs[name],
                qmin=qmin,
                qmax=qmax,
            )
        return qparams_by_name


class ActivationFakeQuantWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        qparams: PostReluActivationQuantizationParams,
    ) -> None:
        super().__init__()
        self.module = module
        self.qparams = qparams

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        output = self.module(*args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Post-ReLU fake quantization expects tensor module outputs, "
                f"got {type(output)!r}."
            )
        return fake_quantize_post_relu_activation(output, self.qparams)


def validate_integer_range(qmin: int, qmax: int) -> None:
    if qmin >= qmax:
        raise ValueError(f"qmin must be less than qmax, got {qmin} >= {qmax}.")


def iter_named_weight_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, TARGET_WEIGHT_MODULE_TYPES)
    ]


def iter_named_post_relu_modules(model: nn.Module) -> list[tuple[str, nn.ReLU]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, TARGET_ACTIVATION_MODULE_TYPES)
    ]


def make_post_relu_activation_qparams(
    clip_max: float,
    qmin: int = ACTIVATION_UINT4_QMIN,
    qmax: int = ACTIVATION_UINT4_QMAX,
) -> PostReluActivationQuantizationParams:
    validate_integer_range(qmin, qmax)
    if qmin != 0:
        raise ValueError("Post-ReLU UINT4 activation quantization requires qmin=0.")

    resolved_clip_max = clip_max if math.isfinite(clip_max) and clip_max > 0.0 else 0.0
    scale = resolved_clip_max / float(qmax - qmin)
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    return PostReluActivationQuantizationParams(
        scale=scale,
        zero_point=0,
        qmin=qmin,
        qmax=qmax,
        clip_min=0.0,
        clip_max=resolved_clip_max,
    )


def fake_quantize_post_relu_activation(
    tensor: torch.Tensor,
    qparams: PostReluActivationQuantizationParams,
) -> torch.Tensor:
    validate_integer_range(qparams.qmin, qparams.qmax)
    if qparams.qmin != 0:
        raise ValueError("Post-ReLU UINT4 activation quantization requires qmin=0.")
    if qparams.zero_point != 0:
        raise ValueError("Post-ReLU UINT4 activation zero-point must be 0.")
    if not math.isfinite(qparams.scale) or qparams.scale <= 0.0:
        raise ValueError(
            f"Activation scale must be finite and positive, got {qparams.scale}."
        )
    if not math.isfinite(qparams.clip_min) or not math.isfinite(qparams.clip_max):
        raise ValueError("Activation clipping bounds must be finite.")
    if qparams.clip_min != 0.0 or qparams.clip_max < qparams.clip_min:
        raise ValueError(
            "Post-ReLU activation clipping interval must be [0, clip_max]."
        )

    scale = torch.as_tensor(qparams.scale, dtype=tensor.dtype, device=tensor.device)
    zero_point = torch.as_tensor(
        qparams.zero_point,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    clipped = torch.clamp(tensor, qparams.clip_min, qparams.clip_max)
    quantized = torch.clamp(
        torch.round(clipped / scale + zero_point),
        qparams.qmin,
        qparams.qmax,
    )
    return (quantized - zero_point) * scale


def fake_quantize_per_channel_symmetric(
    weight: torch.Tensor,
    qmin: int = WEIGHT_INT4_QMIN,
    qmax: int = WEIGHT_INT4_QMAX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_integer_range(qmin, qmax)
    if qmin != -qmax:
        raise ValueError("Symmetric weight quantization requires qmin == -qmax.")
    if weight.ndim < 2:
        raise ValueError("Per-channel weight quantization requires at least 2 dims.")

    source = weight.detach()
    reduce_dims = tuple(dim for dim in range(source.ndim) if dim != 0)
    max_abs = source.abs().amax(dim=reduce_dims, keepdim=True)
    scale = max_abs / float(qmax)
    scale = torch.where(max_abs == 0, torch.ones_like(scale), scale)
    quantized = torch.clamp(torch.round(source / scale), qmin, qmax)
    dequantized = quantized * scale
    return dequantized, quantized, scale


def apply_weight_fake_quantization(
    model: nn.Module,
    qmin: int = WEIGHT_INT4_QMIN,
    qmax: int = WEIGHT_INT4_QMAX,
) -> WeightQuantizationResult:
    module_names: list[str] = []
    min_scale = math.inf
    max_scale = -math.inf

    with torch.no_grad():
        for name, module in iter_named_weight_modules(model):
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


def calibrate_post_relu_activation_ranges(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    qmin: int = ACTIVATION_UINT4_QMIN,
    qmax: int = ACTIVATION_UINT4_QMAX,
    max_batches: int | None = None,
) -> ActivationCalibrationResult:
    validate_integer_range(qmin, qmax)
    model.eval()
    target_modules = iter_named_post_relu_modules(model)
    if not target_modules:
        raise ValueError("No nn.ReLU activation modules found for INT4 calibration.")
    site_names = tuple(name for name, _ in target_modules)
    observer = _PostReluMaxObserver()

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


def attach_post_relu_activation_fake_quantization(
    model: nn.Module,
    qparams_by_name: OrderedDict[str, PostReluActivationQuantizationParams],
) -> tuple[str, ...]:
    wrapped_names: list[str] = []
    for name, qparams in qparams_by_name.items():
        parent, child_name, module = _resolve_child_module(model, name)
        if not isinstance(module, TARGET_ACTIVATION_MODULE_TYPES):
            raise TypeError(f"Module '{name}' is not an nn.ReLU activation target.")
        parent._modules[child_name] = ActivationFakeQuantWrapper(module, qparams)
        wrapped_names.append(name)
    return tuple(wrapped_names)


def compute_activation_reconstruction_mse(
    model: nn.Module,
    loader: Iterable[Any],
    qparams_by_name: OrderedDict[str, PostReluActivationQuantizationParams],
    device: torch.device,
    max_batches: int | None = None,
) -> ActivationMSE:
    model.eval()
    target_modules = dict(iter_named_post_relu_modules(model))
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
            dequantized = fake_quantize_post_relu_activation(tensor, qparams)
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
    observer: _PostReluMaxObserver,
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
