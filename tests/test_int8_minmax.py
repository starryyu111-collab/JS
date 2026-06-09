import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.run_int8_minmax_ptq import (
    _validate_int8_activation_config,
    _validate_int8_weight_config,
    resolve_result_path,
)
from src.quant.int8_minmax import (
    ACTIVATION_UINT8_QMAX,
    ACTIVATION_UINT8_QMIN,
    WEIGHT_INT8_QMAX,
    WEIGHT_INT8_QMIN,
    apply_weight_fake_quantization,
    attach_activation_fake_quantization,
    calibrate_activation_ranges,
    compute_activation_reconstruction_mse,
    fake_quantize_per_channel_symmetric,
    fake_quantize_per_tensor_affine,
)


def test_weight_fake_quantization_shape_scale_and_range() -> None:
    weight = torch.tensor(
        [
            [[[0.0, 0.5], [-0.5, 0.25]]],
            [[[2.0, -1.0], [0.0, 1.0]]],
        ]
    )

    dequantized, quantized, scale = fake_quantize_per_channel_symmetric(weight)

    assert dequantized.shape == weight.shape
    assert quantized.min().item() >= WEIGHT_INT8_QMIN
    assert quantized.max().item() <= WEIGHT_INT8_QMAX
    assert scale.shape == (2, 1, 1, 1)
    assert torch.isfinite(scale).all()
    assert torch.all(scale > 0)


def test_int8_weight_config_requires_per_channel_symmetric_range() -> None:
    _validate_int8_weight_config(
        {
            "dtype": "int8",
            "qmin": WEIGHT_INT8_QMIN,
            "qmax": WEIGHT_INT8_QMAX,
            "granularity": "per_channel",
            "symmetry": "symmetric",
            "channel_axis": 0,
        }
    )

    try:
        _validate_int8_weight_config({"dtype": "int8", "qmin": -128, "qmax": 127})
    except ValueError as exc:
        assert "[-127, 127]" in str(exc)
    else:
        raise AssertionError("INT8 weight qrange drift should be rejected.")

    try:
        _validate_int8_weight_config({"dtype": "int8", "symmetry": "affine"})
    except ValueError as exc:
        assert "symmetry=symmetric" in str(exc)
    else:
        raise AssertionError("INT8 weight symmetry drift should be rejected.")


def test_activation_fake_quant_uses_configured_uint8_range() -> None:
    tensor = torch.tensor([-2.0, 0.0, 2.0])

    dequantized = fake_quantize_per_tensor_affine(
        tensor,
        scale=4.0 / 255.0,
        zero_point=128,
        qmin=ACTIVATION_UINT8_QMIN,
        qmax=ACTIVATION_UINT8_QMAX,
    )

    assert dequantized.shape == tensor.shape
    assert torch.isfinite(dequantized).all()


def test_int8_activation_config_declares_conv_linear_output_site() -> None:
    _validate_int8_activation_config(
        {
            "site": "conv_linear_output",
            "clip_method": "minmax",
        }
    )
    try:
        _validate_int8_activation_config({"site": "post_relu", "clip_method": "minmax"})
    except ValueError as exc:
        assert "site=conv_linear_output" in str(exc)
    else:
        raise AssertionError("INT8-MinMax should reject mismatched activation hook sites.")


def test_observer_and_wrapper_site_names_match() -> None:
    model = _build_tiny_model()
    loader = _build_tiny_loader()

    calibration = calibrate_activation_ranges(
        model,
        loader,
        torch.device("cpu"),
        max_batches=2,
    )
    int8_model = copy.deepcopy(model)
    weight_result = apply_weight_fake_quantization(int8_model)
    wrapped_names = attach_activation_fake_quantization(
        int8_model,
        calibration.qparams_by_name,
    )

    assert calibration.observed_site_names == ("0", "3")
    assert wrapped_names == calibration.observed_site_names
    assert weight_result.num_quantized_modules == len(calibration.observed_site_names)
    assert calibration.min_activation_scale > 0
    assert calibration.min_activation_zero_point >= ACTIVATION_UINT8_QMIN
    assert calibration.max_activation_zero_point <= ACTIVATION_UINT8_QMAX


def test_activation_mse_is_local_reconstruction_error() -> None:
    model = _build_tiny_model()
    loader = _build_tiny_loader()
    calibration = calibrate_activation_ranges(
        model,
        loader,
        torch.device("cpu"),
        max_batches=2,
    )

    mse = compute_activation_reconstruction_mse(
        model,
        loader,
        calibration.qparams_by_name,
        torch.device("cpu"),
        max_batches=2,
    )

    assert mse.num_batches == 2
    assert mse.num_elements > 0
    assert mse.mse >= 0.0


def test_smoke_result_path_defaults_to_separate_csv() -> None:
    result_path = resolve_result_path(
        configured_result_path=Path("outputs/results/int8_minmax_result.csv"),
        is_smoke=True,
        result_path_overridden=False,
    )

    assert result_path.parent == Path("outputs/results")
    assert result_path.name == "int8_minmax_result_smoke.csv"


def _build_tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 2),
    ).eval()


def _build_tiny_loader() -> DataLoader:
    inputs = torch.randn(6, 3, 8, 8)
    targets = torch.zeros(6, dtype=torch.long)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
