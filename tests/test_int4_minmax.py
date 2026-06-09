import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import build_model
from src.experiments.run_int4_minmax_ptq import (
    _validate_int4_post_relu_activation_config,
    _validate_int4_weight_config,
    compute_logit_mse,
    resolve_device,
    resolve_result_path,
)
from src.quant.int4_minmax import (
    ACTIVATION_UINT4_QMAX,
    ACTIVATION_UINT4_QMIN,
    WEIGHT_INT4_QMAX,
    WEIGHT_INT4_QMIN,
    apply_weight_fake_quantization,
    attach_post_relu_activation_fake_quantization,
    calibrate_post_relu_activation_ranges,
    compute_activation_reconstruction_mse,
    fake_quantize_per_channel_symmetric,
    fake_quantize_post_relu_activation,
    iter_named_post_relu_modules,
    make_post_relu_activation_qparams,
)


def test_weight_fake_quantization_shape_scale_and_int4_range() -> None:
    conv_weight = torch.tensor(
        [
            [[[0.0, 0.5], [-0.5, 0.25]]],
            [[[2.0, -1.0], [0.0, 1.0]]],
        ]
    )
    linear_weight = torch.tensor([[0.0, 0.5, -0.5], [2.0, -1.0, 0.0]])

    conv_dequantized, conv_quantized, conv_scale = fake_quantize_per_channel_symmetric(
        conv_weight
    )
    linear_dequantized, linear_quantized, linear_scale = (
        fake_quantize_per_channel_symmetric(linear_weight)
    )

    assert conv_dequantized.shape == conv_weight.shape
    assert linear_dequantized.shape == linear_weight.shape
    assert conv_quantized.min().item() >= WEIGHT_INT4_QMIN
    assert conv_quantized.max().item() <= WEIGHT_INT4_QMAX
    assert linear_quantized.min().item() >= WEIGHT_INT4_QMIN
    assert linear_quantized.max().item() <= WEIGHT_INT4_QMAX
    assert conv_scale.shape == (2, 1, 1, 1)
    assert linear_scale.shape == (2, 1)
    assert torch.isfinite(conv_scale).all()
    assert torch.isfinite(linear_scale).all()
    assert torch.all(conv_scale > 0)
    assert torch.all(linear_scale > 0)


def test_int4_weight_config_requires_per_channel_symmetric_range() -> None:
    _validate_int4_weight_config(
        {
            "dtype": "int4",
            "qmin": WEIGHT_INT4_QMIN,
            "qmax": WEIGHT_INT4_QMAX,
            "granularity": "per_channel",
            "symmetry": "symmetric",
            "channel_axis": 0,
        }
    )

    try:
        _validate_int4_weight_config({"dtype": "int4", "qmin": -8, "qmax": 7})
    except ValueError as exc:
        assert "[-7, 7]" in str(exc)
    else:
        raise AssertionError("INT4 weight qrange drift should be rejected.")

    try:
        _validate_int4_weight_config({"dtype": "int4", "granularity": "per_tensor"})
    except ValueError as exc:
        assert "granularity=per_channel" in str(exc)
    else:
        raise AssertionError("INT4 weight granularity drift should be rejected.")


def test_activation_fake_quant_clips_to_post_relu_uint4_interval() -> None:
    qparams = make_post_relu_activation_qparams(
        clip_max=1.0,
        qmin=ACTIVATION_UINT4_QMIN,
        qmax=ACTIVATION_UINT4_QMAX,
    )
    tensor = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])

    dequantized = fake_quantize_post_relu_activation(tensor, qparams)

    assert qparams.scale == 1.0 / 15.0
    assert qparams.zero_point == 0
    assert qparams.clip_min == 0.0
    assert qparams.clip_max == 1.0
    assert dequantized.shape == tensor.shape
    assert torch.isfinite(dequantized).all()
    assert dequantized.min().item() >= 0.0
    assert dequantized.max().item() <= 1.0
    assert dequantized[0].item() == 0.0
    assert dequantized[-1].item() == 1.0


def test_int4_activation_config_matches_post_relu_hook_granularity() -> None:
    _validate_int4_post_relu_activation_config(
        {
            "dtype": "uint4",
            "qmin": ACTIVATION_UINT4_QMIN,
            "qmax": ACTIVATION_UINT4_QMAX,
            "site": "post_relu",
            "granularity": "per_tensor_per_relu_module",
            "clip_min": 0,
        }
    )

    try:
        _validate_int4_post_relu_activation_config(
            {"dtype": "uint4", "site": "conv_linear_output"}
        )
    except ValueError as exc:
        assert "site=post_relu" in str(exc)
    else:
        raise AssertionError("INT4 activation hook site drift should be rejected.")

    try:
        _validate_int4_post_relu_activation_config(
            {"dtype": "uint4", "granularity": "per_tensor"}
        )
    except ValueError as exc:
        assert "granularity=per_tensor_per_relu_module" in str(exc)
    else:
        raise AssertionError("INT4 activation granularity drift should be rejected.")


def test_zero_activation_max_uses_finite_fallback_scale() -> None:
    qparams = make_post_relu_activation_qparams(clip_max=0.0)
    tensor = torch.tensor([0.0, 1.0])

    dequantized = fake_quantize_post_relu_activation(tensor, qparams)

    assert qparams.scale == 1.0
    assert qparams.zero_point == 0
    assert qparams.clip_max == 0.0
    assert torch.isfinite(dequantized).all()
    assert torch.equal(dequantized, torch.zeros_like(tensor))


def test_observer_and_wrapper_target_relu_sites_only() -> None:
    model = _build_tiny_model()
    loader = _build_tiny_loader()

    calibration = calibrate_post_relu_activation_ranges(
        model,
        loader,
        torch.device("cpu"),
        max_batches=2,
    )
    int4_model = copy.deepcopy(model)
    weight_result = apply_weight_fake_quantization(int4_model)
    wrapped_names = attach_post_relu_activation_fake_quantization(
        int4_model,
        calibration.qparams_by_name,
    )

    assert calibration.observed_site_names == ("2", "5")
    assert wrapped_names == calibration.observed_site_names
    assert weight_result.quantized_module_names == ("0", "3", "7")
    assert weight_result.num_quantized_modules == 3
    assert calibration.min_activation_scale > 0
    assert calibration.min_activation_zero_point == 0
    assert calibration.max_activation_zero_point == 0
    with torch.no_grad():
        output = int4_model(torch.randn(2, 3, 8, 8))
    assert torch.isfinite(output).all()


def test_reused_relu_module_is_one_module_level_activation_site() -> None:
    model = _ReusedReluModule().eval()
    loader = _build_tiny_loader()

    calibration = calibrate_post_relu_activation_ranges(
        model,
        loader,
        torch.device("cpu"),
        max_batches=1,
    )
    wrapped_names = attach_post_relu_activation_fake_quantization(
        model,
        calibration.qparams_by_name,
    )

    assert calibration.observed_site_names == ("relu",)
    assert wrapped_names == ("relu",)
    with torch.no_grad():
        output = model(torch.randn(2, 3, 8, 8))
    assert torch.isfinite(output).all()
    assert output.min().item() >= 0.0


def test_resnet18_cifar_uses_distinct_relu_modules_for_logical_sites() -> None:
    model = build_model("resnet18_cifar").eval()

    site_names = tuple(name for name, _module in iter_named_post_relu_modules(model))

    assert len(site_names) == 17
    assert site_names[:5] == (
        "relu",
        "layer1.0.relu1",
        "layer1.0.relu2",
        "layer1.1.relu1",
        "layer1.1.relu2",
    )
    assert site_names[-2:] == ("layer4.1.relu1", "layer4.1.relu2")


def test_activation_mse_is_local_post_relu_reconstruction_error() -> None:
    model = _build_tiny_model()
    loader = _build_tiny_loader()
    calibration = calibrate_post_relu_activation_ranges(
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


def test_logit_mse_is_finite_and_nonnegative() -> None:
    fp32_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2)).eval()
    int4_model = copy.deepcopy(fp32_model).eval()
    with torch.no_grad():
        int4_model[1].weight.add_(0.01)
    loader = _build_tiny_loader()

    mse = compute_logit_mse(
        fp32_model,
        int4_model,
        loader,
        torch.device("cpu"),
        max_batches=2,
    )

    assert mse >= 0.0
    assert torch.isfinite(torch.tensor(mse))


def test_resolve_device_auto_uses_cuda_when_runtime_probe_passes(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "src.experiments.run_int4_minmax_ptq._cuda_runtime_is_usable",
        lambda: True,
    )

    device = resolve_device("auto")

    assert device == torch.device("cuda")


def test_resolve_device_auto_falls_back_to_cpu_when_cuda_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = resolve_device("auto")

    assert device == torch.device("cpu")


def test_resolve_device_auto_falls_back_to_cpu_when_runtime_probe_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "src.experiments.run_int4_minmax_ptq._cuda_runtime_is_usable",
        lambda: False,
    )

    device = resolve_device("auto")

    assert device == torch.device("cpu")


def test_resolve_device_explicit_cuda_errors_when_cuda_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    try:
        resolve_device("cuda")
    except RuntimeError as exc:
        assert "CUDA device requested" in str(exc)
        assert "not available" in str(exc)
    else:
        raise AssertionError("resolve_device('cuda') should require visible CUDA.")


def test_resolve_device_explicit_cuda_errors_when_runtime_probe_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "src.experiments.run_int4_minmax_ptq._cuda_runtime_is_usable",
        lambda: False,
    )

    try:
        resolve_device("cuda")
    except RuntimeError as exc:
        assert "cannot run kernels" in str(exc)
    else:
        raise AssertionError("resolve_device('cuda') should require a usable runtime.")


def test_smoke_result_path_defaults_to_separate_int4_csv() -> None:
    result_path = resolve_result_path(
        configured_result_path=Path("outputs/results/int4_minmax_result.csv"),
        is_smoke=True,
        result_path_overridden=False,
    )

    assert result_path.parent == Path("outputs/results")
    assert result_path.name == "int4_minmax_result_smoke.csv"


def _build_tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 2),
    ).eval()


def _build_tiny_loader() -> DataLoader:
    inputs = torch.randn(6, 3, 8, 8)
    targets = torch.zeros(6, dtype=torch.long)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)


class _ReusedReluModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        return self.relu(x - 0.5)
