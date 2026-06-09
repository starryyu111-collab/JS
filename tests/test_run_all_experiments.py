from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import pytest

from src.experiments import run_all_experiments
from src.experiments.run_all_experiments import MethodSpec


def test_build_main_rows_preserves_accuracy_drop_and_warns_on_fp32_mismatch() -> None:
    specs = (
        _make_spec("FP32", 32, 32),
        _make_spec("INT4-P99.9", 4, 4),
    )
    raw_rows: OrderedDict[str, dict[str, str]] = OrderedDict(
        [
            (
                "FP32",
                {
                    "method": "FP32",
                    "top1_accuracy": "12.3438",
                    "is_smoke": "false",
                    "test_size": "10000",
                    "evaluated_test_size": "10000",
                },
            ),
            (
                "INT4-P99.9",
                {
                    "method": "INT4-P99.9",
                    "top1_accuracy": "11.6900",
                    "accuracy_drop": "0.5400",
                    "activation_mse": "0.00559793",
                    "logit_mse": "1.14621115",
                    "activation_site_type": "post_relu",
                    "activation_granularity": "per_tensor_per_relu_module",
                    "fp32_top1_accuracy": "12.2300",
                    "is_smoke": "false",
                    "test_size": "10000",
                    "evaluated_test_size": "10000",
                },
            ),
        ]
    )

    with pytest.warns(RuntimeWarning, match="preserving recorded accuracy_drop"):
        rows = run_all_experiments.build_main_rows(raw_rows, specs)

    assert rows == [
        {
            "method": "FP32",
            "weight_bits": "32",
            "activation_bits": "32",
            "activation_site_type": "none",
            "activation_granularity": "none",
            "top1_accuracy": "12.3438",
            "accuracy_drop": "0.0000",
            "activation_mse": "0.00000000",
            "logit_mse": "0.00000000",
        },
        {
            "method": "INT4-P99.9",
            "weight_bits": "4",
            "activation_bits": "4",
            "activation_site_type": "post_relu",
            "activation_granularity": "per_tensor_per_relu_module",
            "top1_accuracy": "11.6900",
            "accuracy_drop": "0.5400",
            "activation_mse": "0.00559793",
            "logit_mse": "1.14621115",
        },
    ]


def test_write_main_tables_uses_exact_columns_and_markdown_values(tmp_path: Path) -> None:
    rows = [
        {
            "method": "FP32",
            "weight_bits": "32",
            "activation_bits": "32",
            "activation_site_type": "none",
            "activation_granularity": "none",
            "top1_accuracy": "12.3438",
            "accuracy_drop": "0.0000",
            "activation_mse": "0.00000000",
            "logit_mse": "0.00000000",
        },
        {
            "method": "INT8-MinMax",
            "weight_bits": "8",
            "activation_bits": "8",
            "activation_site_type": "conv_linear_output",
            "activation_granularity": "per_tensor",
            "top1_accuracy": "12.8100",
            "accuracy_drop": "-0.5800",
            "activation_mse": "0.00028467",
            "logit_mse": "0.01724852",
        },
    ]
    output_csv = tmp_path / "main_results.csv"
    output_md = tmp_path / "main_results.md"

    run_all_experiments.write_main_tables(rows, output_csv, output_md)

    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == run_all_experiments.MAIN_RESULT_COLUMNS
        assert list(reader) == rows

    markdown = output_md.read_text(encoding="utf-8")
    assert (
        "| method | weight_bits | activation_bits | activation_site_type | "
        "activation_granularity | top1_accuracy | accuracy_drop | activation_mse | "
        "logit_mse |"
    ) in markdown
    assert "| FP32 | 32 | 32 | none | none | 12.3438 | 0.0000 | 0.00000000 | 0.00000000 |" in markdown
    assert (
        "| INT8-MinMax | 8 | 8 | conv_linear_output | per_tensor | "
        "12.8100 | -0.5800 | 0.00028467 | 0.01724852 |"
    ) in markdown


def test_write_summary_figures_creates_paper_ready_pngs(tmp_path: Path) -> None:
    rows = [
        {
            "method": "FP32",
            "weight_bits": "32",
            "activation_bits": "32",
            "activation_site_type": "none",
            "activation_granularity": "none",
            "top1_accuracy": "94.1300",
            "accuracy_drop": "0.0000",
            "activation_mse": "0.00000000",
            "logit_mse": "0.00000000",
        },
        {
            "method": "INT4-MinMax",
            "weight_bits": "4",
            "activation_bits": "4",
            "activation_site_type": "post_relu",
            "activation_granularity": "per_tensor_per_relu_module",
            "top1_accuracy": "88.2000",
            "accuracy_drop": "5.9300",
            "activation_mse": "0.00273660",
            "logit_mse": "1.84912006",
        },
        {
            "method": "INT4-P99.9",
            "weight_bits": "4",
            "activation_bits": "4",
            "activation_site_type": "post_relu",
            "activation_granularity": "per_tensor_per_relu_module",
            "top1_accuracy": "92.9700",
            "accuracy_drop": "1.1600",
            "activation_mse": "0.00035606",
            "logit_mse": "0.46350010",
        },
    ]

    run_all_experiments.write_summary_figures(rows, tmp_path)

    accuracy_figure = tmp_path / "accuracy_drop.png"
    error_figure = tmp_path / "error_metrics.png"
    assert accuracy_figure.exists()
    assert accuracy_figure.stat().st_size > 0
    assert error_figure.exists()
    assert error_figure.stat().st_size > 0


def test_validate_full_result_row_rejects_smoke_csv(tmp_path: Path) -> None:
    spec = _make_spec("INT4-MinMax", 4, 4)
    row = {
        "method": "INT4-MinMax",
        "is_smoke": "true",
        "test_size": "128",
        "evaluated_test_size": "128",
    }

    with pytest.raises(ValueError, match="Smoke result CSV"):
        run_all_experiments.validate_full_result_row(
            spec,
            row,
            tmp_path / "int4_minmax_result_smoke.csv",
        )


def test_validate_full_result_row_rejects_missing_evaluated_size(tmp_path: Path) -> None:
    spec = _make_spec("FP32", 32, 32)
    row = {
        "method": "FP32",
        "is_smoke": "false",
        "test_size": "10000",
        "top1_accuracy": "94.1300",
    }

    with pytest.raises(ValueError, match="evaluated_test_size"):
        run_all_experiments.validate_full_result_row(
            spec,
            row,
            tmp_path / "fp32_result.csv",
        )


def test_auto_mode_runs_only_missing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp32_config = tmp_path / "fp32.yaml"
    int8_config = tmp_path / "int8.yaml"
    fp32_result = tmp_path / "fp32_result.csv"
    int8_result = tmp_path / "int8_result.csv"
    specs = (
        _make_spec("FP32", 32, 32, config_path=fp32_config, result_path=fp32_result),
        _make_spec(
            "INT8-MinMax",
            8,
            8,
            config_path=int8_config,
            result_path=int8_result,
        ),
    )
    _write_config(fp32_config, fp32_result)
    _write_config(int8_config, int8_result)
    _write_result_csv(
        fp32_result,
        {
            "method": "FP32",
            "top1_accuracy": "12.3438",
            "is_smoke": "false",
            "test_size": "10000",
            "evaluated_test_size": "10000",
        },
    )
    calls: list[str] = []

    def fake_run_method(spec: MethodSpec, _config: dict[str, object]) -> None:
        calls.append(spec.method)
        _write_result_csv(
            int8_result,
            {
                "method": "INT8-MinMax",
                "top1_accuracy": "12.8100",
                "accuracy_drop": "-0.5800",
                "activation_mse": "0.00028467",
                "logit_mse": "0.01724852",
                "activation_site_type": "conv_linear_output",
                "activation_granularity": "per_tensor",
                "fp32_top1_accuracy": "12.3438",
                "is_smoke": "false",
                "test_size": "10000",
                "evaluated_test_size": "10000",
            },
        )

    monkeypatch.setattr(run_all_experiments, "run_method", fake_run_method)

    raw_rows = run_all_experiments.collect_raw_result_rows(specs, mode="auto")

    assert calls == ["INT8-MinMax"]
    assert list(raw_rows) == ["FP32", "INT8-MinMax"]


def test_read_only_mode_fails_when_result_is_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "fp32.yaml"
    result_path = tmp_path / "fp32_result.csv"
    _write_config(config_path, result_path)
    specs = (
        _make_spec("FP32", 32, 32, config_path=config_path, result_path=result_path),
    )

    with pytest.raises(FileNotFoundError, match="Missing full result CSV"):
        run_all_experiments.collect_raw_result_rows(specs, mode="read-only")


def _make_spec(
    method: str,
    weight_bits: int,
    activation_bits: int,
    config_path: Path | None = None,
    result_path: Path | None = None,
) -> MethodSpec:
    safe_name = method.lower().replace(".", "").replace("-", "_")
    return MethodSpec(
        method=method,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        module_name=f"fake.{safe_name}",
        config_path=config_path or Path(f"{safe_name}.yaml"),
        default_result_path=result_path or Path(f"{safe_name}.csv"),
    )


def _write_config(path: Path, result_path: Path) -> None:
    path.write_text(
        "paths:\n"
        f"  result_path: {result_path.as_posix()}\n",
        encoding="utf-8",
    )


def _write_result_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
