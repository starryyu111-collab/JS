from __future__ import annotations

import argparse
import csv
import importlib
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt


Mode = Literal["auto", "read-only", "run-all"]

MAIN_RESULT_COLUMNS = [
    "method",
    "weight_bits",
    "activation_bits",
    "activation_site_type",
    "activation_granularity",
    "top1_accuracy",
    "accuracy_drop",
    "activation_mse",
    "logit_mse",
]

FP32_REFERENCE_TOLERANCE = 1e-4
FULL_CIFAR10_TEST_SIZE = 10000


@dataclass(frozen=True)
class MethodSpec:
    method: str
    weight_bits: int
    activation_bits: int
    module_name: str
    config_path: Path
    default_result_path: Path


METHOD_SPECS = (
    MethodSpec(
        method="FP32",
        weight_bits=32,
        activation_bits=32,
        module_name="src.experiments.train_fp32",
        config_path=Path("configs/fp32_cifar10.yaml"),
        default_result_path=Path("outputs/results/fp32_result.csv"),
    ),
    MethodSpec(
        method="INT8-MinMax",
        weight_bits=8,
        activation_bits=8,
        module_name="src.experiments.run_int8_minmax_ptq",
        config_path=Path("configs/int8_minmax_cifar10.yaml"),
        default_result_path=Path("outputs/results/int8_minmax_result.csv"),
    ),
    MethodSpec(
        method="INT4-MinMax",
        weight_bits=4,
        activation_bits=4,
        module_name="src.experiments.run_int4_minmax_ptq",
        config_path=Path("configs/int4_minmax_cifar10.yaml"),
        default_result_path=Path("outputs/results/int4_minmax_result.csv"),
    ),
    MethodSpec(
        method="INT4-P99.9",
        weight_bits=4,
        activation_bits=4,
        module_name="src.experiments.run_int4_p999_ptq",
        config_path=Path("configs/int4_p999_cifar10.yaml"),
        default_result_path=Path("outputs/results/int4_p999_result.csv"),
    ),
    MethodSpec(
        method="INT4-MSE-Selected",
        weight_bits=4,
        activation_bits=4,
        module_name="src.experiments.run_int4_mse_selected_ptq",
        config_path=Path("configs/int4_mse_selected_cifar10.yaml"),
        default_result_path=Path("outputs/results/int4_mse_selected_result.csv"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect FP32/INT8/INT4 full experiment results into a main table."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "read-only", "run-all"],
        default="auto",
        help=(
            "auto reads existing full CSVs and runs missing methods; read-only "
            "fails on missing CSVs; run-all reruns all methods first."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/results/main_results.csv",
        help="Path for the normalized main results CSV.",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/results/main_results.md",
        help="Path for the Markdown version of the main results table.",
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/figures",
        help="Directory for paper-ready summary figures.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Do not write summary figures.",
    )
    parser.add_argument(
        "--strict-fp32-reference",
        action="store_true",
        help="Treat FP32 reference mismatches in quantized CSVs as errors.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must contain a YAML mapping.")
    return config


def resolve_result_path(spec: MethodSpec, config: dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    if isinstance(paths, dict) and paths.get("result_path"):
        return Path(str(paths["result_path"]))
    return spec.default_result_path


def collect_raw_result_rows(
    specs: tuple[MethodSpec, ...] = METHOD_SPECS,
    mode: Mode = "auto",
) -> OrderedDict[str, dict[str, str]]:
    raw_rows: OrderedDict[str, dict[str, str]] = OrderedDict()

    for spec in specs:
        config = load_config(spec.config_path)
        result_path = resolve_result_path(spec, config)
        should_run = mode == "run-all" or (mode == "auto" and not result_path.exists())
        if should_run:
            run_method(spec, config)
        elif mode == "read-only" and not result_path.exists():
            raise FileNotFoundError(
                f"Missing full result CSV for {spec.method}: {result_path}"
            )

        if not result_path.exists():
            raise FileNotFoundError(
                f"{spec.method} did not produce expected result CSV: {result_path}"
            )
        row = read_single_row_csv(result_path)
        validate_full_result_row(spec, row, result_path)
        raw_rows[spec.method] = row

    return raw_rows


def run_method(spec: MethodSpec, config: dict[str, Any]) -> None:
    module = importlib.import_module(spec.module_name)
    if spec.method == "FP32":
        module.run(config, status="ok", fallback_reason="")
        return
    module.run(config)


def read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}.")
    return rows[0]


def validate_full_result_row(
    spec: MethodSpec,
    row: dict[str, str],
    result_path: Path,
) -> None:
    method = row.get("method", "")
    if method != spec.method:
        raise ValueError(
            f"Method mismatch in {result_path}: expected {spec.method}, got {method}."
        )

    smoke_value = row.get("is_smoke", "").strip().lower()
    if not smoke_value:
        raise ValueError(f"Full result CSV is missing is_smoke metadata: {result_path}")
    if smoke_value == "true":
        raise ValueError(f"Smoke result CSV cannot be used for the main table: {result_path}")
    if smoke_value != "false":
        raise ValueError(f"Invalid is_smoke value in {result_path}: {smoke_value}")

    test_size = _parse_required_size(row, "test_size", spec.method, result_path)
    if test_size != FULL_CIFAR10_TEST_SIZE:
        raise ValueError(
            f"{spec.method} result is not a full CIFAR-10 test run: test_size={test_size}."
        )

    evaluated_size = _parse_required_size(
        row,
        "evaluated_test_size",
        spec.method,
        result_path,
    )
    if evaluated_size != FULL_CIFAR10_TEST_SIZE:
        raise ValueError(
            f"{spec.method} result is not fully evaluated: "
            f"evaluated_test_size={evaluated_size}."
        )

    if spec.method != "FP32":
        for field_name in ("activation_site_type", "activation_granularity"):
            if not row.get(field_name, "").strip():
                raise ValueError(
                    f"{spec.method} result is missing required activation metadata "
                    f"'{field_name}' in {result_path}."
                )


def build_main_rows(
    raw_rows: OrderedDict[str, dict[str, str]],
    specs: tuple[MethodSpec, ...] = METHOD_SPECS,
    strict_fp32_reference: bool = False,
) -> list[dict[str, str]]:
    fp32_top1 = _parse_finite_metric(raw_rows["FP32"], "top1_accuracy", "FP32")
    main_rows: list[dict[str, str]] = []

    for spec in specs:
        raw_row = raw_rows[spec.method]
        if spec.method == "FP32":
            main_rows.append(
                {
                    "method": spec.method,
                    "weight_bits": str(spec.weight_bits),
                    "activation_bits": str(spec.activation_bits),
                    "activation_site_type": "none",
                    "activation_granularity": "none",
                    "top1_accuracy": _metric_text(raw_row, "top1_accuracy", spec.method),
                    "accuracy_drop": "0.0000",
                    "activation_mse": "0.00000000",
                    "logit_mse": "0.00000000",
                }
            )
            continue

        _check_fp32_reference(
            spec.method,
            raw_row,
            fp32_top1=fp32_top1,
            strict=strict_fp32_reference,
        )
        main_rows.append(
            {
                "method": spec.method,
                "weight_bits": str(spec.weight_bits),
                "activation_bits": str(spec.activation_bits),
                "activation_site_type": _metadata_text(
                    raw_row,
                    "activation_site_type",
                    spec.method,
                ),
                "activation_granularity": _metadata_text(
                    raw_row,
                    "activation_granularity",
                    spec.method,
                ),
                "top1_accuracy": _metric_text(raw_row, "top1_accuracy", spec.method),
                "accuracy_drop": _metric_text(raw_row, "accuracy_drop", spec.method),
                "activation_mse": _metric_text(raw_row, "activation_mse", spec.method),
                "logit_mse": _metric_text(raw_row, "logit_mse", spec.method),
            }
        )

    return main_rows


def write_main_tables(
    rows: list[dict[str, str]],
    output_csv: Path,
    output_md: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAIN_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    output_md.write_text(format_markdown_table(rows), encoding="utf-8")


def write_summary_figures(rows: list[dict[str, str]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_accuracy_drop_figure(rows, figures_dir / "accuracy_drop.png")
    write_error_metric_figure(rows, figures_dir / "error_metrics.png")


def write_accuracy_drop_figure(rows: list[dict[str, str]], output_path: Path) -> None:
    methods = [row["method"] for row in rows]
    top1_values = [float(row["top1_accuracy"]) for row in rows]
    drop_values = [float(row["accuracy_drop"]) for row in rows]
    x_positions = list(range(len(rows)))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    axes[0].bar(x_positions, top1_values, color="#3c6e71")
    axes[0].set_title("Top-1 accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(max(0.0, min(top1_values) - 2.0), min(100.0, max(top1_values) + 1.0))

    axes[1].bar(x_positions, drop_values, color="#c1666b")
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set_title("Accuracy drop vs FP32")
    axes[1].set_ylabel("Drop (percentage points)")

    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(methods, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_error_metric_figure(rows: list[dict[str, str]], output_path: Path) -> None:
    quantized_rows = [row for row in rows if row["method"] != "FP32"]
    methods = [row["method"] for row in quantized_rows]
    activation_mse = [float(row["activation_mse"]) for row in quantized_rows]
    logit_mse = [float(row["logit_mse"]) for row in quantized_rows]
    x_positions = list(range(len(quantized_rows)))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].bar(x_positions, activation_mse, color="#4f7cac")
    axes[0].set_title("Activation reconstruction MSE")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")

    axes[1].bar(x_positions, logit_mse, color="#7a9e7e")
    axes[1].set_title("Logit MSE")
    axes[1].set_ylabel("MSE")
    axes[1].set_yscale("log")

    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(methods, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def format_markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(MAIN_RESULT_COLUMNS) + " |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in MAIN_RESULT_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def _metric_text(row: dict[str, str], column: str, method: str) -> str:
    value = row.get(column, "").strip()
    _parse_finite_metric(row, column, method)
    return value


def _metadata_text(row: dict[str, str], column: str, method: str) -> str:
    value = row.get(column, "").strip()
    if not value:
        raise ValueError(f"Missing {column} for {method}.")
    return value


def _parse_finite_metric(row: dict[str, str], column: str, method: str) -> float:
    raw_value = row.get(column, "").strip()
    if raw_value == "":
        raise ValueError(f"Missing {column} for {method}.")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {column} for {method}: {raw_value}.")
    return value


def _parse_required_size(
    row: dict[str, str],
    column: str,
    method: str,
    result_path: Path,
) -> int:
    raw_value = row.get(column, "").strip()
    if raw_value == "":
        raise ValueError(f"Missing {column} for {method} in {result_path}.")
    value = int(float(raw_value))
    if value <= 0:
        raise ValueError(f"Invalid {column} for {method} in {result_path}: {raw_value}.")
    return value


def _check_fp32_reference(
    method: str,
    row: dict[str, str],
    fp32_top1: float,
    strict: bool,
) -> None:
    raw_reference = row.get("fp32_top1_accuracy", "").strip()
    if raw_reference == "":
        return
    method_reference = float(raw_reference)
    if abs(method_reference - fp32_top1) <= FP32_REFERENCE_TOLERANCE:
        return

    message = (
        f"{method} fp32_top1_accuracy={method_reference:.4f} differs from "
        f"FP32 row top1_accuracy={fp32_top1:.4f}; preserving recorded accuracy_drop."
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def run(
    mode: Mode = "auto",
    output_csv: Path = Path("outputs/results/main_results.csv"),
    output_md: Path = Path("outputs/results/main_results.md"),
    figures_dir: Path = Path("outputs/figures"),
    write_figures: bool = True,
    strict_fp32_reference: bool = False,
) -> list[dict[str, str]]:
    raw_rows = collect_raw_result_rows(METHOD_SPECS, mode=mode)
    main_rows = build_main_rows(
        raw_rows,
        METHOD_SPECS,
        strict_fp32_reference=strict_fp32_reference,
    )
    write_main_tables(main_rows, output_csv=output_csv, output_md=output_md)
    if write_figures:
        write_summary_figures(main_rows, figures_dir=figures_dir)
    return main_rows


def main() -> None:
    args = parse_args()
    run(
        mode=args.mode,
        output_csv=Path(args.output_csv),
        output_md=Path(args.output_md),
        figures_dir=Path(args.figures_dir),
        write_figures=not args.skip_figures,
        strict_fp32_reference=args.strict_fp32_reference,
    )


if __name__ == "__main__":
    main()
