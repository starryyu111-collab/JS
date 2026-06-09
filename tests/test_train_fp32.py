from pathlib import Path

from src.experiments.train_fp32 import resolve_artifact_path


def test_smoke_artifact_paths_are_separated_from_full_fp32_outputs() -> None:
    result_path = resolve_artifact_path(
        Path("outputs/results/fp32_result.csv"),
        is_smoke=True,
        path_overridden=False,
    )
    checkpoint_path = resolve_artifact_path(
        Path("checkpoints/fp32_best.pt"),
        is_smoke=True,
        path_overridden=False,
    )
    log_path = resolve_artifact_path(
        Path("outputs/logs/fp32_cifar10.log"),
        is_smoke=True,
        path_overridden=False,
    )

    assert result_path == Path("outputs/results/fp32_result_smoke.csv")
    assert checkpoint_path == Path("checkpoints/fp32_best_smoke.pt")
    assert log_path == Path("outputs/logs/fp32_cifar10_smoke.log")


def test_explicit_fp32_artifact_path_is_preserved_for_smoke_runs() -> None:
    configured_path = Path("custom/fp32.csv")

    resolved_path = resolve_artifact_path(
        configured_path,
        is_smoke=True,
        path_overridden=True,
    )

    assert resolved_path == configured_path
