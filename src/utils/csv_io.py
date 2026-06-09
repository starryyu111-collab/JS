import csv
from pathlib import Path
from typing import Any


def write_single_row_csv(
    path: str | Path,
    row: dict[str, Any],
    fieldnames: list[str],
) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_rows_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
