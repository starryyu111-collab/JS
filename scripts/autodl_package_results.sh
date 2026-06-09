#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

archive_path = Path("dist/autodl_results.zip")
output_roots = [
    Path("outputs/results"),
    Path("outputs/logs"),
    Path("outputs/figures"),
]

archive_path.parent.mkdir(parents=True, exist_ok=True)
if archive_path.exists():
    archive_path.unlink()

with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
    for root in output_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.as_posix())

print(f"Created {archive_path}")
PY
