"""数据目录覆盖只改变数据根，不改变项目源码和默认目录。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_quant_data_dir_supports_relative_and_absolute_paths(tmp_path: Path):
    relative = subprocess.run(
        [
            sys.executable,
            "-c",
            "from common.paths import DATA_DIR, DB_FILE; "
            "print(DATA_DIR); print(DB_FILE)",
        ],
        cwd=ROOT,
        env={**os.environ, "QUANT_DATA_DIR": ".demo-data"},
        check=True,
        capture_output=True,
        text=True,
    )
    relative_lines = relative.stdout.splitlines()
    assert relative_lines[0] == str(ROOT / ".demo-data")
    assert relative_lines[1] == str(ROOT / ".demo-data" / "databases" / "quant.duckdb")

    absolute = subprocess.run(
        [
            sys.executable,
            "-c",
            "from common.paths import DATA_DIR; print(DATA_DIR)",
        ],
        cwd=ROOT,
        env={**os.environ, "QUANT_DATA_DIR": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert absolute.stdout.strip() == str(tmp_path)
