"""合成演示应能在隔离目录中完成生成、评估和回测。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_demo_end_to_end(tmp_path: Path):
    data_dir = tmp_path / "demo"
    build = subprocess.run(
        [
            sys.executable,
            "examples/build_demo_data.py",
            "--output",
            str(data_dir),
            "--stocks",
            "40",
            "--start",
            "2021-01-04",
            "--end",
            "2022-12-30",
            "--force",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"factors": 5' in build.stdout
    assert (data_dir / "databases" / "quant.duckdb").is_file()
    assert (data_dir / "databases" / "factors.duckdb").is_file()

    run = subprocess.run(
        [
            sys.executable,
            "examples/run_demo.py",
            "--data-dir",
            str(data_dir),
            "--start",
            "2022-01-04",
            "--end",
            "2022-12-30",
        ],
        cwd=ROOT,
        env={**os.environ, "QUANT_DATA_DIR": str(data_dir)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "演示完成" in run.stdout
    assert "运行 MOM_20 Top-20 月频回测" in run.stdout
