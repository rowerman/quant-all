"""沙箱代码限制与因子明细确定性评估测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from mining_sandbox.evaluator import (
    SandboxProposalPolicy,
    evaluate_factor_output,
    parse_research_metrics,
    read_research_metrics,
    validate_proposal_code,
)
from mining_sandbox.feedback import SandboxFailureCategory, SandboxFeedbackError
from mining_sandbox.qlib_snapshot import QlibSnapshotManifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(tmp_path: Path) -> tuple[QlibSnapshotManifest, Path]:
    root = tmp_path / "snapshot"
    calendar = root / "calendars" / "day.txt"
    instruments = root / "instruments" / "all.txt"
    calendar.parent.mkdir(parents=True)
    instruments.parent.mkdir(parents=True)
    calendar.write_text(
        "2019-12-27\n2019-12-30\n2019-12-31\n",
        encoding="utf-8",
    )
    instruments.write_text(
        "SH600000\t2019-12-27\t2019-12-31\n"
        "SZ000001\t2019-12-30\t2019-12-31\n",
        encoding="utf-8",
    )
    files = {
        "calendars/day.txt": {
            "bytes": calendar.stat().st_size,
            "sha256": _sha256(calendar),
        },
        "instruments/all.txt": {
            "bytes": instruments.stat().st_size,
            "sha256": _sha256(instruments),
        },
    }
    manifest = QlibSnapshotManifest(
        context_hash="a" * 64,
        data_version="test_data_v1",
        universe="all_a",
        start_date="2019-12-27",
        end_date="2019-12-31",
        fields=("close", "label"),
        calendar_sha256=files["calendars/day.txt"]["sha256"],
        instruments_sha256=files["instruments/all.txt"]["sha256"],
        files=files,
        row_count=5,
        total_bytes=0,
    )
    return manifest, root


def _metrics(*, coverage: float = 0.6, valid_days: int = 2):
    return {
        "rank_ic": 0.03,
        "icir": 0.4,
        "coverage": coverage,
        "valid_days": valid_days,
    }


def _valid_factor(path: Path) -> Path:
    pd.DataFrame(
        {
            "date": ["2019-12-27", "2019-12-30", "2019-12-30"],
            "instrument": ["SH600000", "SH600000", "SZ000001"],
            "value": [0.1, 0.2, -0.1],
        }
    ).to_parquet(path, index=False)
    return path


def _feedback_code(action) -> str:
    with pytest.raises(SandboxFeedbackError) as exc_info:
        action()
    return exc_info.value.feedback.error_code


def _feedback(action):
    with pytest.raises(SandboxFeedbackError) as exc_info:
        action()
    return exc_info.value.feedback


def test_valid_factor_evaluation_is_complete_and_deterministic(tmp_path: Path):
    manifest, snapshot_dir = _snapshot(tmp_path)
    factor = _valid_factor(tmp_path / "factor.parquet")

    first = evaluate_factor_output(
        factor_file=factor,
        snapshot_manifest=manifest,
        snapshot_dir=snapshot_dir,
        metrics=_metrics(),
    )
    second = evaluate_factor_output(
        factor_file=factor,
        snapshot_manifest=manifest,
        snapshot_dir=snapshot_dir,
        metrics=_metrics(),
    )

    assert first.to_dict() == second.to_dict()
    assert first.factor_sha256 == _sha256(factor)
    assert first.row_count == 5
    assert first.finite_value_count == 3
    assert first.valid_days == 2
    assert first.instrument_count == 2
    assert first.coverage == pytest.approx(0.6)
    assert first.evaluation_hash == second.evaluation_hash


def test_missing_or_malformed_factor_is_output_failure(tmp_path: Path):
    manifest, snapshot_dir = _snapshot(tmp_path)
    assert _feedback_code(
        lambda: evaluate_factor_output(
            factor_file=tmp_path / "missing.parquet",
            snapshot_manifest=manifest,
            snapshot_dir=snapshot_dir,
            metrics=_metrics(),
        )
    ) == "MINING-140"

    wrong = tmp_path / "wrong.parquet"
    pd.DataFrame(
        {
            "date": ["2019-12-27"],
            "instrument": ["SH600000"],
            "value": [0.1],
            "extra": [1],
        }
    ).to_parquet(wrong, index=False)
    assert _feedback_code(
        lambda: evaluate_factor_output(
            factor_file=wrong,
            snapshot_manifest=manifest,
            snapshot_dir=snapshot_dir,
            metrics=_metrics(coverage=0.2, valid_days=1),
        )
    ) == "MINING-140"

    wrong_value = tmp_path / "wrong_value.parquet"
    pd.DataFrame(
        {
            "date": ["2019-12-27"],
            "instrument": ["SH600000"],
            "value": ["not-a-number"],
        }
    ).to_parquet(wrong_value, index=False)
    assert _feedback_code(
        lambda: evaluate_factor_output(
            factor_file=wrong_value,
            snapshot_manifest=manifest,
            snapshot_dir=snapshot_dir,
            metrics=_metrics(coverage=0.2, valid_days=1),
        )
    ) == "MINING-140"


@pytest.mark.parametrize(
    ("frame", "expected_code"),
    [
        (
            pd.DataFrame(
                {
                    "date": ["2019-12-27", "2019-12-27"],
                    "instrument": ["SH600000", "SH600000"],
                    "value": [0.1, 0.2],
                }
            ),
            "MINING-141",
        ),
        (
            pd.DataFrame(
                {
                    "date": ["2019-12-28"],
                    "instrument": ["SH600000"],
                    "value": [0.1],
                }
            ),
            "MINING-141",
        ),
        (
            pd.DataFrame(
                {
                    "date": ["2019-12-27"],
                    "instrument": ["SH999999"],
                    "value": [0.1],
                }
            ),
            "MINING-141",
        ),
        (
            pd.DataFrame(
                {
                    "date": ["2019-12-27"],
                    "instrument": ["SZ000001"],
                    "value": [0.1],
                }
            ),
            "MINING-141",
        ),
    ],
)
def test_factor_index_violations_are_stable(
    tmp_path: Path,
    frame: pd.DataFrame,
    expected_code: str,
):
    manifest, snapshot_dir = _snapshot(tmp_path)
    factor = tmp_path / "factor.parquet"
    frame.to_parquet(factor, index=False)

    first_feedback = _feedback(
        lambda: evaluate_factor_output(
            factor_file=factor,
            snapshot_manifest=manifest,
            snapshot_dir=snapshot_dir,
            metrics=_metrics(coverage=0.2, valid_days=1),
        )
    )
    second_feedback = _feedback(
        lambda: evaluate_factor_output(
            factor_file=factor,
            snapshot_manifest=manifest,
            snapshot_dir=snapshot_dir,
            metrics=_metrics(coverage=0.2, valid_days=1),
        )
    )

    assert first_feedback.to_dict() == second_feedback.to_dict()
    assert first_feedback.feedback_hash == second_feedback.feedback_hash
    assert first_feedback.error_code == expected_code


def test_coverage_numeric_and_empty_output_failures(tmp_path: Path):
    manifest, snapshot_dir = _snapshot(tmp_path)
    factor = _valid_factor(tmp_path / "factor.parquet")
    assert _feedback_code(
        lambda: evaluate_factor_output(
            factor_file=factor,
            snapshot_manifest=manifest,
            snapshot_dir=snapshot_dir,
            metrics=_metrics(valid_days=3),
        )
    ) == "MINING-141"
    assert _feedback_code(
        lambda: evaluate_factor_output(
            factor_file=factor,
            snapshot_manifest=manifest,
            snapshot_dir=snapshot_dir,
            metrics=_metrics(coverage=0.5),
        )
    ) == "MINING-142"

    non_finite = tmp_path / "non_finite.parquet"
    pd.DataFrame(
        {
            "date": ["2019-12-27", "2019-12-30", "2019-12-30"],
            "instrument": ["SH600000", "SH600000", "SZ000001"],
            "value": [0.1, float("inf"), -0.1],
        }
    ).to_parquet(non_finite, index=False)
    assert _feedback_code(
        lambda: evaluate_factor_output(
            factor_file=non_finite,
            snapshot_manifest=manifest,
            snapshot_dir=snapshot_dir,
            metrics=_metrics(),
        )
    ) == "MINING-143"

    empty = tmp_path / "empty.parquet"
    pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "instrument": pd.Series(dtype="string"),
            "value": pd.Series(dtype="float64"),
        }
    ).to_parquet(empty, index=False)
    assert _feedback_code(
        lambda: evaluate_factor_output(
            factor_file=empty,
            snapshot_manifest=manifest,
            snapshot_dir=snapshot_dir,
            metrics=_metrics(coverage=0.0, valid_days=1),
        )
    ) == "MINING-141"


def test_metrics_parser_is_strict(tmp_path: Path):
    assert parse_research_metrics(_metrics())["coverage"] == pytest.approx(0.6)
    assert _feedback_code(
        lambda: parse_research_metrics({**_metrics(), "extra": True})
    ) == "MINING-140"
    assert _feedback_code(
        lambda: parse_research_metrics({**_metrics(), "coverage": 1.2})
    ) == "MINING-142"
    assert _feedback_code(
        lambda: parse_research_metrics({**_metrics(), "rank_ic": float("nan")})
    ) == "MINING-143"

    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text("{broken", encoding="utf-8")
    assert _feedback_code(lambda: read_research_metrics(metrics_file)) == "MINING-140"


def test_code_policy_allows_research_imports_and_output_paths():
    validate_proposal_code(
        "import json\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from qlib import init\n"
        "from scipy.stats import spearmanr\n"
        "init(provider_uri='/snapshot', region='cn')\n"
        "Path('/workspace/factor.parquet').write_bytes(b'')\n"
        "Path('/workspace/metrics.json').write_text(json.dumps({}))\n"
        "print(np.mean([1.0]), pd.DataFrame(), spearmanr)\n"
        "if __name__ == '__main__':\n    pass\n"
    )


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\n",
        "import os\n",
        "from socket import socket\n",
        "eval('1 + 1')\n",
        "__import__('os')\n",
        "open('/snapshot/manifest.json').read()\n",
        "open('/etc/passwd').read()\n",
        "data = '/snapshot/../workspace/metadata.json'\n",
        "data = '/workspace/metrics.json/../../etc/passwd'\n",
        "data = object().__class__\n",
        "from pathlib import *\n",
        "from .policy import load_sandbox_policy\n",
        "import json\nglobals()\n",
    ],
)
def test_code_policy_rejects_clear_escape_attempts(code: str):
    first = _feedback(lambda: validate_proposal_code(code))
    second = _feedback(lambda: validate_proposal_code(code))
    assert first.to_dict() == second.to_dict()
    assert first.feedback_hash == second.feedback_hash
    assert first.error_code == "MINING-144"


def test_code_policy_compile_failure_and_custom_policy():
    assert _feedback_code(lambda: validate_proposal_code("def broken(:\n")) == (
        "MINING-138"
    )
    policy = SandboxProposalPolicy(
        allowed_import_roots=("math",),
        allowed_absolute_paths=("/snapshot",),
        forbidden_call_names=("eval",),
        forbidden_names=(),
    )
    with pytest.raises(SandboxFeedbackError) as exc_info:
        validate_proposal_code("import json\n", policy=policy)
    assert exc_info.value.feedback.category is SandboxFailureCategory.CODE_POLICY
