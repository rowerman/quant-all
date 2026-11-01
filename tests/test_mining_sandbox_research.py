"""沙箱代码研究、产物清单和人工提升测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import os
import subprocess

import duckdb
import pytest

from mining.search_context import FrozenSearchContext
import mining_sandbox.policy as policy_module
from mining_sandbox.artifacts import (
    PromotionRequest,
    copy_artifacts_for_review,
    record_manual_promotion,
    verify_artifact_tree,
)
from mining_sandbox.errors import SandboxArtifactError
from mining_sandbox.feedback import SandboxFailureCategory
from mining_sandbox.policy import load_sandbox_policy
from mining_sandbox.qlib_snapshot import QlibSnapshotManifest, build_qlib_snapshot
from mining_sandbox.research import SandboxResearchRequest, SandboxResearchRunner
from mining_sandbox.runtime import DockerSandboxRunner, SandboxExecutionResult
from mining_sandbox.worker_client import MockSandboxCodeWorkerClient
from mining_worker.protocol_code import WorkerCodeProposalResponse


def _snapshot_manifest() -> QlibSnapshotManifest:
    return QlibSnapshotManifest(
        context_hash="a" * 64,
        data_version="test_data_v1",
        universe="all_a",
        start_date="2012-01-01",
        end_date="2019-12-31",
        fields=("close", "label"),
        calendar_sha256="b" * 64,
        instruments_sha256="c" * 64,
        files={},
        row_count=3,
        total_bytes=0,
    )


_DEFAULT_METRICS = {
    "rank_ic": 0.03,
    "icir": 0.4,
    "coverage": 0.8,
    "valid_days": 100,
}
_UNSET = object()


class FakeDockerRunner:
    def __init__(
        self,
        *,
        passed: bool = True,
        timed_out: bool = False,
        metrics: object = _UNSET,
        metrics_text: str | None = None,
    ) -> None:
        self.passed = passed
        self.timed_out = timed_out
        self.metrics = _DEFAULT_METRICS if metrics is _UNSET else metrics
        self.metrics_text = metrics_text
        self.calls = 0

    def run(
        self,
        *,
        run_id,
        snapshot_dir,
        workspace_dir,
        command,
    ) -> SandboxExecutionResult:
        self.calls += 1
        workspace = Path(workspace_dir)
        if self.metrics_text is not None:
            (workspace / "metrics.json").write_text(
                self.metrics_text,
                encoding="utf-8",
            )
        elif self.passed and self.metrics is not None:
            (workspace / "metrics.json").write_text(
                json.dumps(self.metrics),
                encoding="utf-8",
            )
        exit_code = -9 if self.timed_out else (0 if self.passed else 2)
        return SandboxExecutionResult(
            run_id=run_id,
            container_name=f"test-{run_id}",
            image_id="sha256:" + "d" * 64,
            exit_code=exit_code,
            timed_out=self.timed_out,
            duration_ms=12.5,
            stdout="passed\n" if self.passed else "",
            stderr="" if self.passed else "failed\n",
            stdout_truncated=False,
            stderr_truncated=False,
        )


def _runner(tmp_path: Path, docker) -> tuple[SandboxResearchRunner, object]:
    policy = replace(
        load_sandbox_policy(),
        enabled=True,
        snapshot_root="snapshots",
        workspace_root="work",
        review_root="review",
    )
    worker = MockSandboxCodeWorkerClient(
        (
            WorkerCodeProposalResponse(
                proposal_id="proposal_1",
                code="print('first')",
                rationale="第一轮",
                model="deepseek/deepseek-chat",
            ),
            WorkerCodeProposalResponse(
                proposal_id="proposal_2",
                code="print('second')",
                rationale="根据反馈修订",
                model="deepseek/deepseek-chat",
            ),
        )
    )
    return (
        SandboxResearchRunner(
            policy=policy,
            worker=worker,
            docker_runner=docker,
        ),
        policy,
    )


def _request() -> SandboxResearchRequest:
    return SandboxResearchRequest(
        sandbox_run_id="sandboxrun_research_0001",
        hypothesis="短期价格修复在开发期可能形成可检验因子",
        context_hash="a" * 64,
        snapshot_manifest_hash=_snapshot_manifest().manifest_hash,
        llm_model="deepseek/deepseek-chat",
        max_iterations=2,
    )


def _single_proposal_runner(
    policy,
    docker,
    *,
    code: str,
) -> SandboxResearchRunner:
    return SandboxResearchRunner(
        policy=policy,
        worker=MockSandboxCodeWorkerClient(
            (
                WorkerCodeProposalResponse(
                    proposal_id="proposal_single",
                    code=code,
                    rationale="单轮反馈测试",
                    model="deepseek/deepseek-chat",
                ),
            )
        ),
        docker_runner=docker,
    )


def test_research_loop_archives_all_attempts_and_reuses_sealed_run(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    runner, policy = _runner(tmp_path, FakeDockerRunner())
    snapshot_dir = tmp_path / "snapshots" / "dev"
    snapshot_dir.mkdir(parents=True)

    manifest, reused = runner.run(
        request=_request(),
        snapshot=_snapshot_manifest(),
        snapshot_dir=snapshot_dir,
    )

    assert reused is False
    assert manifest.status == "completed"
    assert len(manifest.attempts) == 2
    assert [item["status"] for item in manifest.attempts] == [
        "passed",
        "passed",
    ]
    assert runner.worker.requests[1].previous_attempt["code_hash"] == (
        manifest.attempts[0]["code_hash"]
    )
    artifact_dir = policy.workspace_root_path / _request().sandbox_run_id
    assert verify_artifact_tree(artifact_dir).manifest_hash == manifest.manifest_hash
    assert (artifact_dir / "iterations/001/proposal.py").stat().st_mode & 0o222 == 0

    second, second_reused = runner.run(
        request=_request(),
        snapshot=_snapshot_manifest(),
        snapshot_dir=snapshot_dir,
    )
    assert second_reused is True
    assert second.manifest_hash == manifest.manifest_hash


def test_failed_attempt_is_preserved_and_fed_to_next_round(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    runner, _ = _runner(tmp_path, FakeDockerRunner(passed=False))
    snapshot_dir = tmp_path / "snapshots" / "dev"
    snapshot_dir.mkdir(parents=True)

    manifest, _ = runner.run(
        request=_request(),
        snapshot=_snapshot_manifest(),
        snapshot_dir=snapshot_dir,
    )

    assert manifest.status == "failed"
    assert [item["error_code"] for item in manifest.attempts] == [
        "MINING-139",
        "MINING-139",
    ]
    assert manifest.attempts[0]["feedback"]["category"] == "execution"
    assert manifest.attempts[0]["feedback"]["repair_hint"]
    assert runner.worker.requests[1].previous_attempt["status"] == "failed"


@pytest.mark.parametrize(
    ("docker_kwargs", "code", "expected_code", "expected_category"),
    [
        (
            {"passed": False},
            "def broken(:\n    pass",
            "MINING-138",
            SandboxFailureCategory.COMPILE.value,
        ),
        (
            {"metrics_text": "{broken"},
            "print('bad metrics')",
            "MINING-140",
            SandboxFailureCategory.OUTPUT_FORMAT.value,
        ),
        (
            {"metrics": {**_DEFAULT_METRICS, "coverage": "0.8"}},
            "print('wrong metric type')",
            "MINING-140",
            SandboxFailureCategory.OUTPUT_FORMAT.value,
        ),
        (
            {"metrics": {**_DEFAULT_METRICS, "valid_days": 0}},
            "print('empty index')",
            "MINING-141",
            SandboxFailureCategory.INDEX.value,
        ),
        (
            {"metrics": {**_DEFAULT_METRICS, "coverage": 1.2}},
            "print('bad coverage')",
            "MINING-142",
            SandboxFailureCategory.COVERAGE.value,
        ),
        (
            {"metrics": {**_DEFAULT_METRICS, "rank_ic": float("nan")}},
            "print('bad number')",
            "MINING-143",
            SandboxFailureCategory.NUMERIC.value,
        ),
        (
            {"timed_out": True},
            "print('timeout')",
            "MINING-145",
            SandboxFailureCategory.RESOURCE.value,
        ),
    ],
)
def test_research_failures_have_structured_categories(
    tmp_path: Path,
    monkeypatch,
    docker_kwargs,
    code,
    expected_code,
    expected_category,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    policy = replace(
        load_sandbox_policy(),
        enabled=True,
        snapshot_root="snapshots",
        workspace_root="work",
        review_root="review",
    )
    runner = _single_proposal_runner(
        policy,
        FakeDockerRunner(**docker_kwargs),
        code=code,
    )
    snapshot_dir = tmp_path / "snapshots" / "dev"
    snapshot_dir.mkdir(parents=True)

    manifest, _ = runner.run(
        request=replace(_request(), max_iterations=1),
        snapshot=_snapshot_manifest(),
        snapshot_dir=snapshot_dir,
    )

    attempt = manifest.attempts[0]
    assert attempt["status"] == "failed"
    assert attempt["error_code"] == expected_code
    assert attempt["feedback"]["error_code"] == expected_code
    assert attempt["feedback"]["category"] == expected_category


def test_manual_promotion_and_review_copy_never_create_formal_candidate(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    runner, policy = _runner(tmp_path, FakeDockerRunner())
    snapshot_dir = tmp_path / "snapshots" / "dev"
    snapshot_dir.mkdir(parents=True)
    manifest, _ = runner.run(
        request=_request(),
        snapshot=_snapshot_manifest(),
        snapshot_dir=snapshot_dir,
    )
    artifact_dir = policy.workspace_root_path / _request().sandbox_run_id
    promotion = PromotionRequest(
        sandbox_run_id=manifest.sandbox_run_id,
        artifact_manifest_hash=manifest.manifest_hash,
        reviewer="reviewer-a",
        reason="转写为受控 DSL 并重新评估",
        target_type="dsl",
        target_reference="rank(close)",
    )

    first = record_manual_promotion(
        policy=policy,
        artifact_dir=artifact_dir,
        request=promotion,
    )
    second = record_manual_promotion(
        policy=policy,
        artifact_dir=artifact_dir,
        request=promotion,
    )
    copied = copy_artifacts_for_review(
        policy=policy,
        artifact_dir=artifact_dir,
    )

    assert first["formal_candidate_created"] is False
    assert second["reused"] is True
    assert copied.is_dir()
    assert verify_artifact_tree(copied).manifest_hash == manifest.manifest_hash


def test_artifact_tampering_is_detected(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    runner, policy = _runner(tmp_path, FakeDockerRunner())
    snapshot_dir = tmp_path / "snapshots" / "dev"
    snapshot_dir.mkdir(parents=True)
    manifest, _ = runner.run(
        request=_request(),
        snapshot=_snapshot_manifest(),
        snapshot_dir=snapshot_dir,
    )
    artifact_dir = policy.workspace_root_path / manifest.sandbox_run_id
    proposal = artifact_dir / "iterations/001/proposal.py"
    os.chmod(artifact_dir / "iterations/001", 0o755)
    os.chmod(proposal, 0o644)
    proposal.write_text("tampered", encoding="utf-8")

    with pytest.raises(SandboxArtifactError):
        verify_artifact_tree(artifact_dir)


def test_resume_reuses_persisted_code_without_new_model_call(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    runner, policy = _runner(tmp_path, FakeDockerRunner())
    snapshot_dir = tmp_path / "snapshots" / "dev"
    snapshot_dir.mkdir(parents=True)
    request = replace(_request(), max_iterations=1)
    artifact_dir = policy.workspace_root_path / request.sandbox_run_id
    iteration_dir = artifact_dir / "iterations" / "001"
    iteration_dir.mkdir(parents=True)
    code = "print('persisted')"
    (iteration_dir / "proposal.py").write_text(code, encoding="utf-8")
    (artifact_dir / "run-state.json").write_text(
        json.dumps(
            {
                "schema_version": "mining_sandbox_research_v1",
                "request_hash": request.request_hash,
                "status": "running",
                "attempts": [
                    {
                        "iteration": 1,
                        "proposal_id": "persisted_proposal",
                        "code_file": "iterations/001/proposal.py",
                        "code_hash": __import__("hashlib").sha256(
                            code.encode("utf-8")
                        ).hexdigest(),
                        "rationale": "崩溃前已生成",
                        "model": request.llm_model,
                        "prompt_version": request.prompt_version,
                        "usage": {},
                        "status": "proposed",
                        "metrics": None,
                        "error_code": None,
                        "image_id": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest, reused = runner.run(
        request=request,
        snapshot=_snapshot_manifest(),
        snapshot_dir=snapshot_dir,
    )

    assert reused is False
    assert manifest.status == "completed"
    assert runner.worker.requests == []


def _database(path: Path) -> None:
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE SCHEMA md")
        conn.execute(
            "CREATE TABLE md.calendar(trade_date DATE, is_open INTEGER)"
        )
        conn.execute(
            "INSERT INTO md.calendar VALUES "
            "('2019-12-27',1),('2019-12-30',1),('2019-12-31',1)"
        )
        conn.execute(
            """
            CREATE TABLE md.daily_hfq(
                stock_code VARCHAR, trade_date DATE,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
                vwap DOUBLE, volume_share DOUBLE, amount_cny DOUBLE,
                ret DOUBLE, turnover_rate DOUBLE, circ_mv DOUBLE,
                total_mv DOUBLE
            )
            """
        )
        conn.execute(
            "INSERT INTO md.daily_hfq VALUES "
            "('SH600000','2019-12-27',1,2,0.5,1.5,1.4,100,1000,0.1,1,2,3),"
            "('SH600000','2019-12-30',1.5,2.5,1,2,2.1,110,1100,0.2,1,2,3),"
            "('SH600000','2019-12-31',2,3,1.5,2.5,2.4,120,1200,0.25,1,2,3)"
        )
        conn.execute(
            "CREATE TABLE md.fwd_return("
            "stock_code VARCHAR, trade_date DATE, fwd_ret_open_5 DOUBLE)"
        )
        conn.execute(
            "INSERT INTO md.fwd_return VALUES "
            "('SH600000','2019-12-27',0.01),"
            "('SH600000','2019-12-30',0.02),"
            "('SH600000','2019-12-31',0.03)"
        )


@pytest.mark.skipif(
    subprocess.run(
        ["docker", "image", "inspect", "quant-mining-sandbox:s5"],
        capture_output=True,
    ).returncode
    != 0,
    reason="沙箱镜像不可用",
)
def test_real_sandbox_reads_qlib_and_returns_research_metrics(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    database = tmp_path / "quant.duckdb"
    _database(database)
    context = FrozenSearchContext(data_fingerprint="e" * 64)
    snapshot_dir = tmp_path / "snapshots" / "dev"
    snapshot, _ = build_qlib_snapshot(
        context=context,
        db_file=database,
        output_dir=snapshot_dir,
        minimum_free_bytes=1,
    )
    policy = replace(
        load_sandbox_policy(),
        enabled=True,
        snapshot_root="snapshots",
        workspace_root="work",
        review_root="review",
        timeout_seconds=30,
    )
    code = """
import json
from pathlib import Path
from qlib import init
from qlib.data import D
init(provider_uri="/snapshot", region="cn", kernels=1)
frame = D.features(["SH600000"], ["$close", "$label"])
frame = frame.dropna()
rank_ic = float(frame["$close"].corr(frame["$label"], method="spearman"))
Path("/workspace/metrics.json").write_text(
    json.dumps({
        "rank_ic": rank_ic,
        "icir": rank_ic,
        "coverage": 1.0,
        "valid_days": len(frame),
    }),
    encoding="utf-8",
)
"""
    worker = MockSandboxCodeWorkerClient(
        (
            WorkerCodeProposalResponse(
                proposal_id="proposal_real",
                code=code,
                rationale="读取冻结 Qlib 快照",
                model="deepseek/deepseek-chat",
            ),
        )
    )
    runner = SandboxResearchRunner(
        policy=policy,
        worker=worker,
        docker_runner=DockerSandboxRunner(policy),
    )
    request = SandboxResearchRequest(
        sandbox_run_id="sandboxrun_real_research_0001",
        hypothesis="短期价格变化与未来收益可能相关",
        context_hash=context.context_hash,
        snapshot_manifest_hash=snapshot.manifest_hash,
        llm_model="deepseek/deepseek-chat",
        max_iterations=1,
    )

    manifest, _ = runner.run(
        request=request,
        snapshot=snapshot,
        snapshot_dir=snapshot_dir,
    )

    assert manifest.status == "completed"
    assert manifest.attempts[0]["metrics"]["rank_ic"] == pytest.approx(1.0)
    assert manifest.image_id.startswith("sha256:")
    audit = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "name=mining-sandbox-",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert manifest.attempts[0]["container_name"] not in audit.stdout
