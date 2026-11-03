"""沙箱故障注入、资源边界、产物污染和安全恢复验收。"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

import mining_sandbox.policy as policy_module
from mining_sandbox.artifacts import (
    SandboxArtifactManifest,
    seal_artifact_tree,
    verify_artifact_tree,
)
from mining_sandbox.errors import (
    SandboxArtifactError,
    SandboxKnowledgeError,
    SandboxRunnerError,
)
from mining_sandbox.knowledge import load_sandbox_knowledge
from mining_sandbox.qlib_snapshot import (
    QlibSnapshotManifest,
    verify_qlib_snapshot,
)
from mining_sandbox.research import SandboxResearchRequest, SandboxResearchRunner
from mining_sandbox.runtime import DockerSandboxRunner, SandboxExecutionResult
from mining_sandbox.worker_client import MockSandboxCodeWorkerClient
from mining_worker.protocol_code import WorkerCodeProposalResponse


_METRICS = {
    "rank_ic": 0.03,
    "icir": 0.4,
    "coverage": 1.0,
    "valid_days": 3,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(tmp_path: Path) -> tuple[QlibSnapshotManifest, Path]:
    root = tmp_path / "snapshots" / "dev"
    calendar = root / "calendars" / "day.txt"
    instruments = root / "instruments" / "all.txt"
    calendar.parent.mkdir(parents=True)
    instruments.parent.mkdir(parents=True)
    calendar.write_text(
        "2019-12-27\n2019-12-30\n2019-12-31\n",
        encoding="utf-8",
    )
    instruments.write_text(
        "SH600000\t2019-12-27\t2019-12-31\n",
        encoding="utf-8",
    )
    files = {
        "calendars/day.txt": {
            "bytes": calendar.stat().st_size,
            "sha256": _sha(calendar),
        },
        "instruments/all.txt": {
            "bytes": instruments.stat().st_size,
            "sha256": _sha(instruments),
        },
    }
    manifest = QlibSnapshotManifest(
        context_hash="a" * 64,
        data_version="security_test_v1",
        universe="all_a",
        start_date="2019-12-27",
        end_date="2019-12-31",
        fields=("close", "label"),
        calendar_sha256=files["calendars/day.txt"]["sha256"],
        instruments_sha256=files["instruments/all.txt"]["sha256"],
        files=files,
        row_count=3,
        total_bytes=sum(int(item["bytes"]) for item in files.values()),
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest.to_dict()),
        encoding="utf-8",
    )
    return manifest, root


def _policy(tmp_path: Path):
    return replace(
        policy_module.load_sandbox_policy(),
        enabled=True,
        snapshot_root="snapshots",
        workspace_root="work",
        review_root="review",
        output_mb=1,
        workspace_mb=2,
    )


def _request(
    snapshot: QlibSnapshotManifest,
    *,
    run_id: str = "sandboxrun_security_0001",
) -> SandboxResearchRequest:
    return SandboxResearchRequest(
        sandbox_run_id=run_id,
        hypothesis="短期价格变化与未来收益可能相关",
        context_hash=snapshot.context_hash,
        snapshot_manifest_hash=snapshot.manifest_hash,
        llm_model="deepseek/deepseek-chat",
        max_iterations=1,
    )


def _write_outputs(workspace: Path) -> None:
    pd.DataFrame(
        {
            "date": ["2019-12-27", "2019-12-30", "2019-12-31"],
            "instrument": ["SH600000", "SH600000", "SH600000"],
            "value": [0.1, 0.2, 0.3],
        }
    ).to_parquet(workspace / "factor.parquet", index=False)
    (workspace / "metrics.json").write_text(
        json.dumps(_METRICS),
        encoding="utf-8",
    )


class _ResultDocker:
    def __init__(self, action=None) -> None:
        self.action = action
        self.calls = 0

    def resolve_image_id(self) -> str:
        return "sha256:" + "d" * 64

    def run(self, *, run_id, snapshot_dir, workspace_dir, command):
        self.calls += 1
        workspace = Path(workspace_dir)
        if self.action is not None:
            self.action(workspace)
        else:
            _write_outputs(workspace)
        return SandboxExecutionResult(
            run_id=run_id,
            container_name=f"test-{run_id}",
            image_id="sha256:" + "d" * 64,
            exit_code=0,
            timed_out=False,
            duration_ms=1.0,
            stdout="ok\n",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
        )


class _RaisingDocker:
    def resolve_image_id(self) -> str:
        return "sha256:" + "d" * 64

    def run(self, **kwargs):
        raise SandboxRunnerError("注入容器启动故障")


def _worker(code: str = "print('ok')") -> MockSandboxCodeWorkerClient:
    return MockSandboxCodeWorkerClient(
        (
            WorkerCodeProposalResponse(
                proposal_id="proposal_security",
                code=code,
                rationale="安全验收",
                model="deepseek/deepseek-chat",
            ),
        )
    )


def test_runner_uses_immutable_image_and_explicit_resource_flags(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    policy = replace(
        _policy(tmp_path),
        image="python:3.11-slim",
        timeout_seconds=2,
    )
    snapshot = tmp_path / "snapshots" / "dev"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    commands: list[list[str]] = []

    class Process:
        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    class Completed:
        returncode = 0
        stdout = "sha256:" + "e" * 64

    def inspect(args, **kwargs):
        if args[1:3] == ["image", "inspect"]:
            return Completed()
        return Completed()

    runner = DockerSandboxRunner(
        policy,
        popen=lambda args, **kwargs: (
            commands.append(list(args)) or Process()
        ),
        run=inspect,
    )
    built, _, _, _ = runner.build_command(
        run_id="sandboxrun_security_flags",
        snapshot_dir=snapshot,
        workspace_dir=tmp_path / "work" / "flags",
        command=("python", "-c", "print('ok')"),
    )

    assert "--network=none" in built
    assert "--read-only" in built
    assert "--cap-drop=ALL" in built
    assert "--security-opt=no-new-privileges" in built
    assert f"--user={policy.user}" in built
    assert "core=0:0" in built
    assert "fsize=2097152:2097152" in built
    assert not any("docker.sock" in item for item in built)
    mounts = [item for item in built if item.startswith("type=bind,src=")]
    assert len(mounts) == 2
    assert all(
        str(tmp_path / "snapshots") in item
        or str(tmp_path / "work") in item
        for item in mounts
    )

    result = runner.run(
        run_id="sandboxrun_security_flags",
        snapshot_dir=snapshot,
        workspace_dir=tmp_path / "work" / "flags-run",
        command=("python", "-c", "print('ok')"),
    )

    assert result.limit_exceeded is None
    assert "sha256:" + "e" * 64 in commands[0]
    assert "python:3.11-slim" not in commands[0]


def test_runner_terminates_output_and_workspace_overruns(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    policy = replace(
        _policy(tmp_path),
        image="python:3.11-slim",
        timeout_seconds=2,
    )
    snapshot = tmp_path / "snapshots" / "dev"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    cleanup: list[list[str]] = []

    class EndlessProcess:
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("docker", timeout)

        def kill(self):
            return None

    class Completed:
        returncode = 0
        stdout = "sha256:" + "e" * 64

    def image_inspect(args, **kwargs):
        if args[1:3] == ["image", "inspect"]:
            return Completed()
        cleanup.append(list(args))
        return Completed()

    def output_popen(args, *, stdout, **kwargs):
        stdout.write(b"x" * (2 * 1024 * 1024))
        stdout.flush()
        return EndlessProcess()

    output_result = DockerSandboxRunner(
        policy,
        popen=output_popen,
        run=image_inspect,
    ).run(
        run_id="sandboxrun_security_output",
        snapshot_dir=snapshot,
        workspace_dir=tmp_path / "work" / "output",
        command=("python", "-c", "print('ok')"),
    )
    assert output_result.limit_exceeded == "output"
    assert output_result.timed_out is False
    assert any(item[1] == "kill" for item in cleanup)

    workspace_path = tmp_path / "work" / "workspace"

    def workspace_popen(args, **kwargs):
        workspace_path.mkdir(parents=True, exist_ok=True)
        (workspace_path / "oversize.bin").write_bytes(
            b"x" * (3 * 1024 * 1024)
        )
        return EndlessProcess()

    workspace_result = DockerSandboxRunner(
        policy,
        popen=workspace_popen,
        run=image_inspect,
    ).run(
        run_id="sandboxrun_security_workspace",
        snapshot_dir=snapshot,
        workspace_dir=workspace_path,
        command=("python", "-c", "print('ok')"),
    )
    assert workspace_result.limit_exceeded == "workspace"


def test_snapshot_and_artifact_verifiers_reject_extra_and_links(tmp_path: Path):
    snapshot, snapshot_dir = _snapshot(tmp_path)
    assert verify_qlib_snapshot(snapshot_dir, snapshot).manifest_hash == (
        snapshot.manifest_hash
    )
    (snapshot_dir / "extra.bin").write_bytes(b"x")
    with pytest.raises(Exception, match="文件集合变化"):
        verify_qlib_snapshot(snapshot_dir, snapshot)
    (snapshot_dir / "extra.bin").unlink()
    os.symlink(
        snapshot_dir / "calendars" / "day.txt",
        snapshot_dir / "calendar-link",
    )
    with pytest.raises(Exception, match="符号链接"):
        verify_qlib_snapshot(snapshot_dir, snapshot)

    artifact_dir = tmp_path / "artifact"
    attempt_path = artifact_dir / "iterations" / "001" / "proposal.py"
    attempt_path.parent.mkdir(parents=True)
    attempt_path.write_text("print('ok')\n", encoding="utf-8")
    files = {
        "iterations/001/proposal.py": {
            "bytes": attempt_path.stat().st_size,
            "sha256": _sha(attempt_path),
        }
    }
    manifest = SandboxArtifactManifest(
        sandbox_run_id="sandboxrun_security_artifact",
        request_hash="a" * 64,
        context_hash="b" * 64,
        snapshot_manifest_hash="c" * 64,
        hypothesis_hash="d" * 64,
        image_id="sha256:" + "e" * 64,
        status="completed",
        attempts=({"iteration": 1, "status": "passed"},),
        environment={},
        files=files,
    )
    seal_artifact_tree(artifact_dir, manifest)
    assert verify_artifact_tree(artifact_dir).manifest_hash == (
        manifest.manifest_hash
    )
    os.chmod(artifact_dir, 0o755)
    (artifact_dir / "extra.bin").write_bytes(b"x")
    with pytest.raises(SandboxArtifactError, match="文件集合变化"):
        verify_artifact_tree(artifact_dir)
    (artifact_dir / "extra.bin").unlink()
    os.symlink(attempt_path, artifact_dir / "link")
    with pytest.raises(SandboxArtifactError, match="符号链接"):
        verify_artifact_tree(artifact_dir)


@pytest.mark.parametrize(
    ("action", "expected_violation"),
    [
        (
            lambda workspace: (
                _write_outputs(workspace),
                (workspace / "proposal.py").write_text(
                    "print('changed')\n",
                    encoding="utf-8",
                ),
            ),
            False,
        ),
        (
            lambda workspace: (
                _write_outputs(workspace),
                (workspace / "leak").symlink_to("/tmp"),
            ),
            True,
        ),
    ],
)
def test_research_rejects_runtime_code_and_workspace_pollution(
    tmp_path: Path,
    monkeypatch,
    action,
    expected_violation,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    snapshot, snapshot_dir = _snapshot(tmp_path)
    runner = SandboxResearchRunner(
        policy=_policy(tmp_path),
        worker=_worker(),
        docker_runner=_ResultDocker(action),
    )

    manifest, _ = runner.run(
        request=_request(snapshot),
        snapshot=snapshot,
        snapshot_dir=snapshot_dir,
    )

    attempt = manifest.attempts[0]
    assert manifest.status == "failed"
    assert attempt["error_code"] == "MINING-144"
    assert attempt["status"] == "failed"
    assert attempt["metrics"] is None
    assert bool(attempt["workspace_violations"]) is expected_violation
    if expected_violation:
        assert attempt["executed_code_hash"] == attempt["code_hash"]
    else:
        assert attempt["executed_code_hash"] != attempt["code_hash"]
    artifact_dir = (
        _policy(tmp_path).workspace_root_path / manifest.sandbox_run_id
    )
    verify_artifact_tree(artifact_dir)
    proposal = artifact_dir / "iterations" / "001" / "proposal.py"
    assert _sha(proposal) == attempt["code_hash"]


def test_sealed_knowledge_pollution_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    snapshot, snapshot_dir = _snapshot(tmp_path)
    runner = SandboxResearchRunner(
        policy=_policy(tmp_path),
        worker=_worker(),
        docker_runner=_ResultDocker(),
    )
    manifest, _ = runner.run(
        request=_request(snapshot, run_id="sandboxrun_security_knowledge"),
        snapshot=snapshot,
        snapshot_dir=snapshot_dir,
    )
    artifact_dir = (
        _policy(tmp_path).workspace_root_path / manifest.sandbox_run_id
    )
    knowledge_path = artifact_dir / "knowledge.json"
    os.chmod(artifact_dir, 0o755)
    os.chmod(knowledge_path, 0o644)
    payload = json.loads(knowledge_path.read_text(encoding="utf-8"))
    payload["entries"][0]["summary"] = "污染后的知识"
    knowledge_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SandboxKnowledgeError, match="产物校验失败"):
        load_sandbox_knowledge(artifact_dir)


def test_runner_fault_is_isolated_and_following_run_recovers(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    snapshot, snapshot_dir = _snapshot(tmp_path)
    first = SandboxResearchRunner(
        policy=_policy(tmp_path),
        worker=_worker(),
        docker_runner=_RaisingDocker(),
    )

    with pytest.raises(SandboxRunnerError, match="注入"):
        first.run(
            request=_request(snapshot, run_id="sandboxrun_security_fault"),
            snapshot=snapshot,
            snapshot_dir=snapshot_dir,
        )

    resumed_worker = MockSandboxCodeWorkerClient(())
    resumed = SandboxResearchRunner(
        policy=_policy(tmp_path),
        worker=resumed_worker,
        docker_runner=_ResultDocker(),
    )
    resumed_manifest, resumed_reused = resumed.run(
        request=_request(snapshot, run_id="sandboxrun_security_fault"),
        snapshot=snapshot,
        snapshot_dir=snapshot_dir,
    )
    assert resumed_reused is False
    assert resumed_manifest.status == "completed"
    assert resumed_worker.requests == []

    second = SandboxResearchRunner(
        policy=_policy(tmp_path),
        worker=_worker(),
        docker_runner=_ResultDocker(),
    )
    manifest, reused = second.run(
        request=_request(snapshot, run_id="sandboxrun_security_healthy"),
        snapshot=snapshot,
        snapshot_dir=snapshot_dir,
    )

    assert reused is False
    assert manifest.status == "completed"
    assert manifest.sandbox_run_id == "sandboxrun_security_healthy"


@pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="Docker 不可用",
)
@pytest.mark.skipif(
    subprocess.run(
        ["docker", "image", "inspect", "quant-mining-sandbox:s5"],
        capture_output=True,
    ).returncode
    != 0,
    reason="沙箱镜像不可用",
)
def test_locked_sandbox_image_enforces_runtime_boundaries(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    policy = replace(
        _policy(tmp_path),
        timeout_seconds=20,
    )
    snapshot = tmp_path / "snapshots" / "dev"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    code = (
        "import pathlib,socket\n"
        "try:\n"
        " socket.create_connection(('1.1.1.1',53),timeout=1)\n"
        " offline=False\n"
        "except OSError:\n"
        " offline=True\n"
        "try:\n"
        " pathlib.Path('/root-write').write_text('x')\n"
        " root_readonly=False\n"
        "except OSError:\n"
        " root_readonly=True\n"
        "try:\n"
        " pathlib.Path('/snapshot/write').write_text('x')\n"
        " snapshot_readonly=False\n"
        "except OSError:\n"
        " snapshot_readonly=True\n"
        "workspace_write=pathlib.Path('/workspace/ok').write_text('x') == 1\n"
        "uid_ok=__import__('os').getuid() != 0\n"
        "print(offline, root_readonly, snapshot_readonly, "
        "workspace_write, uid_ok)\n"
        "raise SystemExit(0 if offline and root_readonly and "
        "snapshot_readonly and workspace_write and uid_ok else 2)\n"
    )

    result = DockerSandboxRunner(policy).run(
        run_id="sandboxrun_security_locked",
        snapshot_dir=snapshot,
        workspace_dir=tmp_path / "work" / "locked",
        command=("python", "-c", code),
    )

    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "True True True True True"
    assert result.image_id.startswith("sha256:")
