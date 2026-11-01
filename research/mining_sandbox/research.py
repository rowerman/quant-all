"""RD-Agent 代码提案与 Qlib 沙箱执行的有限研究循环。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from mining.contract_types import (
    canonical_json,
    identifier,
    sha256_json,
    text,
    utc_now_iso,
)
from mining_sandbox.artifacts import (
    SandboxArtifactManifest,
    seal_artifact_tree,
    verify_artifact_tree,
)
from mining_sandbox.errors import SandboxRunnerError, SandboxSnapshotError
from mining_sandbox.feedback import (
    SandboxCodeFeedback,
    SandboxFailureCategory,
    SandboxFeedbackError,
)
from mining_sandbox.logging_conf import get_sandbox_logger, log_sandbox_feedback
from mining_sandbox.policy import SandboxPolicy
from mining_sandbox.qlib_snapshot import QlibSnapshotManifest
from mining_sandbox.runtime import DockerSandboxRunner
from mining_sandbox.worker_client import SandboxCodeWorkerClient
from mining_worker.protocol_code import (
    CODE_PROMPT_VERSION,
    WorkerCodeProposalRequest,
)

SANDBOX_RESEARCH_SCHEMA_VERSION = "mining_sandbox_research_v1"
_METRICS_FIELDS = {"rank_ic", "icir", "coverage", "valid_days"}
_LOGGER = get_sandbox_logger("mining_sandbox.research")


@dataclass(frozen=True)
class SandboxResearchRequest:
    """一次有限、可恢复的沙箱代码研究请求。"""

    sandbox_run_id: str
    hypothesis: str
    context_hash: str
    snapshot_manifest_hash: str
    llm_model: str
    max_iterations: int = 3
    max_code_bytes: int = 32768
    prompt_version: str = CODE_PROMPT_VERSION
    schema_version: str = SANDBOX_RESEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SANDBOX_RESEARCH_SCHEMA_VERSION:
            raise SandboxRunnerError("沙箱研究请求版本不匹配")
        object.__setattr__(
            self,
            "sandbox_run_id",
            identifier(self.sandbox_run_id, "sandbox_run_id"),
        )
        object.__setattr__(
            self,
            "hypothesis",
            text(self.hypothesis, "hypothesis", minimum=8, maximum=4000),
        )
        for name in ("context_hash", "snapshot_manifest_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise SandboxRunnerError(f"{name} 必须是 SHA-256")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or not 1 <= self.max_iterations <= 3
        ):
            raise SandboxRunnerError("max_iterations 必须在 1～3")
        if (
            isinstance(self.max_code_bytes, bool)
            or not isinstance(self.max_code_bytes, int)
            or not 1024 <= self.max_code_bytes <= 65536
        ):
            raise SandboxRunnerError("max_code_bytes 必须在 1024～65536")
        if not isinstance(self.llm_model, str) or not self.llm_model.strip():
            raise SandboxRunnerError("llm_model 不能为空")
        if self.prompt_version != CODE_PROMPT_VERSION:
            raise SandboxRunnerError("prompt_version 不匹配")

    def material(self) -> dict[str, Any]:
        """返回不包含宿主路径的稳定请求内容。"""

        return {
            "schema_version": self.schema_version,
            "sandbox_run_id": self.sandbox_run_id,
            "hypothesis": self.hypothesis,
            "context_hash": self.context_hash,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "llm_model": self.llm_model,
            "max_iterations": self.max_iterations,
            "max_code_bytes": self.max_code_bytes,
            "prompt_version": self.prompt_version,
        }

    @property
    def request_hash(self) -> str:
        return sha256_json(self.material())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(dict(payload)), encoding="utf-8")
    os.replace(temporary, path)


def _read_metrics(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SandboxFeedbackError(
            SandboxCodeFeedback.create(
                SandboxFailureCategory.OUTPUT_FORMAT,
                summary=f"沙箱指标不可读：{type(exc).__name__}",
                repair_hint="重新写出合法的 metrics.json，字段固定为四个研究指标",
            )
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != _METRICS_FIELDS:
        raise SandboxFeedbackError(
            SandboxCodeFeedback.create(
                SandboxFailureCategory.OUTPUT_FORMAT,
                summary="沙箱指标字段不符合契约",
                repair_hint=(
                    "metrics.json 必须且只能包含 rank_ic、icir、coverage、valid_days"
                ),
            )
        )
    if isinstance(payload["valid_days"], bool) or not isinstance(
        payload["valid_days"],
        int,
    ):
        raise SandboxFeedbackError(
            SandboxCodeFeedback.create(
                SandboxFailureCategory.OUTPUT_FORMAT,
                summary="valid_days 类型不符合契约",
                repair_hint="valid_days 必须写成大于零的整数",
            )
        )
    if payload["valid_days"] <= 0:
        raise SandboxFeedbackError(
            SandboxCodeFeedback.create(
                SandboxFailureCategory.INDEX,
                summary="valid_days 没有形成有效交易日索引",
                repair_hint="对齐冻结交易日和股票索引，并确认至少有一个有效观察日",
            )
        )
    for name in ("rank_ic", "icir", "coverage"):
        value = payload[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise SandboxFeedbackError(
                SandboxCodeFeedback.create(
                    SandboxFailureCategory.OUTPUT_FORMAT,
                    summary=f"{name} 类型不符合契约",
                    repair_hint=f"{name} 必须写成 JSON 数值",
                )
            )
        if not math.isfinite(float(value)):
            raise SandboxFeedbackError(
                SandboxCodeFeedback.create(
                    SandboxFailureCategory.NUMERIC,
                    summary=f"{name} 不是有限数值",
                    repair_hint=f"修正 {name} 的计算，过滤空值和不可计算的样本",
                )
            )
    if not -1 <= float(payload["rank_ic"]) <= 1:
        raise SandboxFeedbackError(
            SandboxCodeFeedback.create(
                SandboxFailureCategory.NUMERIC,
                summary="rank_ic 超出合法范围",
                repair_hint="检查收益和因子方向，确保 rank_ic 落在 -1～1",
            )
        )
    if not 0 <= float(payload["coverage"]) <= 1:
        raise SandboxFeedbackError(
            SandboxCodeFeedback.create(
                SandboxFailureCategory.COVERAGE,
                summary="coverage 超出合法范围",
                repair_hint="按冻结股票和交易日重算覆盖率，确保结果落在 0～1",
            )
        )
    return {
        "rank_ic": float(payload["rank_ic"]),
        "icir": float(payload["icir"]),
        "coverage": float(payload["coverage"]),
        "valid_days": payload["valid_days"],
    }


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _compile_feedback(code: str) -> SandboxCodeFeedback | None:
    """静态检查提案语法，不执行提案代码。"""

    try:
        compile(code, "<proposal>", "exec")
    except (SyntaxError, ValueError, RecursionError) as exc:
        line = getattr(exc, "lineno", None)
        location = f"第 {line} 行" if isinstance(line, int) and line > 0 else ""
        return SandboxCodeFeedback.create(
            SandboxFailureCategory.COMPILE,
            summary=f"提案代码无法通过静态编译 {location}".strip(),
            repair_hint="修复 Python 语法并保持脚本可独立运行",
        )
    return None


class SandboxResearchRunner:
    """协调 RD-Agent 代码提案和隔离容器执行的有限循环。"""

    def __init__(
        self,
        *,
        policy: SandboxPolicy,
        worker: SandboxCodeWorkerClient,
        docker_runner: DockerSandboxRunner | None = None,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.policy = policy
        self.worker = worker
        self.docker_runner = docker_runner or DockerSandboxRunner(policy)
        self.clock = clock

    def run(
        self,
        *,
        request: SandboxResearchRequest,
        snapshot: QlibSnapshotManifest,
        snapshot_dir: Path | str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[SandboxArtifactManifest, bool]:
        """执行或恢复一次研究运行。"""

        if snapshot.manifest_hash != request.snapshot_manifest_hash:
            raise SandboxSnapshotError("研究请求绑定了不同快照")
        if snapshot.context_hash != request.context_hash:
            raise SandboxSnapshotError("研究请求上下文与快照不一致")
        artifact_dir = self.policy.workspace_root_path / request.sandbox_run_id
        if artifact_dir.exists() and (artifact_dir / "manifest.json").is_file():
            manifest = verify_artifact_tree(artifact_dir)
            if (
                manifest.request_hash != request.request_hash
                or manifest.snapshot_manifest_hash != snapshot.manifest_hash
            ):
                raise SandboxRunnerError("已有沙箱运行与请求不一致")
            return manifest, True
        state_path = artifact_dir / "run-state.json"
        if artifact_dir.exists():
            if not state_path.is_file():
                raise SandboxRunnerError("不完整的沙箱目录缺少运行状态")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("request_hash") != request.request_hash:
                raise SandboxRunnerError("沙箱恢复请求与已有运行不一致")
            attempts = list(state.get("attempts") or [])
        else:
            artifact_dir.mkdir(parents=True, exist_ok=False)
            attempts = []
            state = {
                "schema_version": SANDBOX_RESEARCH_SCHEMA_VERSION,
                "request_hash": request.request_hash,
                "status": "running",
                "attempts": attempts,
            }
            _atomic_json(state_path, state)

        completed = {
            int(item["iteration"]): item
            for item in attempts
            if item.get("status") in {"passed", "failed"}
        }
        latest_attempt = next(
            (
                item
                for item in reversed(attempts)
                if item.get("status") in {"passed", "failed"}
            ),
            None,
        )
        image_id = ""
        for iteration in range(1, request.max_iterations + 1):
            if should_cancel is not None and should_cancel():
                raise SandboxRunnerError("沙箱研究已取消")
            if iteration in completed:
                image_id = str(completed[iteration].get("image_id") or image_id)
                continue
            iteration_dir = artifact_dir / "iterations" / f"{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            code_path = iteration_dir / "proposal.py"
            proposal_meta = next(
                (
                    item
                    for item in attempts
                    if int(item["iteration"]) == iteration
                ),
                None,
            )
            if not code_path.is_file() or proposal_meta is None:
                proposal = self.worker.propose(
                    WorkerCodeProposalRequest(
                        request_id=f"{request.sandbox_run_id}:code:{iteration}",
                        run_id=request.sandbox_run_id,
                        iteration=iteration,
                        hypothesis=request.hypothesis,
                        snapshot={
                            "context_hash": snapshot.context_hash,
                            "manifest_hash": snapshot.manifest_hash,
                            "start_date": snapshot.start_date,
                            "end_date": snapshot.end_date,
                            "fields": list(snapshot.fields),
                            "image_id": self.policy.image,
                        },
                        previous_attempt=(
                            {
                                "iteration": int(latest_attempt["iteration"]),
                                "code_hash": str(latest_attempt["code_hash"]),
                                "status": str(latest_attempt["status"]),
                                "metrics": latest_attempt.get("metrics"),
                                "error_code": latest_attempt.get("error_code"),
                            }
                            if latest_attempt is not None
                            else None
                        ),
                        max_code_bytes=request.max_code_bytes,
                        llm_model=request.llm_model,
                    ),
                    should_cancel=should_cancel,
                )
                if len(proposal.code.encode("utf-8")) > request.max_code_bytes:
                    raise SandboxRunnerError("代码提案超过请求上限")
                if proposal.model != request.llm_model:
                    raise SandboxRunnerError("代码提案模型与请求不一致")
                code_path.write_text(proposal.code, encoding="utf-8")
                proposal_meta = {
                    "iteration": iteration,
                    "proposal_id": proposal.proposal_id,
                    "code_file": code_path.relative_to(artifact_dir).as_posix(),
                    "code_hash": _code_hash(proposal.code),
                    "rationale": proposal.rationale,
                    "model": proposal.model,
                    "prompt_version": proposal.prompt_version,
                    "usage": dict(proposal.usage),
                    "status": "proposed",
                    "metrics": None,
                    "error_code": None,
                    "feedback": None,
                    "image_id": None,
                }
                attempts.append(proposal_meta)
                _atomic_json(state_path, {**state, "attempts": attempts})

            code = code_path.read_text(encoding="utf-8")
            compile_feedback = _compile_feedback(code)
            result = self.docker_runner.run(
                run_id=f"{request.sandbox_run_id}-{iteration:03d}",
                snapshot_dir=snapshot_dir,
                workspace_dir=iteration_dir,
                command=("python", "/workspace/proposal.py"),
            )
            image_id = result.image_id
            (iteration_dir / "stdout.log").write_text(
                result.stdout,
                encoding="utf-8",
            )
            (iteration_dir / "stderr.log").write_text(
                result.stderr,
                encoding="utf-8",
            )
            metrics = None
            feedback = None
            if result.timed_out:
                feedback = SandboxCodeFeedback.create(
                    SandboxFailureCategory.RESOURCE,
                    summary="沙箱执行超过资源时限",
                    repair_hint="减少计算量或数据读取范围，避免无限循环",
                )
            elif result.exit_code != 0:
                feedback = compile_feedback or SandboxCodeFeedback.create(
                    SandboxFailureCategory.EXECUTION,
                    summary=f"沙箱执行以非零状态退出：{result.exit_code}",
                    repair_hint="检查运行时异常并修复脚本中的执行路径",
                )
            else:
                try:
                    metrics = dict(_read_metrics(iteration_dir / "metrics.json"))
                except SandboxFeedbackError as exc:
                    feedback = exc.feedback
            error_code = feedback.error_code if feedback is not None else None
            attempt = {
                **proposal_meta,
                "status": "passed" if error_code is None else "failed",
                "metrics": metrics,
                "error_code": error_code,
                "feedback": feedback.to_dict() if feedback is not None else None,
                "image_id": result.image_id,
                "container_name": result.container_name,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_ms": result.duration_ms,
                "stdout_file": (
                    iteration_dir / "stdout.log"
                ).relative_to(artifact_dir).as_posix(),
                "stderr_file": (
                    iteration_dir / "stderr.log"
                ).relative_to(artifact_dir).as_posix(),
                "metrics_file": (
                    "iterations/"
                    f"{iteration:03d}/metrics.json"
                    if metrics is not None
                    else None
                ),
            }
            attempts = [
                item
                for item in attempts
                if int(item["iteration"]) != iteration
            ]
            attempts.append(attempt)
            attempts.sort(key=lambda item: int(item["iteration"]))
            latest_attempt = attempt
            _atomic_json(state_path, {**state, "attempts": attempts})
            if feedback is not None:
                log_sandbox_feedback(
                    _LOGGER,
                    run_id=request.sandbox_run_id,
                    iteration=iteration,
                    code_hash=str(attempt["code_hash"]),
                    feedback=feedback,
                )

        if not image_id:
            raise SandboxRunnerError("沙箱研究没有产生容器执行记录")
        status = (
            "completed"
            if any(item["status"] == "passed" for item in attempts)
            else "failed"
        )
        _atomic_json(
            state_path,
            {
                **state,
                "status": status,
                "completed_at": self.clock(),
                "attempts": attempts,
            },
        )
        files = {
            path.relative_to(artifact_dir).as_posix(): {
                "sha256": _file_sha(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(artifact_dir.rglob("*"))
            if path.is_file()
        }
        manifest = SandboxArtifactManifest(
            sandbox_run_id=request.sandbox_run_id,
            request_hash=request.request_hash,
            context_hash=request.context_hash,
            snapshot_manifest_hash=request.snapshot_manifest_hash,
            hypothesis_hash=hashlib.sha256(
                request.hypothesis.encode("utf-8")
            ).hexdigest(),
            image_id=image_id,
            status=status,
            attempts=tuple(attempts),
            environment={
                "python": "3.11",
                "qlib": "0.9.7",
                "network": "none",
                "read_only_root": True,
                "user": self.policy.user,
                "dependency_lock_sha256": _file_sha(
                    Path(__file__).with_name("requirements.lock")
                ),
                "model": request.llm_model,
                "prompt_version": request.prompt_version,
            },
            files=files,
            created_at=self.clock(),
        )
        seal_artifact_tree(artifact_dir, manifest)
        return manifest, False


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SANDBOX_RESEARCH_SCHEMA_VERSION",
    "SandboxResearchRequest",
    "SandboxResearchRunner",
]
