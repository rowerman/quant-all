"""从当前沙箱运行目录构建、写入和加载知识索引。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from mining.contract_types import canonical_json, sha256_json
from mining_sandbox.artifacts import SandboxArtifactError, verify_artifact_tree
from mining_sandbox.errors import SandboxKnowledgeError
from mining_sandbox.feedback import SandboxCodeFeedback
from mining_sandbox.knowledge_contract import (
    SANDBOX_KNOWLEDGE_SCHEMA_VERSION,
    SandboxKnowledgeArtifact,
    SandboxKnowledgeEntry,
    SandboxKnowledgeSource,
    knowledge_entry_summary,
    knowledge_hash,
    knowledge_iteration,
    knowledge_json_mapping,
    knowledge_relative_path,
    knowledge_text,
)
from mining_sandbox.knowledge_index import (
    SandboxKnowledgeBase,
    SandboxKnowledgeSelection,
)

if TYPE_CHECKING:
    from mining_sandbox.research import SandboxResearchRequest

SANDBOX_KNOWLEDGE_FILE = "knowledge.json"


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attempt_artifact_paths(
    attempt: Mapping[str, Any],
) -> tuple[str, ...]:
    if attempt.get("code_file") is None:
        raise SandboxKnowledgeError("尝试缺少 code_file")
    result = []
    for key in (
        "code_file",
        "evaluation_file",
        "factor_file",
        "metrics_file",
        "stdout_file",
        "stderr_file",
    ):
        value = attempt.get(key)
        if value is not None:
            result.append(
                knowledge_relative_path(value, f"attempt.{key}")
            )
    return tuple(sorted(set(result)))


def _artifact_reference(
    root: Path,
    relative: str,
) -> SandboxKnowledgeArtifact:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise SandboxKnowledgeError(
            f"知识来源超出当前运行：{relative}"
        ) from exc
    if not path.is_file():
        raise SandboxKnowledgeError(f"知识来源文件不存在：{relative}")
    return SandboxKnowledgeArtifact(
        path=relative,
        bytes=path.stat().st_size,
        sha256=_file_sha(path),
    )


def _knowledge_source(
    root: Path,
    *,
    sandbox_run_id: str,
    attempt: Mapping[str, Any],
) -> SandboxKnowledgeSource:
    iteration = knowledge_iteration(
        attempt.get("iteration"),
        "attempt.iteration",
    )
    artifacts = tuple(
        _artifact_reference(root, path)
        for path in _attempt_artifact_paths(attempt)
    )
    material = {
        "schema_version": SANDBOX_KNOWLEDGE_SCHEMA_VERSION,
        "version": 1,
        "kind": "attempt",
        "sandbox_run_id": sandbox_run_id,
        "iteration": iteration,
        "attempt_hash": sha256_json(dict(attempt)),
        "artifacts": [item.to_dict() for item in artifacts],
    }
    return SandboxKnowledgeSource(
        kind="attempt",
        sandbox_run_id=sandbox_run_id,
        iteration=iteration,
        attempt_hash=material["attempt_hash"],
        artifacts=artifacts,
        source_hash=sha256_json(material),
    )


def _knowledge_entry(
    *,
    kind: str,
    task_hash: str,
    sandbox_run_id: str,
    iteration: int,
    sources: Sequence[SandboxKnowledgeSource],
    code_hash: str,
    feedback: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None,
    evaluation_hash: str | None,
) -> SandboxKnowledgeEntry:
    source_iterations = tuple(item.iteration for item in sources)
    summary = knowledge_entry_summary(
        kind=kind,
        iteration=iteration,
        source_iterations=source_iterations,
        feedback=feedback,
    )
    material = {
        "schema_version": SANDBOX_KNOWLEDGE_SCHEMA_VERSION,
        "version": 1,
        "kind": kind,
        "task_hash": task_hash,
        "sandbox_run_id": sandbox_run_id,
        "iteration": iteration,
        "source_iterations": list(source_iterations),
        "code_hash": code_hash,
        "feedback": dict(feedback) if feedback is not None else None,
        "metrics": dict(metrics) if metrics is not None else None,
        "evaluation_hash": evaluation_hash,
        "summary": summary,
        "sources": [item.to_dict() for item in sources],
    }
    entry_hash = sha256_json(material)
    return SandboxKnowledgeEntry(
        kind=kind,
        task_hash=task_hash,
        sandbox_run_id=sandbox_run_id,
        iteration=iteration,
        source_iterations=source_iterations,
        code_hash=code_hash,
        feedback=feedback,
        metrics=metrics,
        evaluation_hash=evaluation_hash,
        summary=summary,
        sources=tuple(sources),
        entry_id=f"knowledge_{entry_hash[:24]}",
        entry_hash=entry_hash,
    )


def _terminal_attempts(
    attempts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    selected = [
        dict(item)
        for item in attempts
        if item.get("status") in {"passed", "failed"}
    ]
    selected.sort(
        key=lambda item: knowledge_iteration(item.get("iteration"))
    )
    iterations = [item["iteration"] for item in selected]
    if len(iterations) != len(set(iterations)):
        raise SandboxKnowledgeError("沙箱尝试轮次不能重复")
    return tuple(selected)


def _read_run_attempts(root: Path) -> tuple[dict[str, Any], ...]:
    state_path = root / "run-state.json"
    manifest_path = root / "manifest.json"
    try:
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, Mapping):
                raise SandboxKnowledgeError("run-state 必须是 JSON 对象")
            attempts = state.get("attempts")
        elif manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping):
                raise SandboxKnowledgeError("manifest 必须是 JSON 对象")
            attempts = manifest.get("attempts")
        else:
            raise SandboxKnowledgeError(
                "沙箱运行缺少 run-state 或 manifest"
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise SandboxKnowledgeError(
            f"沙箱运行状态不可用：{exc}"
        ) from exc
    if not isinstance(attempts, (list, tuple)):
        raise SandboxKnowledgeError("沙箱运行 attempts 必须是数组")
    return _terminal_attempts(attempts)


def build_sandbox_knowledge(
    artifact_dir: Path | str,
    request: SandboxResearchRequest,
    *,
    attempts: Sequence[Mapping[str, Any]] | None = None,
) -> SandboxKnowledgeBase:
    """只从当前运行目录构建确定性知识索引。"""

    root = Path(artifact_dir)
    if not root.is_dir():
        raise SandboxKnowledgeError("沙箱运行目录不存在")
    sandbox_run_id = knowledge_text(
        getattr(request, "sandbox_run_id", None),
        "sandbox_run_id",
        maximum=160,
    )
    request_hash = knowledge_hash(
        getattr(request, "request_hash", None),
        "request_hash",
    )
    task_hash = knowledge_hash(
        getattr(request, "task_hash", None),
        "task_hash",
    )
    if attempts is None:
        state_path = root / "run-state.json"
        if state_path.is_file():
            try:
                state = json.loads(
                    state_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise SandboxKnowledgeError(
                    f"沙箱运行状态不可用：{exc}"
                ) from exc
            if not isinstance(state, Mapping):
                raise SandboxKnowledgeError(
                    "run-state 必须是 JSON 对象"
                )
            if state.get("request_hash") != request_hash:
                raise SandboxKnowledgeError(
                    "沙箱运行状态绑定了不同请求"
                )
        attempts = _read_run_attempts(root)
    terminal = _terminal_attempts(attempts)

    entries: list[SandboxKnowledgeEntry] = []
    pending_failures: list[
        tuple[int, SandboxKnowledgeSource, Mapping[str, Any]]
    ] = []
    for attempt in terminal:
        iteration = knowledge_iteration(attempt.get("iteration"))
        source = _knowledge_source(
            root,
            sandbox_run_id=sandbox_run_id,
            attempt=attempt,
        )
        code_hash = knowledge_hash(
            attempt.get("code_hash"),
            "attempt.code_hash",
        )
        feedback_value = attempt.get("feedback")
        metrics_value = attempt.get("metrics")
        metrics = (
            knowledge_json_mapping(metrics_value, "attempt.metrics")
            if metrics_value is not None
            else None
        )
        if attempt.get("status") == "failed":
            if feedback_value is None:
                raise SandboxKnowledgeError("失败尝试缺少结构化反馈")
            feedback = knowledge_json_mapping(
                feedback_value,
                "attempt.feedback",
            )
            try:
                SandboxCodeFeedback.from_dict(feedback)
            except Exception as exc:
                raise SandboxKnowledgeError("失败尝试反馈非法") from exc
            entries.append(
                _knowledge_entry(
                    kind="failure",
                    task_hash=task_hash,
                    sandbox_run_id=sandbox_run_id,
                    iteration=iteration,
                    sources=(source,),
                    code_hash=code_hash,
                    feedback=feedback,
                    metrics=metrics,
                    evaluation_hash=None,
                )
            )
            pending_failures.append((iteration, source, feedback))
            continue

        evaluation_value = attempt.get("evaluation")
        if not isinstance(evaluation_value, Mapping):
            raise SandboxKnowledgeError("成功尝试缺少评估结果")
        evaluation_hash = knowledge_hash(
            evaluation_value.get("evaluation_hash"),
            "attempt.evaluation_hash",
        )
        if metrics is None:
            raise SandboxKnowledgeError("成功尝试缺少指标")
        entries.append(
            _knowledge_entry(
                kind="success",
                task_hash=task_hash,
                sandbox_run_id=sandbox_run_id,
                iteration=iteration,
                sources=(source,),
                code_hash=code_hash,
                feedback=None,
                metrics=metrics,
                evaluation_hash=evaluation_hash,
            )
        )
        if pending_failures:
            sources = tuple(
                [item[1] for item in pending_failures] + [source]
            )
            latest_feedback = pending_failures[-1][2]
            entries.append(
                _knowledge_entry(
                    kind="repair",
                    task_hash=task_hash,
                    sandbox_run_id=sandbox_run_id,
                    iteration=iteration,
                    sources=sources,
                    code_hash=code_hash,
                    feedback=latest_feedback,
                    metrics=metrics,
                    evaluation_hash=evaluation_hash,
                )
            )
            pending_failures.clear()

    material = {
        "schema_version": SANDBOX_KNOWLEDGE_SCHEMA_VERSION,
        "version": 1,
        "sandbox_run_id": sandbox_run_id,
        "request_hash": request_hash,
        "task_hash": task_hash,
        "entries": [item.to_dict() for item in entries],
    }
    return SandboxKnowledgeBase(
        sandbox_run_id=sandbox_run_id,
        request_hash=request_hash,
        task_hash=task_hash,
        entries=tuple(entries),
        index_hash=sha256_json(material),
    )


def write_sandbox_knowledge(
    knowledge: SandboxKnowledgeBase,
    artifact_dir: Path | str,
) -> Path:
    """原子写入当前运行知识索引。"""

    if not isinstance(knowledge, SandboxKnowledgeBase):
        raise SandboxKnowledgeError(
            "knowledge 必须是 SandboxKnowledgeBase"
        )
    root = Path(artifact_dir)
    if not root.is_dir():
        raise SandboxKnowledgeError("沙箱运行目录不存在")
    target = root / SANDBOX_KNOWLEDGE_FILE
    temporary = target.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            canonical_json(knowledge.to_dict()),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except OSError as exc:
        raise SandboxKnowledgeError(f"知识索引写入失败：{exc}") from exc
    return target


def _verify_sources(
    root: Path,
    knowledge: SandboxKnowledgeBase,
) -> None:
    terminal = {
        item["iteration"]: item
        for item in _read_run_attempts(root)
    }
    for entry in knowledge.entries:
        current = [
            item
            for item in entry.sources
            if item.iteration == entry.iteration
        ]
        if len(current) != 1 or current[0].attempt_hash != sha256_json(
            terminal.get(entry.iteration, {})
        ):
            raise SandboxKnowledgeError("知识来源绑定了不同尝试")
        for source in entry.sources:
            attempt = terminal.get(source.iteration)
            if attempt is None:
                raise SandboxKnowledgeError("知识来源轮次不存在")
            if source.sandbox_run_id != knowledge.sandbox_run_id:
                raise SandboxKnowledgeError("知识来源绑定了不同运行")
            if source.attempt_hash != sha256_json(attempt):
                raise SandboxKnowledgeError("知识来源尝试哈希变化")
            expected_paths = _attempt_artifact_paths(attempt)
            actual_paths = tuple(item.path for item in source.artifacts)
            if actual_paths != expected_paths:
                raise SandboxKnowledgeError("知识来源文件集合变化")
            if (
                source.iteration == entry.iteration
                and attempt.get("code_hash") != entry.code_hash
            ):
                raise SandboxKnowledgeError("知识条目代码哈希变化")
            for artifact in source.artifacts:
                current_artifact = _artifact_reference(
                    root,
                    artifact.path,
                )
                if current_artifact != artifact:
                    raise SandboxKnowledgeError("知识来源文件哈希变化")


def load_sandbox_knowledge(
    artifact_dir: Path | str,
) -> SandboxKnowledgeBase:
    """加载并校验当前运行的知识索引。"""

    root = Path(artifact_dir)
    if not root.is_dir():
        raise SandboxKnowledgeError("沙箱运行目录不存在")
    if (root / "manifest.json").is_file():
        try:
            verify_artifact_tree(root)
        except SandboxArtifactError as exc:
            raise SandboxKnowledgeError(
                f"沙箱产物校验失败：{exc}"
            ) from exc
    target = root / SANDBOX_KNOWLEDGE_FILE
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SandboxKnowledgeError("当前运行没有知识索引") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SandboxKnowledgeError(
            f"知识索引不可用：{exc}"
        ) from exc
    knowledge = SandboxKnowledgeBase.from_dict(payload)
    _verify_sources(root, knowledge)
    return knowledge


__all__ = [
    "SANDBOX_KNOWLEDGE_FILE",
    "SANDBOX_KNOWLEDGE_SCHEMA_VERSION",
    "SandboxKnowledgeArtifact",
    "SandboxKnowledgeBase",
    "SandboxKnowledgeEntry",
    "SandboxKnowledgeSelection",
    "SandboxKnowledgeSource",
    "build_sandbox_knowledge",
    "load_sandbox_knowledge",
    "write_sandbox_knowledge",
]
