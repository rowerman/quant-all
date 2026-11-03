"""沙箱运行内知识索引、检索和来源校验测试。"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from mining.contract_types import canonical_json, sha256_json
from mining_sandbox.artifacts import (
    SandboxArtifactManifest,
    seal_artifact_tree,
)
from mining_sandbox.errors import SandboxKnowledgeError
from mining_sandbox.feedback import (
    SandboxCodeFeedback,
    SandboxFailureCategory,
)
from mining_sandbox.knowledge import (
    SandboxKnowledgeArtifact,
    build_sandbox_knowledge,
    load_sandbox_knowledge,
    write_sandbox_knowledge,
)
from mining_sandbox.knowledge_evidence import (
    build_worker_knowledge_evidence,
)
from mining_sandbox.research import SandboxResearchRequest


def _request(
    sandbox_run_id: str = "sandboxrun_knowledge_0001",
    *,
    max_iterations: int = 3,
) -> SandboxResearchRequest:
    return SandboxResearchRequest(
        sandbox_run_id=sandbox_run_id,
        hypothesis="短期价格修复在开发期可能形成可检验因子",
        context_hash="a" * 64,
        snapshot_manifest_hash="b" * 64,
        llm_model="deepseek/deepseek-chat",
        max_iterations=max_iterations,
    )


def _attempt(
    root: Path,
    iteration: int,
    *,
    status: str,
) -> dict[str, object]:
    iteration_dir = root / "iterations" / f"{iteration:03d}"
    iteration_dir.mkdir(parents=True)
    code = f"print('iteration {iteration}')\n"
    code_path = iteration_dir / "proposal.py"
    code_path.write_text(code, encoding="utf-8")
    stdout_path = iteration_dir / "stdout.log"
    stderr_path = iteration_dir / "stderr.log"
    stdout_path.write_text("ok\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    metrics = {
        "rank_ic": 0.01 * iteration,
        "icir": 0.1 * iteration,
        "coverage": 0.8,
        "valid_days": 3,
    }
    metrics_path = iteration_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    attempt: dict[str, object] = {
        "iteration": iteration,
        "proposal_id": f"proposal_{iteration}",
        "code_file": code_path.relative_to(root).as_posix(),
        "code_hash": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "rationale": f"第 {iteration} 轮",
        "model": "deepseek/deepseek-chat",
        "prompt_version": "mining_qlib_code_prompt_v2",
        "usage": {},
        "status": status,
        "metrics": metrics,
        "error_code": None,
        "feedback": None,
        "image_id": "sha256:" + "d" * 64,
        "container_name": f"test-{iteration}",
        "exit_code": 0 if status == "passed" else 2,
        "timed_out": False,
        "duration_ms": 1.0,
        "stdout_file": stdout_path.relative_to(root).as_posix(),
        "stderr_file": stderr_path.relative_to(root).as_posix(),
        "metrics_file": metrics_path.relative_to(root).as_posix(),
        "factor_file": None,
        "evaluation": None,
        "evaluation_file": None,
    }
    if status == "failed":
        feedback = SandboxCodeFeedback.create(
            SandboxFailureCategory.EXECUTION,
            summary=f"第 {iteration} 轮执行失败",
            repair_hint="修复运行时异常",
        )
        attempt["error_code"] = feedback.error_code
        attempt["feedback"] = feedback.to_dict()
    else:
        evaluation = {
            "schema_version": "mining_sandbox_evaluation_v1",
            "evaluation_hash": sha256_json(
                {"iteration": iteration, "metrics": metrics}
            ),
        }
        evaluation_path = iteration_dir / "evaluation.json"
        evaluation_path.write_text(
            json.dumps(evaluation),
            encoding="utf-8",
        )
        attempt["evaluation"] = evaluation
        attempt["evaluation_file"] = evaluation_path.relative_to(root).as_posix()
    return attempt


def _state(
    root: Path,
    request: SandboxResearchRequest,
    attempts: list[dict[str, object]],
) -> None:
    (root / "run-state.json").write_text(
        canonical_json(
            {
                "schema_version": "mining_sandbox_research_v1",
                "request_hash": request.request_hash,
                "status": "running",
                "attempts": attempts,
            }
        ),
        encoding="utf-8",
    )


def test_task_hash_is_stable_across_run_ids(tmp_path: Path):
    left = _request("sandboxrun_knowledge_0001")
    right = _request("sandboxrun_knowledge_0002")

    assert left.task_hash == right.task_hash
    assert left.request_hash != right.request_hash
    assert replace(left, hypothesis="另一个可检验的短期价格修复假设").task_hash != (
        left.task_hash
    )


def test_failure_trajectory_success_and_repair_are_stable(tmp_path: Path):
    request = _request()
    root = tmp_path / request.sandbox_run_id
    root.mkdir()
    attempts = [
        _attempt(root, 1, status="failed"),
        _attempt(root, 2, status="failed"),
        _attempt(root, 3, status="passed"),
    ]
    _state(root, request, attempts)

    first = build_sandbox_knowledge(root, request, attempts=attempts)
    second = build_sandbox_knowledge(root, request, attempts=attempts)
    write_sandbox_knowledge(first, root)
    loaded = load_sandbox_knowledge(root)
    selected = loaded.retrieve(request.task_hash)

    assert first.to_dict() == second.to_dict() == loaded.to_dict()
    assert [item.kind for item in loaded.entries] == [
        "failure",
        "failure",
        "success",
        "repair",
    ]
    assert [item.iteration for item in selected.failures] == [1, 2]
    assert [item.iteration for item in selected.successes] == [3]
    assert len(selected.repairs) == 1
    assert selected.repairs[0].source_iterations == (1, 2, 3)
    assert selected.repairs[0].feedback["error_code"] == "MINING-139"
    assert selected.repairs[0].summary.startswith("第 1,2 轮失败后")
    assert len({item.source_hash for item in selected.repairs[0].sources}) == 3


def test_worker_evidence_is_a_stable_whitelist(tmp_path: Path):
    request = _request()
    root = tmp_path / request.sandbox_run_id
    root.mkdir()
    attempts = [
        _attempt(root, 1, status="failed"),
        _attempt(root, 2, status="passed"),
    ]
    _state(root, request, attempts)
    knowledge = build_sandbox_knowledge(root, request, attempts=attempts)

    first = build_worker_knowledge_evidence(knowledge)
    second = build_worker_knowledge_evidence(knowledge)
    payload = first.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)

    assert first == second
    assert first.task_hash == knowledge.task_hash
    assert first.index_hash == knowledge.index_hash
    assert first.retrieval_hash == knowledge.retrieve(
        request.task_hash
    ).selection_hash
    assert [item.kind for item in first.entries] == [
        "failure",
        "success",
        "repair",
    ]
    assert "sources" not in encoded
    assert "sandbox_run_id" not in encoded
    assert str(root) not in encoded
    assert payload["entries"][0]["feedback"]["repair_hint"] == (
        "修复运行时异常"
    )


def test_pure_success_has_no_repair_entry(tmp_path: Path):
    request = _request(max_iterations=1)
    root = tmp_path / request.sandbox_run_id
    root.mkdir()
    attempts = [_attempt(root, 1, status="passed")]
    _state(root, request, attempts)

    knowledge = build_sandbox_knowledge(root, request, attempts=attempts)
    selected = knowledge.retrieve(request.task_hash)

    assert len(selected.successes) == 1
    assert selected.failures == ()
    assert selected.repairs == ()


def test_retrieval_rejects_another_task(tmp_path: Path):
    request = _request(max_iterations=1)
    root = tmp_path / request.sandbox_run_id
    root.mkdir()
    attempts = [_attempt(root, 1, status="failed")]
    knowledge = build_sandbox_knowledge(root, request, attempts=attempts)

    with pytest.raises(SandboxKnowledgeError, match="任务与当前运行不一致"):
        knowledge.retrieve("f" * 64)


def test_build_rejects_state_from_another_run(tmp_path: Path):
    request = _request("sandboxrun_knowledge_0001", max_iterations=1)
    other = _request("sandboxrun_knowledge_0002", max_iterations=1)
    root = tmp_path / request.sandbox_run_id
    root.mkdir()
    attempts = [_attempt(root, 1, status="passed")]
    _state(root, other, attempts)

    with pytest.raises(SandboxKnowledgeError, match="不同请求"):
        build_sandbox_knowledge(root, request)


def test_contract_rejects_unknown_fields_and_tampered_hashes(tmp_path: Path):
    request = _request(max_iterations=1)
    root = tmp_path / request.sandbox_run_id
    root.mkdir()
    attempts = [_attempt(root, 1, status="passed")]
    base = build_sandbox_knowledge(root, request, attempts=attempts)
    payload = base.to_dict()
    payload["unknown"] = True

    with pytest.raises(SandboxKnowledgeError, match="字段不匹配"):
        type(base).from_dict(payload)

    tampered = base.to_dict()
    tampered["entries"][0]["summary"] = "被篡改的摘要"
    with pytest.raises(SandboxKnowledgeError, match="摘要与条目内容不一致"):
        type(base).from_dict(tampered)


def test_source_paths_cannot_escape_the_current_run(tmp_path: Path):
    request = _request(max_iterations=1)
    root = tmp_path / request.sandbox_run_id
    root.mkdir()
    attempts = [_attempt(root, 1, status="passed")]
    base = build_sandbox_knowledge(root, request, attempts=attempts)
    payload = base.to_dict()
    payload["entries"][0]["sources"][0]["artifacts"][0]["path"] = (
        "../other-run/proposal.py"
    )

    with pytest.raises(SandboxKnowledgeError, match="规范相对路径"):
        type(base).from_dict(payload)

    outside = attempts[0] | {"code_file": "../other-run/proposal.py"}
    with pytest.raises(SandboxKnowledgeError, match="规范相对路径"):
        build_sandbox_knowledge(root, request, attempts=[outside])


def test_loader_detects_changed_source_artifact(tmp_path: Path):
    request = _request(max_iterations=1)
    root = tmp_path / request.sandbox_run_id
    root.mkdir()
    attempts = [_attempt(root, 1, status="passed")]
    _state(root, request, attempts)
    knowledge = build_sandbox_knowledge(root, request, attempts=attempts)
    write_sandbox_knowledge(knowledge, root)
    (root / attempts[0]["code_file"]).write_text(
        "print('changed')\n",
        encoding="utf-8",
    )

    with pytest.raises(SandboxKnowledgeError, match="尝试哈希变化|文件哈希变化"):
        load_sandbox_knowledge(root)


def test_legacy_sealed_run_can_be_derived_without_modification(tmp_path: Path):
    request = _request(max_iterations=1)
    root = tmp_path / request.sandbox_run_id
    root.mkdir()
    attempts = [_attempt(root, 1, status="passed")]
    _state(root, request, attempts)
    files = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest = SandboxArtifactManifest(
        sandbox_run_id=request.sandbox_run_id,
        request_hash=request.request_hash,
        context_hash=request.context_hash,
        snapshot_manifest_hash=request.snapshot_manifest_hash,
        hypothesis_hash=sha256_json(request.hypothesis),
        image_id="sha256:" + "d" * 64,
        status="completed",
        attempts=tuple(attempts),
        environment={},
        files=files,
    )
    seal_artifact_tree(root, manifest)

    knowledge = build_sandbox_knowledge(root, request)

    assert len(knowledge.successes(request.task_hash)) == 1
    assert not (root / "knowledge.json").exists()
    with pytest.raises(SandboxKnowledgeError, match="没有知识索引"):
        load_sandbox_knowledge(root)


def test_artifact_contract_requires_sha256():
    with pytest.raises(SandboxKnowledgeError, match="SHA-256"):
        SandboxKnowledgeArtifact(path="iterations/001/a.py", bytes=1, sha256="bad")
