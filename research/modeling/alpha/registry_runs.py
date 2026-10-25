"""扫描模型关联的评估、账本、监控、生命周期和最终留出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from alpha.artifacts import file_sha256
from alpha.errors import AlphaArtifactError
from alpha.holdout import load_holdout_windows

DOWNSTREAM_KINDS = (
    "evaluations",
    "acceptances",
    "monitoring",
    "lifecycles",
    "final_holdouts",
)


def _read_manifest(path: Path, label: str) -> dict:
    """读取运行 manifest，损坏时整次索引重建失败。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AlphaArtifactError(f"无法读取{label}：{path}") from error
    if not isinstance(payload, dict):
        raise AlphaArtifactError(f"{label}必须是 JSON 对象：{path}")
    return payload


def _manifest_link(path: Path) -> dict:
    """返回所有运行链接共用的 manifest 身份字段。"""

    return {"manifest_sha256": file_sha256(path)}


def _run_id(manifest: Mapping, folder: Path) -> str:
    """读取运行 ID。"""

    return str(
        manifest.get("analysis_id")
        or manifest.get("holdout_id")
        or manifest.get("monitor_id")
        or manifest.get("lifecycle_id")
        or folder.name
    )


def _model_candidates(manifest: dict) -> dict[str, list[str]]:
    """从评估计划提取每个模型关联的候选名。"""

    result: dict[str, list[str]] = {}
    plan = manifest.get("plan") or {}
    for candidate in (
        *(plan.get("candidates") or ()),
        *(plan.get("controls") or ()),
    ):
        source = candidate.get("source") or {}
        kind = source.get("kind")
        name = str(candidate.get("name") or "")
        if not name:
            continue
        model_ids = (
            [str(source.get("id") or "")]
            if kind == "model"
            else [str(item) for item in source.get("members") or ()]
            if kind == "blend"
            else []
        )
        for model_id in model_ids:
            if model_id:
                result.setdefault(model_id, []).append(name)
    return result


def _scan_evaluations(root: Path) -> dict[str, list[dict]]:
    """扫描评估计划中的模型候选关系。"""

    result: dict[str, list[dict]] = {}
    folder = root / "runs" / "evaluation"
    if not folder.is_dir():
        return result
    for path in sorted(folder.iterdir()):
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            continue
        manifest = _read_manifest(manifest_path, "评估 manifest")
        run_id = _run_id(manifest, path)
        for model_id, candidates in _model_candidates(manifest).items():
            result.setdefault(model_id, []).append(
                {
                    "kind": "evaluation",
                    "id": run_id,
                    "status": str(manifest.get("status") or "unknown"),
                    "start": manifest.get("start"),
                    "end": manifest.get("end"),
                    "selected_candidate": manifest.get("selected_candidate"),
                    "candidates": sorted(set(candidates)),
                    "path": path.relative_to(root).as_posix(),
                    **_manifest_link(manifest_path),
                }
            )
    return result


def _acceptance_source(item: dict) -> tuple[str, str] | None:
    """兼容通用账本和旧 P4 账本的候选来源字段。"""

    source = item.get("source")
    if isinstance(source, dict):
        kind = str(source.get("kind") or "")
        model_id = str(source.get("id") or "")
        return (kind, model_id) if kind == "model" and model_id else None
    legacy = (item.get("spec") or {}).get("source")
    if isinstance(legacy, str) and legacy:
        return "model", legacy
    return None


def _scan_acceptances(root: Path) -> dict[str, list[dict]]:
    """扫描通用账本和旧 P4 账本。"""

    result: dict[str, list[dict]] = {}
    folder = root / "runs" / "acceptance"
    if not folder.is_dir():
        return result
    for path in sorted(folder.iterdir()):
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            continue
        manifest = _read_manifest(manifest_path, "完整账本 manifest")
        run_id = _run_id(manifest, path)
        scope = manifest.get("scope") or {}
        for candidate, item in (manifest.get("candidates") or {}).items():
            source = _acceptance_source(item)
            if source is None:
                continue
            model_id = source[1]
            result.setdefault(model_id, []).append(
                {
                    "kind": "acceptance",
                    "id": run_id,
                    "status": str(manifest.get("status") or "unknown"),
                    "evaluation_id": manifest.get("evaluation_id"),
                    "candidate": str(candidate),
                    "complete": bool(item.get("complete")),
                    "passed": bool(item.get("passed")),
                    "start": scope.get("start"),
                    "end": scope.get("end"),
                    "path": path.relative_to(root).as_posix(),
                    **_manifest_link(manifest_path),
                }
            )
    return result


def _scan_model_artifacts(
    root: Path,
    model_folders: Mapping[str, Path],
) -> dict[str, dict[str, list[dict]]]:
    """扫描模型目录内的监控和生命周期包。"""

    result: dict[str, dict[str, list[dict]]] = {}
    for model_id, model_folder in model_folders.items():
        payload = {"monitoring": [], "lifecycles": []}
        monitor_root = model_folder / "monitoring"
        if monitor_root.is_dir():
            for path in sorted(monitor_root.iterdir()):
                manifest_path = path / "manifest.json"
                if not path.is_dir() or not manifest_path.is_file():
                    continue
                manifest = _read_manifest(manifest_path, "监控 manifest")
                summary = manifest.get("summary") or {}
                payload["monitoring"].append(
                    {
                        "kind": "monitoring",
                        "id": _run_id(manifest, path),
                        "status": str(summary.get("status") or "unknown"),
                        "evaluation_id": manifest.get("evaluation_id"),
                        "candidate": manifest.get("candidate"),
                        "start": summary.get("start"),
                        "end": summary.get("end"),
                        "path": path.relative_to(root).as_posix(),
                        **_manifest_link(manifest_path),
                    }
                )
        lifecycle_root = model_folder / "lifecycle"
        if lifecycle_root.is_dir():
            for path in sorted(lifecycle_root.iterdir()):
                manifest_path = path / "manifest.json"
                complete_path = path / "complete.json"
                if not path.is_dir() or not manifest_path.is_file():
                    continue
                manifest = _read_manifest(manifest_path, "生命周期 manifest")
                summary = manifest.get("summary") or {}
                complete = (
                    _read_manifest(complete_path, "生命周期完成标记")
                    if complete_path.is_file()
                    else {}
                )
                payload["lifecycles"].append(
                    {
                        "kind": "lifecycle",
                        "id": _run_id(manifest, path),
                        "status": "complete"
                        if complete.get("complete") is True
                        else "incomplete",
                        "state": summary.get("latest_state"),
                        "monitor_id": (manifest.get("sources") or {}).get(
                            "monitor_id"
                        ),
                        "start": summary.get("start"),
                        "end": summary.get("end"),
                        "path": path.relative_to(root).as_posix(),
                        **_manifest_link(manifest_path),
                    }
                )
        result[model_id] = payload
    return result


def _scan_final_holdouts(root: Path) -> dict[str, list[dict]]:
    """扫描最终留出 manifest 中承担角色的模型。"""

    result: dict[str, list[dict]] = {}
    folder = root / "runs" / "final_holdout"
    if not folder.is_dir():
        return result
    windows = load_holdout_windows()
    for path in sorted(folder.iterdir()):
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            continue
        manifest = _read_manifest(manifest_path, "最终留出 manifest")
        scope = manifest.get("scope") or {}
        start = scope.get("data_start") or scope.get("start")
        end = scope.get("data_end") or scope.get("end")
        window_id = next(
            (
                window.window_id
                for window in windows
                if window.start == start and window.end == end
            ),
            None,
        )
        roles: dict[str, list[str]] = {}
        for role, source in (manifest.get("sources") or {}).items():
            if not isinstance(source, dict) or not source.get("model_id"):
                continue
            roles.setdefault(str(source["model_id"]), []).append(str(role))
        for model_id, role_names in roles.items():
            result.setdefault(model_id, []).append(
                {
                    "kind": "final_holdout",
                    "id": _run_id(manifest, path),
                    "status": str(manifest.get("status") or "unknown"),
                    "window_id": window_id,
                    "roles": sorted(set(role_names)),
                    "start": start,
                    "end": end,
                    "path": path.relative_to(root).as_posix(),
                    **_manifest_link(manifest_path),
                }
            )
    return result


def _empty_downstream() -> dict[str, list[dict]]:
    """返回固定结构，保证无运行链接的模型也能稳定读取。"""

    return {name: [] for name in DOWNSTREAM_KINDS}


def scan_downstream_runs(
    root: Path,
    model_folders: Mapping[str, Path],
) -> tuple[dict[str, dict], int]:
    """扫描全部下游运行并按模型 ID 汇总。"""

    downstream = {
        model_id: _empty_downstream() for model_id in model_folders
    }
    append_keys = {
        "evaluations": _scan_evaluations(root),
        "acceptances": _scan_acceptances(root),
        "final_holdouts": _scan_final_holdouts(root),
    }
    for name, records in append_keys.items():
        for model_id, links in records.items():
            if model_id in downstream:
                downstream[model_id][name].extend(links)
    local = _scan_model_artifacts(root, model_folders)
    for model_id, records in local.items():
        downstream[model_id]["monitoring"].extend(records["monitoring"])
        downstream[model_id]["lifecycles"].extend(records["lifecycles"])
    for records in downstream.values():
        for name in DOWNSTREAM_KINDS:
            records[name].sort(
                key=lambda item: (
                    str(item.get("id") or ""),
                    str(item.get("candidate") or ""),
                )
            )
    link_count = sum(
        len(records)
        for payload in downstream.values()
        for records in payload.values()
    )
    return downstream, link_count


__all__ = ["DOWNSTREAM_KINDS", "scan_downstream_runs"]
