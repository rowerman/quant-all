"""Worker 固定配置读取。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


class WorkerConfigError(ValueError):
    """Worker 配置缺失或不合法。"""


def _default_config_path() -> Path:
    override = os.environ.get("MINING_WORKER_CONFIG_FILE", "").strip()
    if override:
        return Path(override)
    source_path = Path(__file__).resolve().parents[2] / "configs" / "mining" / "worker.json"
    if source_path.exists():
        return source_path
    return Path("/app/configs/mining/worker.json")


@dataclass(frozen=True)
class WorkerConfig:
    """Worker 版本、模型和超时默认值。"""

    schema_version: str
    rdagent_version: str
    python_version: str
    default_model: str
    job_timeout_seconds: float
    client_timeout_seconds: float
    poll_interval_seconds: float
    max_candidates_per_round: int

    def __post_init__(self) -> None:
        if self.schema_version != "mining_worker_v1":
            raise WorkerConfigError("Worker 配置 schema_version 不匹配")
        if self.rdagent_version != "1.0.0" or self.python_version != "3.11":
            raise WorkerConfigError("Worker 版本必须固定为 RD-Agent 1.0.0 / Python 3.11")
        if not self.default_model.strip():
            raise WorkerConfigError("default_model 不能为空")
        if not 0 < self.job_timeout_seconds <= self.client_timeout_seconds:
            raise WorkerConfigError("Worker 超时必须大于 0 且不大于主服务超时")
        if self.poll_interval_seconds <= 0:
            raise WorkerConfigError("poll_interval_seconds 必须大于 0")
        if not 1 <= self.max_candidates_per_round <= 8:
            raise WorkerConfigError("max_candidates_per_round 必须在 1～8")


def load_worker_config(path: Path | str | None = None) -> WorkerConfig:
    """读取并校验 Worker 固定配置。"""

    target = Path(path) if path is not None else _default_config_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkerConfigError(f"Worker 配置不存在或不可读：{target}") from exc
    except json.JSONDecodeError as exc:
        raise WorkerConfigError(f"Worker 配置 JSON 非法：{exc}") from exc
    if not isinstance(payload, dict):
        raise WorkerConfigError("Worker 配置根节点必须是对象")
    expected = {
        "schema_version",
        "rdagent_version",
        "python_version",
        "default_model",
        "job_timeout_seconds",
        "client_timeout_seconds",
        "poll_interval_seconds",
        "max_candidates_per_round",
    }
    if set(payload) != expected:
        raise WorkerConfigError("Worker 配置字段不完整或包含未知字段")
    return WorkerConfig(**payload)
