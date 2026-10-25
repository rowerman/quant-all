"""Worker 侧 RD-Agent 子进程适配器。"""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from typing import Mapping

from mining_worker.protocol import (
    WorkerGenerateRequest,
    WorkerGenerationResponse,
    WorkerProtocolError,
)
from mining_worker.worker_config import load_worker_config

_SECRET_PATTERN = re.compile(r"(?i)(sk-[A-Za-z0-9_-]{8,}|bearer\s+\S+)")


class WorkerJobTimeout(RuntimeError):
    """生成子进程超时。"""


class WorkerJobCancelled(RuntimeError):
    """生成子进程被取消。"""


class WorkerGenerationFailed(RuntimeError):
    """生成子进程失败。"""


def _sanitize(message: str) -> str:
    result = _SECRET_PATTERN.sub("[REDACTED]", str(message))
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name, "")
        if value:
            result = result.replace(value, "[REDACTED]")
    return result[:500]


class SubprocessGenerator:
    """每个作业启动独立 Python 子进程，便于超时后强制终止。"""

    def __init__(self, *, start_timeout: float = 5.0) -> None:
        self.start_timeout = float(start_timeout)
        self._processes: dict[str, subprocess.Popen] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()

    def health(self) -> Mapping[str, object]:
        """检查固定版本、模型和 Key 是否配置；不调用外部 API。"""

        try:
            version = importlib.metadata.version("rdagent")
        except importlib.metadata.PackageNotFoundError:
            return {
                "status": "unavailable",
                "error_code": "MINING-119",
                "error_message": "RD-Agent 未安装",
            }
        config = load_worker_config()
        model = os.environ.get("CHAT_MODEL", config.default_model)
        configured = bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())
        status = "ready" if version == config.rdagent_version and configured else "unavailable"
        return {
            "status": status,
            "mode": "rdagent",
            "rdagent_version": version,
            "model": model,
            "llm_configured": configured,
        }

    def generate(
        self,
        request: WorkerGenerateRequest,
        *,
        timeout_seconds: float,
    ) -> WorkerGenerationResponse:
        """在独立进程组中生成一轮候选。"""

        with self._lock:
            if request.request_id in self._cancelled:
                raise WorkerJobCancelled("作业已取消")
            command = [sys.executable, "-m", "mining_worker.rdagent_generator"]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                start_new_session=True,
            )
            self._processes[request.request_id] = process
        try:
            stdout, _ = process.communicate(
                json.dumps(request.to_dict(), ensure_ascii=False),
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            self.cancel(request.request_id)
            raise WorkerJobTimeout("Worker 生成超时") from exc
        finally:
            with self._lock:
                self._processes.pop(request.request_id, None)
                cancelled = request.request_id in self._cancelled
        if cancelled:
            raise WorkerJobCancelled("作业已取消")
        try:
            payload = self._parse_output(stdout)
        except WorkerGenerationFailed:
            raise WorkerGenerationFailed("RD-Agent 子进程执行失败") from None
        if payload.get("ok") is not True:
            raise WorkerGenerationFailed(
                _sanitize(str(payload.get("error_message", "RD-Agent 生成失败")))
            )
        if process.returncode != 0:
            raise WorkerGenerationFailed("RD-Agent 子进程异常退出")
        raw = payload.get("response")
        try:
            return WorkerGenerationResponse.from_dict(raw)
        except WorkerProtocolError as exc:
            raise WorkerGenerationFailed(f"RD-Agent 返回非法候选：{exc}") from exc

    def cancel(self, request_id: str) -> None:
        """标记取消并终止对应子进程组。"""

        with self._lock:
            self._cancelled.add(request_id)
            process = self._processes.get(request_id)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=self.start_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    @staticmethod
    def _parse_output(stdout: str) -> dict:
        for line in reversed(stdout.splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise WorkerGenerationFailed("RD-Agent 子进程没有返回 JSON")


def worker_state_dir() -> Path:
    """返回 Worker 容器内的状态目录。"""

    return Path(os.environ.get("MINING_WORKER_STATE_DIR", "/state"))
