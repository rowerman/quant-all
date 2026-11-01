"""Worker v5 研报抽取子进程适配器。"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from typing import Mapping

from mining_worker.generator import (
    WorkerGenerationFailed,
    WorkerJobCancelled,
    WorkerJobTimeout,
    _sanitize,
)
from mining_worker.protocol import WorkerProtocolError
from mining_worker.protocol_research import (
    WorkerResearchExtractionRequest,
    WorkerResearchExtractionResponse,
)


class SubprocessResearchGenerator:
    """为每个研报抽取窗口启动独立子进程。"""

    def __init__(self, *, start_timeout: float = 5.0) -> None:
        self.start_timeout = float(start_timeout)
        self._processes: dict[str, subprocess.Popen] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()

    def health(self) -> Mapping[str, object]:
        """复用 Worker 固定模型和 Key 的健康检查。"""

        from mining_worker.generator import SubprocessGenerator

        return SubprocessGenerator().health()

    def generate(
        self,
        request: WorkerResearchExtractionRequest,
        *,
        timeout_seconds: float,
    ) -> WorkerResearchExtractionResponse:
        """执行一个研报抽取请求。"""

        with self._lock:
            if request.request_id in self._cancelled:
                raise WorkerJobCancelled("作业已取消")
            process = subprocess.Popen(
                [sys.executable, "-m", "mining_worker.rdagent_research_generator"],
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
            raise WorkerJobTimeout("Worker v5 研报抽取超时") from exc
        finally:
            with self._lock:
                self._processes.pop(request.request_id, None)
                cancelled = request.request_id in self._cancelled
        if cancelled:
            raise WorkerJobCancelled("作业已取消")
        payload = self._parse_output(stdout)
        if payload.get("ok") is not True:
            raise WorkerGenerationFailed(
                _sanitize(str(payload.get("error_message", "研报抽取失败")))
            )
        if process.returncode != 0:
            raise WorkerGenerationFailed("研报抽取子进程异常退出")
        try:
            return WorkerResearchExtractionResponse.from_dict(
                payload.get("response")
            )
        except WorkerProtocolError as exc:
            raise WorkerGenerationFailed(f"研报抽取响应非法：{exc}") from exc

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
        raise WorkerGenerationFailed("研报抽取子进程没有返回 JSON")


__all__ = ["SubprocessResearchGenerator"]
