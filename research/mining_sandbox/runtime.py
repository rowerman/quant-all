"""使用 Docker 启动隔离、无网络、资源受限的代码研究容器。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable, Sequence

from mining.contract_types import identifier
from mining_sandbox.errors import (
    SandboxIsolationError,
    SandboxPolicyError,
    SandboxRunnerError,
)
from mining_sandbox.policy import SandboxPolicy


@dataclass(frozen=True)
class SandboxExecutionResult:
    """一次隔离容器执行的可审计结果。"""

    run_id: str
    container_name: str
    image_id: str
    exit_code: int
    timed_out: bool
    duration_ms: float
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool

    def to_dict(self) -> dict[str, object]:
        """返回脱敏审计字典。"""

        return {
            "run_id": self.run_id,
            "container_name": self.container_name,
            "image_id": self.image_id,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
        }


def _within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SandboxIsolationError(f"沙箱路径越界：{path}") from exc
    return resolved


def _truncated_text(data: bytes, maximum_bytes: int) -> tuple[str, bool]:
    encoded = data[:maximum_bytes]
    return encoded.decode("utf-8", errors="replace"), len(data) > maximum_bytes


class DockerSandboxRunner:
    """只允许通过 Python 命令在隔离容器内执行研究代码。"""

    def __init__(
        self,
        policy: SandboxPolicy,
        *,
        docker_binary: str = "docker",
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        if not policy.enabled:
            raise SandboxPolicyError("沙箱政策未启用")
        self.policy = policy
        self.docker_binary = docker_binary
        self._popen = popen
        self._run = run

    def build_command(
        self,
        *,
        run_id: str,
        snapshot_dir: Path | str,
        workspace_dir: Path | str,
        command: Sequence[str],
    ) -> tuple[list[str], str, Path, Path]:
        """构造严格 Docker 参数并校验两个挂载目录。"""

        run_id = identifier(run_id, "run_id")
        snapshot = _within(
            Path(snapshot_dir),
            self.policy.snapshot_root_path,
        )
        workspace = _within(
            Path(workspace_dir),
            self.policy.workspace_root_path,
        )
        if not snapshot.is_dir() or not (snapshot / "manifest.json").is_file():
            raise SandboxIsolationError("沙箱快照目录或清单不存在")
        workspace.mkdir(parents=True, exist_ok=True)
        os.chmod(workspace, 0o777)
        if (
            isinstance(command, (str, bytes))
            or not command
            or any(not isinstance(item, str) or "\x00" in item for item in command)
        ):
            raise SandboxRunnerError("容器命令必须是非空字符串数组")
        if command[0] not in {"python", "python3"}:
            raise SandboxRunnerError("沙箱只接受 Python 解释器入口")
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:24]
        container_name = f"mining-sandbox-{digest}"
        args = [
            self.docker_binary,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--user={self.policy.user}",
            f"--pids-limit={self.policy.pids_limit}",
            f"--memory={self.policy.memory_mb}m",
            f"--cpus={self.policy.cpu_limit:g}",
            "--ulimit",
            "nofile=256:256",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--workdir",
            "/workspace",
            "--mount",
            f"type=bind,src={snapshot},dst=/snapshot,readonly",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace",
            self.policy.image,
            *command,
        ]
        return args, container_name, snapshot, workspace

    def _image_id(self) -> str:
        try:
            completed = self._run(
                [
                    self.docker_binary,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    self.policy.image,
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SandboxRunnerError(f"无法读取沙箱镜像：{exc}") from exc
        if completed.returncode != 0:
            raise SandboxRunnerError("沙箱镜像不存在或不可读取")
        image_id = str(completed.stdout).strip()
        if not image_id.startswith("sha256:") or len(image_id) != 71:
            raise SandboxRunnerError("沙箱镜像 ID 不合法")
        return image_id

    def resolve_image_id(self) -> str:
        """返回当前沙箱镜像的不可变 Docker ID。"""

        return self._image_id()

    def run(
        self,
        *,
        run_id: str,
        snapshot_dir: Path | str,
        workspace_dir: Path | str,
        command: Sequence[str],
    ) -> SandboxExecutionResult:
        """执行容器；超时后强制终止并返回显式超时状态。"""

        args, container_name, _, _ = self.build_command(
            run_id=run_id,
            snapshot_dir=snapshot_dir,
            workspace_dir=workspace_dir,
            command=command,
        )
        image_id = self._image_id()
        maximum_output = self.policy.output_mb * 1024 * 1024
        started = time.monotonic()
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            try:
                process = self._popen(
                    args,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                try:
                    exit_code = int(
                        process.wait(timeout=self.policy.timeout_seconds)
                    )
                    timed_out = False
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._terminate_container(container_name)
                    try:
                        exit_code = int(process.wait(timeout=5))
                    except subprocess.TimeoutExpired:
                        process.kill()
                        exit_code = -9
            except (OSError, subprocess.SubprocessError) as exc:
                raise SandboxRunnerError(f"沙箱容器执行失败：{exc}") from exc
            duration_ms = (time.monotonic() - started) * 1000
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout, stdout_truncated = _truncated_text(
                stdout_file.read(maximum_output + 1),
                maximum_output,
            )
            stderr, stderr_truncated = _truncated_text(
                stderr_file.read(maximum_output + 1),
                maximum_output,
            )
        return SandboxExecutionResult(
            run_id=run_id,
            container_name=container_name,
            image_id=image_id,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def _terminate_container(self, container_name: str) -> None:
        try:
            self._run(
                [self.docker_binary, "kill", container_name],
                capture_output=True,
                timeout=10,
                check=False,
            )
            self._run(
                [self.docker_binary, "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return


__all__ = ["DockerSandboxRunner", "SandboxExecutionResult"]
