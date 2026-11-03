"""沙箱政策、Docker 隔离边界和 Qlib 冻结快照测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import duckdb
import numpy as np
import pytest

from mining.search_context import FrozenSearchContext
import mining_sandbox.policy as policy_module
from mining_sandbox.errors import (
    SandboxIsolationError,
    SandboxPolicyError,
    SandboxSnapshotError,
)
from mining_sandbox.policy import load_sandbox_policy
from mining_sandbox.qlib_snapshot import build_qlib_snapshot
from mining_sandbox.runtime import DockerSandboxRunner


def _database(path: Path) -> None:
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE SCHEMA md")
        conn.execute(
            "CREATE TABLE md.calendar(trade_date DATE, is_open INTEGER)"
        )
        conn.execute(
            "INSERT INTO md.calendar VALUES "
            "('2019-12-27',1),('2019-12-30',1),('2019-12-31',1),"
            "('2020-01-02',1)"
        )
        conn.execute(
            """
            CREATE TABLE md.daily_hfq(
                stock_code VARCHAR,
                trade_date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                vwap DOUBLE,
                volume_share DOUBLE,
                amount_cny DOUBLE,
                ret DOUBLE,
                turnover_rate DOUBLE,
                circ_mv DOUBLE,
                total_mv DOUBLE
            )
            """
        )
        conn.execute(
            "INSERT INTO md.daily_hfq VALUES "
            "('SH600000','2019-12-27',1,2,0.5,1.5,1.4,100,1000,0.1,1,2,3),"
            "('SH600000','2019-12-30',1.5,2.5,1,2,2.1,110,1100,0.2,1,2,3),"
            "('SH600000','2019-12-31',2,3,1.5,2.5,2.4,120,1200,0.25,1,2,3),"
            "('SZ000001','2019-12-30',3,4,2,3.5,3.3,200,2200,0.1,1,4,5),"
            "('SZ000001','2019-12-31',3.5,4.5,2.5,4,3.8,210,2300,0.14,1,4,5)"
        )
        conn.execute(
            "CREATE TABLE md.fwd_return("
            "stock_code VARCHAR, trade_date DATE, fwd_ret_open_5 DOUBLE)"
        )
        conn.execute(
            "INSERT INTO md.fwd_return VALUES "
            "('SH600000','2019-12-27',0.01),"
            "('SH600000','2019-12-30',0.02),"
            "('SH600000','2019-12-31',0.03),"
            "('SZ000001','2019-12-30',-0.01),"
            "('SZ000001','2019-12-31',-0.02)"
        )


def _context() -> FrozenSearchContext:
    return FrozenSearchContext(data_fingerprint="a" * 64)


def test_policy_is_restrictive_and_default_disabled():
    policy = load_sandbox_policy()
    assert policy.enabled is False
    assert policy.network is False
    assert policy.read_only_root is True
    assert policy.user != "0:0"
    with pytest.raises(SandboxPolicyError):
        replace(policy, network=True)
    with pytest.raises(SandboxPolicyError):
        replace(policy, read_only_root=False)
    with pytest.raises(SandboxPolicyError):
        replace(policy, user="0:0")
    with pytest.raises(SandboxPolicyError):
        replace(policy, snapshot_root="../outside")


def test_qlib_snapshot_is_reproducible_read_only_and_development_only(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    database = tmp_path / "quant.duckdb"
    _database(database)
    output = tmp_path / "snapshots" / "dev"

    manifest, reused = build_qlib_snapshot(
        context=_context(),
        db_file=database,
        output_dir=output,
        minimum_free_bytes=1,
    )

    assert reused is False
    assert manifest.start_date == "2012-01-01"
    assert manifest.end_date == "2019-12-31"
    assert manifest.row_count == 5
    assert set(manifest.fields) == {
        "amount_cny",
        "circ_mv",
        "close",
        "high",
        "label",
        "low",
        "open",
        "ret",
        "total_mv",
        "turnover_rate",
        "volume_share",
        "vwap",
    }
    calendar = (output / "calendars" / "day.txt").read_text().splitlines()
    instruments = (output / "instruments" / "all.txt").read_text().splitlines()
    assert calendar == ["2019-12-27", "2019-12-30", "2019-12-31"]
    assert len(instruments) == 2
    assert "2020-01-02" not in calendar
    assert (output / "manifest.json").stat().st_mode & 0o222 == 0

    raw = np.fromfile(
        output / "features" / "sh600000" / "close.day.bin",
        dtype="<f4",
    )
    assert float(raw[0]) == 0
    assert raw[1:].tolist() == [1.5, 2.0, 2.5]
    from qlib import init as qlib_init
    from qlib.data import D

    qlib_init(provider_uri=str(output), region="cn", kernels=1)
    features = D.features(
        ["SH600000"],
        ["$close", "$label"],
        start_time="2019-12-27",
        end_time="2019-12-31",
    )
    assert features["$close"].tolist() == [1.5, 2.0, 2.5]
    assert features["$label"].tolist() == pytest.approx([0.01, 0.02, 0.03])

    second, second_reused = build_qlib_snapshot(
        context=_context(),
        db_file=database,
        output_dir=output,
        minimum_free_bytes=1,
    )
    assert second_reused is True
    assert second.manifest_hash == manifest.manifest_hash

    changed = FrozenSearchContext(data_fingerprint="b" * 64)
    with pytest.raises(SandboxSnapshotError):
        build_qlib_snapshot(
            context=changed,
            db_file=database,
            output_dir=output,
            minimum_free_bytes=1,
        )


def test_qlib_snapshot_refuses_insufficient_capacity(tmp_path: Path):
    database = tmp_path / "quant.duckdb"
    _database(database)
    with pytest.raises(SandboxSnapshotError):
        build_qlib_snapshot(
            context=_context(),
            db_file=database,
            output_dir=tmp_path / "snapshot",
            minimum_free_bytes=10**18,
        )


def test_runner_rejects_path_escape_and_non_python_entry(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    policy = replace(
        load_sandbox_policy(),
        enabled=True,
        timeout_seconds=1,
        snapshot_root="snapshots",
        workspace_root="work",
    )
    snapshot = tmp_path / "snapshots" / "dev"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    runner = DockerSandboxRunner(policy)

    command, _, _, _ = runner.build_command(
        run_id="sandboxrun_test_0001",
        snapshot_dir=snapshot,
        workspace_dir=tmp_path / "work" / "run",
        command=("python", "-c", "print('ok')"),
    )
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert not any("docker.sock" in item for item in command)
    assert not any("/data/factors" in item for item in command)

    with pytest.raises(SandboxIsolationError):
        runner.build_command(
            run_id="sandboxrun_test_0002",
            snapshot_dir=tmp_path / "outside",
            workspace_dir=tmp_path / "work" / "run2",
            command=("python", "-c", "print('bad')"),
        )
    with pytest.raises(Exception):
        runner.build_command(
            run_id="sandboxrun_test_0003",
            snapshot_dir=snapshot,
            workspace_dir=tmp_path / "work" / "run3",
            command=("sh", "-c", "id"),
        )


def test_runner_timeout_kills_container(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    policy = replace(
        load_sandbox_policy(),
        enabled=True,
        timeout_seconds=1,
        snapshot_root="snapshots",
        workspace_root="work",
    )
    snapshot = tmp_path / "snapshots" / "dev"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")

    class FakeProcess:
        def __init__(self):
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            raise subprocess.TimeoutExpired("docker", timeout)

        def kill(self):
            return None

    calls = []

    class Completed:
        returncode = 0
        stdout = "sha256:" + "a" * 64

    runner = DockerSandboxRunner(
        policy,
        popen=lambda *args, **kwargs: FakeProcess(),
        run=lambda args, **kwargs: (
            calls.append(args) or Completed()
        ),
    )
    result = runner.run(
        run_id="sandboxrun_timeout_0001",
        snapshot_dir=snapshot,
        workspace_dir=tmp_path / "work" / "timeout",
        command=("python", "-c", "while True: pass"),
    )

    assert result.timed_out is True
    assert result.exit_code == -9
    assert any(command[1] == "kill" for command in calls)
    assert any(command[1] == "rm" for command in calls)


@pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="Docker 不可用",
)
def test_real_container_is_offline_and_read_only(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    policy = replace(
        load_sandbox_policy(),
        enabled=True,
        image="python:3.11-slim",
        timeout_seconds=20,
        snapshot_root="snapshots",
        workspace_root="work",
    )
    snapshot = tmp_path / "snapshots" / "dev"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    runner = DockerSandboxRunner(policy)
    script = (
        "import pathlib,socket,subprocess\n"
        "try:\n"
        " socket.create_connection(('1.1.1.1',53),timeout=1)\n"
        " offline=False\n"
        "except OSError:\n"
        " offline=True\n"
        "try:\n"
        " pathlib.Path('/readonly_probe').write_text('x')\n"
        " readonly=False\n"
        "except OSError:\n"
        " readonly=True\n"
        "child=subprocess.run(['python','-c',"
        "\"open('/child_probe','w').write('x')\"])\n"
        "child_blocked=child.returncode != 0\n"
        "workdir=pathlib.Path.cwd().as_posix() == '/workspace'\n"
        "snapshot=pathlib.Path('/snapshot/manifest.json').is_file()\n"
        "print(offline, readonly, child_blocked, workdir, snapshot)\n"
        "raise SystemExit(0 if offline and readonly and child_blocked "
        "and workdir and snapshot else 2)\n"
    )
    result = runner.run(
        run_id="sandboxrun_real_0001",
        snapshot_dir=snapshot,
        workspace_dir=tmp_path / "work" / "real",
        command=("python", "-c", script),
    )

    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "True True True True True"
    assert result.image_id.startswith("sha256:")


@pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="Docker 不可用",
)
def test_real_container_memory_limit_stops_allocation(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(policy_module, "PROJECT_ROOT", tmp_path)
    policy = replace(
        load_sandbox_policy(),
        enabled=True,
        image="python:3.11-slim",
        memory_mb=128,
        timeout_seconds=20,
        snapshot_root="snapshots",
        workspace_root="work",
    )
    snapshot = tmp_path / "snapshots" / "dev"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text("{}", encoding="utf-8")
    runner = DockerSandboxRunner(policy)

    result = runner.run(
        run_id="sandboxrun_oom_0001",
        snapshot_dir=snapshot,
        workspace_dir=tmp_path / "work" / "oom",
        command=(
            "python",
            "-c",
            "bytearray(256*1024*1024); print('unexpected')",
        ),
    )

    assert result.exit_code != 0
    assert "unexpected" not in result.stdout
