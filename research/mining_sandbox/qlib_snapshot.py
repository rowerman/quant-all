"""把冻结开发期行情导出为只读 Qlib 文件快照。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

import duckdb
import numpy as np

from mining.contract_types import canonical_json, sha256_json, utc_now_iso, validate_utc_timestamp
from mining.search_context import FrozenSearchContext
from mining_sandbox.errors import SandboxSnapshotError

QLIB_SNAPSHOT_SCHEMA_VERSION = "mining_qlib_snapshot_v1"
QLIB_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume_share",
    "amount_cny",
    "ret",
    "turnover_rate",
    "circ_mv",
    "total_mv",
    "label",
)
_STOCK_CODE = re.compile(r"^[A-Za-z0-9._-]+$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class QlibSnapshotManifest:
    """Qlib 快照的冻结输入、文件树和稳定哈希。"""

    context_hash: str
    data_version: str
    universe: str
    start_date: str
    end_date: str
    fields: tuple[str, ...]
    calendar_sha256: str
    instruments_sha256: str
    files: Mapping[str, Mapping[str, Any]]
    row_count: int
    total_bytes: int
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = QLIB_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != QLIB_SNAPSHOT_SCHEMA_VERSION:
            raise SandboxSnapshotError("Qlib 快照 schema_version 不匹配")
        for name in (
            "context_hash",
            "calendar_sha256",
            "instruments_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise SandboxSnapshotError(f"{name} 必须是 SHA-256")
        if self.row_count < 0 or self.total_bytes < 0:
            raise SandboxSnapshotError("快照行数和字节数必须非负")
        object.__setattr__(self, "fields", tuple(sorted(set(self.fields))))
        object.__setattr__(
            self,
            "files",
            {
                key: dict(value)
                for key, value in sorted(self.files.items())
            },
        )
        object.__setattr__(
            self,
            "created_at",
            validate_utc_timestamp(self.created_at, "created_at"),
        )

    def material(self) -> dict[str, Any]:
        """返回不包含创建时间的稳定清单内容。"""

        return {
            "schema_version": self.schema_version,
            "context_hash": self.context_hash,
            "data_version": self.data_version,
            "universe": self.universe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "fields": list(self.fields),
            "calendar_sha256": self.calendar_sha256,
            "instruments_sha256": self.instruments_sha256,
            "files": {key: dict(value) for key, value in self.files.items()},
            "row_count": self.row_count,
            "total_bytes": self.total_bytes,
        }

    @property
    def manifest_hash(self) -> str:
        """返回快照稳定内容哈希。"""

        return sha256_json(self.material())

    @classmethod
    def from_dict(cls, value: Any) -> QlibSnapshotManifest:
        """严格读取清单。"""

        if not isinstance(value, Mapping):
            raise SandboxSnapshotError("快照清单必须是 JSON 对象")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise SandboxSnapshotError("快照清单字段不完整或包含未知字段")
        payload = dict(value)
        payload["fields"] = tuple(payload["fields"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回持久化清单。"""

        return {**self.material(), "created_at": self.created_at}


def _write_feature_bin(
    path: Path,
    calendar: tuple[str, ...],
    values: Mapping[str, float],
) -> None:
    data = np.full(len(calendar), np.nan, dtype="<f4")
    positions = {day: index for index, day in enumerate(calendar)}
    for day, value in values.items():
        position = positions.get(day)
        if position is None:
            raise SandboxSnapshotError(f"特征日期不在冻结日历内：{day}")
        data[position] = np.float32(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.hstack([np.float32(0), data]).astype("<f").tofile(stream)


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def build_qlib_snapshot(
    *,
    context: FrozenSearchContext,
    db_file: Path | str,
    output_dir: Path | str,
    minimum_free_bytes: int = 256 * 1024 * 1024,
) -> tuple[QlibSnapshotManifest, bool]:
    """从开发期数据构建只读 Qlib 快照，返回清单和是否复用。"""

    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise SandboxSnapshotError("已有快照目录缺少清单")
        existing = QlibSnapshotManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        expected = {
            "context_hash": context.context_hash,
            "start_date": context.windows.development_start,
            "end_date": context.windows.development_end,
            "fields": tuple(sorted(QLIB_FIELDS)),
        }
        actual = {
            "context_hash": existing.context_hash,
            "start_date": existing.start_date,
            "end_date": existing.end_date,
            "fields": existing.fields,
        }
        if actual != expected:
            raise SandboxSnapshotError("已有快照与冻结上下文不一致")
        _verify_snapshot_files(target, existing)
        return existing, True

    source = Path(db_file)
    if not source.is_file():
        raise SandboxSnapshotError(f"源行情库不存在：{source}")
    try:
        conn = duckdb.connect(str(source), read_only=True)
    except duckdb.Error as exc:
        raise SandboxSnapshotError(f"无法只读打开行情库：{exc}") from exc
    try:
        calendar = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT trade_date FROM md.calendar "
                "WHERE is_open=1 AND trade_date BETWEEN ? AND ? "
                "ORDER BY trade_date",
                [
                    context.windows.development_start,
                    context.windows.development_end,
                ],
            ).fetchall()
        )
        if not calendar:
            raise SandboxSnapshotError("冻结开发期没有可用交易日")
        instruments = conn.execute(
            "SELECT stock_code, min(trade_date), max(trade_date) "
            "FROM md.daily_hfq WHERE trade_date BETWEEN ? AND ? "
            "GROUP BY stock_code ORDER BY stock_code",
            [
                context.windows.development_start,
                context.windows.development_end,
            ],
        ).fetchall()
        if not instruments:
            raise SandboxSnapshotError("冻结开发期没有可用股票")
        for stock_code, _, _ in instruments:
            if not _STOCK_CODE.fullmatch(str(stock_code)):
                raise SandboxSnapshotError(f"股票代码包含非法路径字符：{stock_code}")
        row_count = int(
            conn.execute(
                "SELECT count(*) FROM md.daily_hfq "
                "WHERE trade_date BETWEEN ? AND ?",
                [
                    context.windows.development_start,
                    context.windows.development_end,
                ],
            ).fetchone()[0]
        )
        estimate = max(
            minimum_free_bytes,
            int(row_count * len(QLIB_FIELDS) * 4 * 1.5),
        )
        free = int(shutil.disk_usage(target.parent).free)
        if free < estimate:
            raise SandboxSnapshotError(
                f"快照容量不足：预计至少 {estimate} 字节，可用 {free} 字节"
            )
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
        )
        try:
            manifest = _build_snapshot_tree(
                conn=conn,
                temporary=temporary,
                context=context,
                calendar=calendar,
                instruments=instruments,
                row_count=row_count,
            )
            if target.exists():
                raise SandboxSnapshotError("快照目标在构建期间被其他进程创建")
            _seal_tree(temporary)
            os.replace(temporary, target)
        except Exception:
            if temporary.exists():
                for path in sorted(temporary.rglob("*"), reverse=True):
                    os.chmod(path, 0o755 if path.is_dir() else 0o644)
                os.chmod(temporary, 0o755)
                shutil.rmtree(temporary)
            raise
    except duckdb.Error as exc:
        raise SandboxSnapshotError(f"读取冻结行情失败：{exc}") from exc
    finally:
        conn.close()
    _verify_snapshot_files(target, manifest)
    return manifest, False


def _build_snapshot_tree(
    *,
    conn,
    temporary: Path,
    context: FrozenSearchContext,
    calendar: tuple[str, ...],
    instruments: list[tuple],
    row_count: int,
) -> QlibSnapshotManifest:
    calendar_text = "\n".join(calendar) + "\n"
    calendar_path = temporary / "calendars" / "day.txt"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_path.write_text(calendar_text, encoding="utf-8")

    instruments_text = "".join(
        f"{stock_code}\t{start}\t{end}\n"
        for stock_code, start, end in instruments
    )
    instruments_path = temporary / "instruments" / "all.txt"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    instruments_path.write_text(instruments_text, encoding="utf-8")

    sql = """
        SELECT h.trade_date, h.open, h.high, h.low, h.close, h.vwap,
               h.volume_share, h.amount_cny, h.ret, h.turnover_rate,
               h.circ_mv, h.total_mv, f.fwd_ret_open_5 AS label
        FROM md.daily_hfq h
        LEFT JOIN md.fwd_return f
          ON f.stock_code=h.stock_code AND f.trade_date=h.trade_date
        WHERE h.stock_code=? AND h.trade_date BETWEEN ? AND ?
        ORDER BY h.trade_date
    """
    for stock_code, _, _ in instruments:
        rows = conn.execute(
            sql,
            [
                stock_code,
                context.windows.development_start,
                context.windows.development_end,
            ],
        ).fetchall()
        for field_index, field_name in enumerate(QLIB_FIELDS, start=1):
            values = {
                str(row[0]): float(row[field_index])
                for row in rows
                if row[field_index] is not None
            }
            _write_feature_bin(
                temporary
                / "features"
                / str(stock_code).lower()
                / f"{field_name}.day.bin",
                calendar,
                values,
            )

    files = {
        path.relative_to(temporary).as_posix(): {
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(temporary.rglob("*"))
        if path.is_file()
    }
    manifest = QlibSnapshotManifest(
        context_hash=context.context_hash,
        data_version=context.data_version,
        universe=context.universe,
        start_date=context.windows.development_start,
        end_date=context.windows.development_end,
        fields=QLIB_FIELDS,
        calendar_sha256=_file_sha256(calendar_path),
        instruments_sha256=_file_sha256(instruments_path),
        files=files,
        row_count=row_count,
        total_bytes=sum(int(item["bytes"]) for item in files.values()),
    )
    (temporary / "manifest.json").write_text(
        canonical_json(manifest.to_dict()),
        encoding="utf-8",
    )
    return manifest


def _verify_snapshot_files(
    root: Path,
    manifest: QlibSnapshotManifest,
) -> None:
    for relative, expected in manifest.files.items():
        path = root / relative
        if not path.is_file():
            raise SandboxSnapshotError(f"快照文件缺失：{relative}")
        if path.stat().st_size != int(expected["bytes"]):
            raise SandboxSnapshotError(f"快照文件大小变化：{relative}")
        if _file_sha256(path) != expected["sha256"]:
            raise SandboxSnapshotError(f"快照文件哈希变化：{relative}")
    if _file_sha256(root / "calendars" / "day.txt") != manifest.calendar_sha256:
        raise SandboxSnapshotError("快照日历哈希变化")
    if (
        _file_sha256(root / "instruments" / "all.txt")
        != manifest.instruments_sha256
    ):
        raise SandboxSnapshotError("快照股票池哈希变化")


__all__ = [
    "QLIB_FIELDS",
    "QLIB_SNAPSHOT_SCHEMA_VERSION",
    "QlibSnapshotManifest",
    "build_qlib_snapshot",
]
