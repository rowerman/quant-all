"""策略版本和回测结果的 DuckDB 持久化。"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
import json
from pathlib import Path
import uuid

import duckdb
import pandas as pd

from backtest.builtin import BUILTINS
from backtest.config import DEFAULT_BACKTEST_DB
from backtest.engine import BacktestResult
from backtest.errors import BacktestError, RunConflictError, RunNotFoundError
from backtest.metrics import summarize
from backtest.specs import BacktestParams, StrategySpec

RESULT_TABLES = (
    "daily",
    "position",
    "order",
    "rejection",
    "target_weight",
    "holding",
    "liquidation",
    "allocation",
    "corporate_action",
)


def benchmark_code(value) -> str | None:
    """从运行元数据里的基准信息取出指数代码。"""
    if isinstance(value, dict):
        return value.get("code")
    return value


class BacktestStore:
    """管理策略不可变版本和回测运行产物。"""

    def __init__(self, path: Path | str = DEFAULT_BACKTEST_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._ensure_schema(conn)

    def _connect(self):
        return duckdb.connect(str(self.path))

    @staticmethod
    def _ensure_schema(conn) -> None:
        """创建策略和回测结果表。"""
        conn.execute("CREATE SCHEMA IF NOT EXISTS strategy")
        # 数据库文件默认名也是 backtest，DuckDB 会把它作为 catalog 名，
        # 因此使用 bt 避免 `backtest.run` 的 catalog/schema 歧义。
        conn.execute("CREATE SCHEMA IF NOT EXISTS bt")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS strategy.definition (
                strategy_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                current_version INTEGER,
                description VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS strategy.version (
                strategy_id VARCHAR,
                version INTEGER,
                config_json VARCHAR,
                config_hash VARCHAR,
                created_at TIMESTAMP,
                PRIMARY KEY(strategy_id, version)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS bt.run (
                run_id VARCHAR PRIMARY KEY,
                strategy_id VARCHAR,
                strategy_version INTEGER,
                start_date DATE,
                end_date DATE,
                init_cash DOUBLE,
                status VARCHAR,
                metadata_json VARCHAR,
                error_message VARCHAR,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                universe VARCHAR,
                benchmark VARCHAR
            )"""
        )
        # 老库没有这两列：股票池与基准原来只在 metadata_json 里，SQL 里筛不了
        conn.execute("ALTER TABLE bt.run ADD COLUMN IF NOT EXISTS universe VARCHAR")
        conn.execute("ALTER TABLE bt.run ADD COLUMN IF NOT EXISTS benchmark VARCHAR")
        for table in RESULT_TABLES:
            conn.execute(f"CREATE TABLE IF NOT EXISTS bt.{table} (run_id VARCHAR)")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _insert_version(conn, strategy_id: str, spec: StrategySpec, description: str, now: datetime) -> int:
        """写入策略第一版，返回版本号。"""
        conn.execute(
            "INSERT INTO strategy.definition VALUES (?, ?, ?, ?, ?, ?)",
            [strategy_id, spec.name, spec.version, description, now, now],
        )
        conn.execute(
            "INSERT INTO strategy.version VALUES (?, ?, ?, ?, ?)",
            [strategy_id, spec.version, spec.to_json(), spec.config_hash, now],
        )
        return spec.version

    def create_strategy(self, spec: StrategySpec, description: str = "") -> tuple[str, int]:
        """创建策略并保存第一版配置。"""
        strategy_id = uuid.uuid4().hex
        now = self._now()
        with self._connect() as conn:
            version = self._insert_version(conn, strategy_id, spec, description, now)
        return strategy_id, version

    def ensure_strategy(self, strategy_id: str, spec: StrategySpec, description: str = "") -> tuple[int, bool]:
        """按固定标识确保策略存在；定义变了就追加新版本。

        参数:
            strategy_id: 固定策略标识。
            spec: 目标配置。
            description: 策略说明。

        返回:
            ``(当前版本, 是否新建了版本)``。
        """
        now = self._now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT current_version FROM strategy.definition WHERE strategy_id = ?",
                [strategy_id],
            ).fetchone()
            if row is None:
                return self._insert_version(conn, strategy_id, spec, description, now), True
            current = int(row[0])
            stored = conn.execute(
                "SELECT config_json, config_hash FROM strategy.version WHERE strategy_id = ? AND version = ?",
                [strategy_id, current],
            ).fetchone()
        desired = replace(spec, version=current)
        if stored is not None and stored[1] == desired.config_hash:
            return current, False
        if stored is not None:
            stored_spec = StrategySpec.from_dict(json.loads(stored[0]))
            if replace(stored_spec, version=current).to_json() == desired.to_json():
                now = self._now()
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE strategy.version SET config_json = ?, config_hash = ? "
                        "WHERE strategy_id = ? AND version = ?",
                        [desired.to_json(), desired.config_hash, strategy_id, current],
                    )
                return current, False
        return self.save_version(strategy_id, spec, description), True

    def save_version(self, strategy_id: str, spec: StrategySpec, description: str = "") -> int:
        """为已有策略追加不可变版本。"""
        now = self._now()
        with self._connect() as conn:
            current = conn.execute(
                "SELECT current_version FROM strategy.definition WHERE strategy_id = ?",
                [strategy_id],
            ).fetchone()
            if current is None:
                raise BacktestError(f"策略不存在：{strategy_id}")
            version = int(current[0]) + 1
            versioned = replace(spec, version=version)
            conn.execute(
                "INSERT INTO strategy.version VALUES (?, ?, ?, ?, ?)",
                [strategy_id, version, versioned.to_json(), versioned.config_hash, now],
            )
            conn.execute(
                "UPDATE strategy.definition SET name = ?, current_version = ?, description = ?, updated_at = ? "
                "WHERE strategy_id = ?",
                [spec.name, version, description, now, strategy_id],
            )
        return version

    def ensure_builtins(self) -> list[dict]:
        """把内置策略写进策略表，与用户策略共用同一条运行路径。"""
        for strategy_id, spec, description in BUILTINS:
            self.ensure_strategy(strategy_id, spec, description)
        return self.list_strategies()

    def list_strategies(self) -> list[dict]:
        """列出策略定义。"""
        with self._connect() as conn:
            return conn.execute(
                "SELECT strategy_id, name, current_version, description, created_at, updated_at "
                "FROM strategy.definition ORDER BY updated_at DESC"
            ).df().to_dict("records")

    def get_run(self, run_id: str) -> dict:
        """读取回测运行状态。"""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM bt.run WHERE run_id = ?", [run_id]).fetchone()
            if row is None:
                raise RunNotFoundError(f"回测运行不存在：{run_id}")
            columns = [item[0] for item in conn.execute("DESCRIBE bt.run").fetchall()]
        return dict(zip(columns, row))

    def delete_run(self, run_id: str) -> None:
        """事务删除一次已结束运行及其全部结果表记录。

        参数:
            run_id: 要删除的回测运行标识。
        """
        with self._connect() as conn:
            row = conn.execute("SELECT status FROM bt.run WHERE run_id = ?", [run_id]).fetchone()
            if row is None:
                raise RunNotFoundError(f"回测运行不存在：{run_id}")
            if row[0] == "running":
                raise RunConflictError(f"回测运行仍在执行，不能删除：{run_id}")
            conn.begin()
            try:
                for table in RESULT_TABLES:
                    conn.execute(f"DELETE FROM bt.{table} WHERE run_id = ?", [run_id])
                conn.execute("DELETE FROM bt.run WHERE run_id = ?", [run_id])
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def list_runs(self, limit: int = 50) -> list[dict]:
        """按开始时间倒序列出回测运行。

        参数:
            limit: 最多返回多少条。
        """
        with self._connect() as conn:
            frame = conn.execute(
                "SELECT run_id, strategy_id, strategy_version, start_date, end_date, init_cash, "
                "status, started_at, finished_at, error_message, universe, benchmark, metadata_json "
                "FROM bt.run "
                "ORDER BY started_at DESC NULLS LAST LIMIT ?",
                [int(limit)],
            ).df()
        records = frame.to_dict("records")
        for record in records:
            metadata = json.loads(record.pop("metadata_json")) if record.get("metadata_json") else {}
            record["strategy_hash"] = metadata.get("strategy_hash")
            # 老运行只在元数据里有这两项，取不到就留空
            record["universe"] = record.get("universe") or metadata.get("universe")
            record["benchmark"] = record.get("benchmark") or benchmark_code(metadata.get("benchmark"))
            record["elapsed_seconds"] = metadata.get("elapsed_seconds")
            record["metrics"] = metadata.get("metrics")
        return records

    def _metadata(self, run_id: str) -> dict:
        """读取运行的元数据 JSON。"""
        with self._connect() as conn:
            row = conn.execute("SELECT metadata_json FROM bt.run WHERE run_id = ?", [run_id]).fetchone()
        if row is None or not row[0]:
            return {}
        return json.loads(row[0])

    def get_metrics(self, run_id: str) -> dict:
        """按已落库的日结果重算绩效指标，老运行也能查。"""
        with self._connect() as conn:
            frame = conn.execute("SELECT * FROM bt.daily WHERE run_id = ?", [run_id]).df()
        if frame.empty:
            raise BacktestError(f"回测运行没有日结果：{run_id}")
        date_column = "trade_date" if "trade_date" in frame.columns else "index"
        frame = frame.copy()
        frame[date_column] = pd.to_datetime(frame[date_column])
        return summarize(frame.set_index(date_column).sort_index())

    def get_result_table(self, run_id: str, table: str) -> list[dict]:
        """读取某次运行的标准化结果表。"""
        if table not in RESULT_TABLES:
            raise BacktestError(f"不支持的回测结果表：{table}；可选 {list(RESULT_TABLES)}")
        with self._connect() as conn:
            return conn.execute(f"SELECT * FROM bt.{table} WHERE run_id = ?", [run_id]).df().to_dict("records")

    def get_version(self, strategy_id: str, version: int | None = None) -> StrategySpec:
        """读取指定版本策略。"""
        with self._connect() as conn:
            if version is None:
                row = conn.execute(
                    "SELECT current_version FROM strategy.definition WHERE strategy_id = ?",
                    [strategy_id],
                ).fetchone()
                if row is None:
                    raise BacktestError(f"策略不存在：{strategy_id}")
                version = int(row[0])
            row = conn.execute(
                "SELECT config_json FROM strategy.version WHERE strategy_id = ? AND version = ?",
                [strategy_id, version],
            ).fetchone()
        if row is None:
            raise BacktestError(f"策略版本不存在：{strategy_id} v{version}")
        return replace(StrategySpec.from_dict(json.loads(row[0])), version=version)

    def start_run(
        self,
        strategy_id: str,
        version: int,
        params: BacktestParams,
        universe: str | None = None,
        benchmark: str | None = None,
    ) -> str:
        """登记一次运行，记录本次实际使用的股票池与基准。"""
        run_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO bt.run (run_id, strategy_id, strategy_version, start_date, end_date, "
                "init_cash, status, universe, benchmark, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    strategy_id,
                    version,
                    params.start,
                    params.end,
                    params.init_cash,
                    "running",
                    universe,
                    benchmark,
                    self._now(),
                ],
            )
        return run_id

    @staticmethod
    def _write_frame(conn, table: str, frame: pd.DataFrame) -> None:
        """把标准化结果追加到指定表。"""
        if not len(frame.columns):
            return
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index(drop=False)
        conn.register("_backtest_frame", frame)
        existing = {row[0] for row in conn.execute(f"DESCRIBE bt.{table}").fetchall()}
        for name, data_type, *_ in conn.execute("DESCRIBE _backtest_frame").fetchall():
            if name not in existing:
                conn.execute(f'ALTER TABLE bt.{table} ADD COLUMN "{name}" {data_type}')
        columns = [row[0] for row in conn.execute("DESCRIBE _backtest_frame").fetchall()]
        quoted = ", ".join(f'"{name}"' for name in columns)
        conn.execute(f'INSERT INTO bt.{table} ({quoted}) SELECT {quoted} FROM _backtest_frame')
        conn.unregister("_backtest_frame")

    def finish_run(self, run_id: str, result: BacktestResult) -> None:
        """保存回测结果并标记成功。"""
        daily = result.daily.copy()
        daily.index.name = "trade_date"
        position = result.positions.rename(columns={"Column": "stock_code"}).copy()
        holdings = result.holdings.copy()
        orders = result.orders.copy()
        rejection = result.rejections.copy()
        liquidation = result.liquidations.copy()
        allocation = result.allocations.copy()
        corporate_action = result.corporate_actions.copy()
        target_weight = result.target_weights.stack(future_stack=True).dropna().rename("target_weight").reset_index()
        target_weight.columns = ["trade_date", "stock_code", "target_weight"]
        frames = []
        for table, frame in (
            ("daily", daily),
            ("position", position),
            ("holding", holdings),
            ("order", orders),
            ("rejection", rejection),
            ("liquidation", liquidation),
            ("allocation", allocation),
            ("corporate_action", corporate_action),
            ("target_weight", target_weight),
        ):
            if frame.empty:
                continue
            payload = frame.copy()
            payload.insert(0, "run_id", run_id)
            frames.append((table, payload))
        metadata = dict(result.metadata)
        metadata["metrics"] = result.metrics
        with self._connect() as conn:
            for table, payload in frames:
                self._write_frame(conn, table, payload)
            conn.execute(
                "UPDATE bt.run SET status = ?, metadata_json = ?, finished_at = ? WHERE run_id = ?",
                ["success", json.dumps(metadata, ensure_ascii=False, default=str), self._now(), run_id],
            )

    def fail_run(self, run_id: str, error: Exception) -> None:
        """保存失败状态和错误摘要。"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE bt.run SET status = ?, error_message = ?, finished_at = ? WHERE run_id = ?",
                ["failed", str(error), self._now(), run_id],
            )

    def cancel_run(self, run_id: str) -> None:
        """把运行标记为已取消。"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE bt.run SET status = ?, finished_at = ? WHERE run_id = ?",
                ["cancelled", self._now(), run_id],
            )

