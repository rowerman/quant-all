"""回测输入的内容摘要快照。

快照只记录本次运行实际使用到的数据范围与规则内容，不复制行情和因子原始数据。
行情、交易日历、股票基础表、公司行动、行业和基准通过 DuckDB 对精确行集做
顺序无关的哈希；因子直接读取落盘 Parquet。相同内容重复构建会得到相同
``snapshot_id``，区间外数据刷新不会改变摘要。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd

import factors
from quant.db import get_conn

from backtest.config import FACTOR_DB, FACTOR_DIR
from backtest.costs import cost_model_summary
from backtest.errors import DataAlignmentError, SpecError
from backtest.providers import provider_snapshot_metadata
from backtest.specs import StrategySpec, TOTAL_RETURN_BENCHMARKS
from backtest.trading_rules import RULE_VERSION, rule_summary
from backtest.universe import membership_rows, resolve_universe, universe_stocks


SNAPSHOT_SCHEMA_VERSION = "backtest_input_snapshot_v2"
DATA_SNAPSHOT_SCHEMA_VERSION = "backtest_data_snapshot_v2"
DB_HASH_ALGORITHM = "duckdb_hash_v1"
PANDAS_HASH_ALGORITHM = "pandas_hash_v1"
DELISTING_RULE_VERSION = "cn_a_delisting_v1"


@dataclass(frozen=True)
class InputSnapshot:
    """一次回测的不可变输入快照。"""

    snapshot_id: str
    schema_version: str
    created_at: str
    scope: dict[str, Any]
    components: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为可落库的字典。"""
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "scope": self.scope,
            "components": self.components,
            "provenance": self.provenance,
        }


def _canonical_json(payload: Any) -> str:
    """生成用于哈希的稳定 JSON。"""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_text(text: str) -> str:
    """计算文本的 SHA-256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: Any) -> str:
    """计算规范化 JSON 的 SHA-256。"""
    return _sha256_text(_canonical_json(payload))


def _quote(name: str) -> str:
    """引用 SQL 标识符。"""
    return '"' + str(name).replace('"', '""') + '"'


def _where(conditions: Iterable[str]) -> str:
    """拼出 WHERE 子句。"""
    values = [item for item in conditions if item]
    return (" WHERE " + " AND ".join(values)) if values else ""


def _stock_condition(stocks: Sequence[str] | None) -> tuple[str | None, list[str]]:
    """生成股票过滤条件。"""
    if not stocks:
        return None, []
    placeholders = ", ".join("?" for _ in stocks)
    return f"stock_code IN ({placeholders})", [str(item) for item in stocks]


def _component_digest(
    algorithm: str,
    columns: Sequence[dict[str, str]],
    rows: int,
    hash_sum: str,
    hash_xor: str,
) -> str:
    """把行数、列定义与哈希聚合结果压成一个摘要。"""
    return _sha256_payload(
        {
            "algorithm": algorithm,
            "columns": list(columns),
            "rows": int(rows),
            "sum": str(hash_sum),
            "xor": str(hash_xor),
        }
    )


def _sql_component(
    conn: duckdb.DuckDBPyConnection,
    source: str,
    params: list[Any],
    columns: Sequence[tuple[str, str, str]],
    where: str,
    label: str,
) -> dict[str, Any]:
    """对一条精确过滤的 SQL 行集生成摘要。

    参数:
        conn: DuckDB 连接。
        source: 表名或 ``read_parquet(...)`` 表达式。
        params: SQL 参数。
        columns: ``(名称, SQL 表达式, 类型说明)`` 序列。
        where: WHERE 子句。
        label: 出错时的数据名称。
    """
    selects = ", ".join(
        f"{expression} AS {_quote(name)}" for name, expression, _ in columns
    )
    inner = f"SELECT {selects} FROM {source}{where}"
    arguments = ", ".join(_quote(name) for name, _, _ in columns)
    sql = (
        "SELECT count(*) AS row_count, "
        f"coalesce(sum(hash({arguments})), 0)::VARCHAR AS hash_sum, "
        f"coalesce(bit_xor(hash({arguments})), 0)::VARCHAR AS hash_xor "
        f"FROM ({inner}) AS _snapshot_rows"
    )
    try:
        row = conn.execute(sql, params).fetchone()
    except Exception as error:
        raise DataAlignmentError(f"生成{label}快照失败：{error}") from error
    count = int(row[0])
    hash_sum, hash_xor = str(row[1]), str(row[2])
    column_payload = [{"name": name, "type": kind} for name, _, kind in columns]
    return {
        "algorithm": DB_HASH_ALGORITHM,
        "rows": count,
        "columns": column_payload,
        "sum": hash_sum,
        "xor": hash_xor,
        "digest": _component_digest(
            DB_HASH_ALGORITHM, column_payload, count, hash_sum, hash_xor
        ),
    }


def _pandas_component(
    frame: pd.DataFrame | pd.Series,
    *,
    include_index: bool = True,
    label: str = "数据",
) -> dict[str, Any]:
    """对小型 DataFrame 或合成面板生成稳定摘要。"""
    if isinstance(frame, pd.Series):
        payload = frame.to_frame(name=frame.name or "value")
    else:
        payload = frame
    if include_index:
        payload = payload.copy()
        payload.index.name = payload.index.name or "index"
    hashes = pd.util.hash_pandas_object(
        payload, index=include_index, categorize=False
    ).to_numpy(dtype="uint64")
    count = int(len(hashes))
    hash_sum = str(sum(int(item) for item in hashes))
    hash_xor = str(int(np.bitwise_xor.reduce(hashes))) if count else "0"
    columns = [
        {"name": str(name), "type": str(dtype)}
        for name, dtype in payload.dtypes.items()
    ]
    return {
        "algorithm": PANDAS_HASH_ALGORITHM,
        "rows": count,
        "columns": columns,
        "sum": hash_sum,
        "xor": hash_xor,
        "digest": _component_digest(
            PANDAS_HASH_ALGORITHM, columns, count, hash_sum, hash_xor
        ),
    }


def _stock_set_summary(stocks: Sequence[str] | None) -> dict[str, Any]:
    """描述股票并集；None 表示全 A 不额外收窄。"""
    if not stocks:
        return {"all_a": True, "count": None, "sha256": None}
    ordered = sorted({str(item) for item in stocks})
    return {
        "all_a": False,
        "count": len(ordered),
        "sha256": _sha256_payload(ordered),
    }


def _market_columns() -> tuple[tuple[str, str, str], ...]:
    """回测实际加载的行情与规则列。"""
    return (
        ("stock_code", '"stock_code"', "VARCHAR"),
        ("trade_date", '"trade_date"', "DATE"),
        ("open", 'CAST("open" AS FLOAT)', "FLOAT"),
        ("close", 'CAST("close" AS FLOAT)', "FLOAT"),
        ("can_buy_close", 'CAST("can_buy_close" AS BOOLEAN)', "BOOLEAN"),
        ("can_buy_open_next", 'CAST("can_buy_open_next" AS BOOLEAN)', "BOOLEAN"),
        ("can_sell_open_next", 'CAST("can_sell_open_next" AS BOOLEAN)', "BOOLEAN"),
        ("is_suspended", 'CAST("is_suspended" AS BOOLEAN)', "BOOLEAN"),
        ("is_st", 'CAST("is_st" AS BOOLEAN)', "BOOLEAN"),
        ("is_delisting", 'CAST("is_delisting" AS BOOLEAN)', "BOOLEAN"),
        ("is_limit_up", 'CAST("is_limit_up" AS BOOLEAN)', "BOOLEAN"),
        ("is_limit_down", 'CAST("is_limit_down" AS BOOLEAN)', "BOOLEAN"),
        ("total_mv", 'CAST("total_mv" AS FLOAT)', "FLOAT"),
        ("amount_cny", 'CAST("amount_cny" AS FLOAT)', "FLOAT"),
    )


def _corporate_action_columns() -> tuple[tuple[str, str, str], ...]:
    """公司行动账本字段。"""
    return (
        ("stock_code", '"stock_code"', "VARCHAR"),
        ("ex_date", 'CAST("ex_date" AS DATE)', "DATE"),
        ("record_date", 'CAST("record_date" AS DATE)', "DATE"),
        ("pay_date", 'CAST("pay_date" AS DATE)', "DATE"),
        ("share_multiplier", 'CAST("share_multiplier" AS DOUBLE)', "DOUBLE"),
        ("cash_div_per_share", 'CAST("cash_div_per_share" AS DOUBLE)', "DOUBLE"),
        ("action_type", '"action_type"', "VARCHAR"),
        ("source", '"source"', "VARCHAR"),
        ("source_event_key", '"source_event_key"', "VARCHAR"),
    )


def _industry_columns() -> tuple[tuple[str, str, str], ...]:
    """申万历史行业字段。"""
    return (
        ("stock_code", '"stock_code"', "VARCHAR"),
        ("classification", 'CAST("classification" AS VARCHAR)', "VARCHAR"),
        ("l1_code", '"l1_code"', "VARCHAR"),
        ("l1_name", '"l1_name"', "VARCHAR"),
        ("l2_code", '"l2_code"', "VARCHAR"),
        ("l2_name", '"l2_name"', "VARCHAR"),
        ("l3_code", '"l3_code"', "VARCHAR"),
        ("l3_name", '"l3_name"', "VARCHAR"),
        ("in_date", 'CAST("in_date" AS DATE)', "DATE"),
        ("out_date", 'CAST("out_date" AS DATE)', "DATE"),
        ("source_in_date", 'CAST("source_in_date" AS DATE)', "DATE"),
        ("source_out_date", 'CAST("source_out_date" AS DATE)', "DATE"),
        ("source_l3_code", '"source_l3_code"', "VARCHAR"),
        ("source_api", '"source_api"', "VARCHAR"),
        ("carry_forward", 'CAST("carry_forward" AS BOOLEAN)', "BOOLEAN"),
        ("is_new", '"is_new"', "VARCHAR"),
    )


def _factor_family(name: str) -> tuple[str, str]:
    """返回因子所属族和定义摘要。"""
    try:
        spec = factors.get(name)
    except Exception as error:
        raise DataAlignmentError(f"未知因子 {name}：{error}") from error
    definition = {
        "name": spec.name,
        "family": spec.family,
        "layers": list(spec.layers),
        "lookback": int(spec.lookback),
        "direction": int(spec.direction),
        "fields": list(spec.fields),
        "extra_sources": list(spec.extra_sources),
    }
    return spec.family, _sha256_payload(definition)


def _factor_component(
    name: str,
    start: str,
    end: str,
    stocks: Sequence[str] | None,
    factor_dir: Path | str,
) -> dict[str, Any]:
    """对单个因子在运行范围内的值生成摘要。"""
    family, definition_digest = _factor_family(name)
    root = Path(factor_dir) / family / "year=*" / "part.parquet"
    if not list(Path(factor_dir).glob(f"{family}/year=*/part.parquet")):
        raise DataAlignmentError(
            f"因子 {name} 所属族 {family} 没有落盘数据：{Path(factor_dir) / family}"
        )
    source = f"read_parquet('{str(root).replace(chr(39), chr(39) * 2)}', hive_partitioning = 1)"
    conditions = ["trade_date >= CAST(? AS DATE)", "trade_date <= CAST(? AS DATE)"]
    params: list[Any] = [start, end]
    stock_sql, stock_params = _stock_condition(stocks)
    if stock_sql:
        conditions.append(stock_sql)
        params.extend(stock_params)
    columns = (
        ("stock_code", '"stock_code"', "VARCHAR"),
        ("trade_date", 'CAST("trade_date" AS DATE)', "DATE"),
        (name, f'CAST({_quote(name)} AS FLOAT)', "FLOAT"),
    )
    with duckdb.connect() as conn:
        component = _sql_component(
            conn, source, params, columns, _where(conditions), f"因子 {name} "
        )
    component["family"] = family
    component["definition_sha256"] = definition_digest
    component["digest"] = _component_digest(
        component["algorithm"],
        [
            *component["columns"],
            {"name": "_definition_sha256", "type": "VARCHAR"},
        ],
        int(component["rows"]),
        f'{component["sum"]}:{definition_digest}',
        component["xor"],
    )
    return component


def _index_name(conn: duckdb.DuckDBPyConnection, code: str) -> str:
    """读取基准指数名称；老库缺表时退回代码。"""
    try:
        row = conn.execute(
            "SELECT index_name FROM ref.index_basic WHERE index_code = ? LIMIT 1",
            [code],
        ).fetchone()
    except Exception:
        return code
    return str(row[0]) if row and row[0] is not None else code


def _has_column(
    conn: duckdb.DuckDBPyConnection, table: str, column: str
) -> bool:
    """判断表是否有指定列。"""
    try:
        columns = {
            str(row[0])
            for row in conn.execute(f"DESCRIBE {table}").fetchall()
        }
    except Exception:
        return False
    return column in columns


def _industry_manifest(conn: duckdb.DuckDBPyConnection) -> str | None:
    """读取申万行业内容摘要，缺失时返回 None。"""
    try:
        row = conn.execute(
            "SELECT digest FROM meta.sw_industry_manifest LIMIT 1"
        ).fetchone()
    except Exception:
        return None
    return None if row is None or row[0] is None else str(row[0])


def build_data_snapshot(
    start: str,
    end: str,
    factor_names: Sequence[str],
    *,
    universe: str = "all_a",
    db_path: Path | str | None = None,
    factor_dir: Path | str | None = None,
    benchmark_code: str | None = "H00300.CSI",
    membership_end: str | None = None,
    stock_codes: Sequence[str] | None = None,
    industry_level: str | None = None,
    industry_report: bool = False,
    impact_volatility_window: int | None = None,
) -> dict[str, Any]:
    """构造一次回测的数据部分快照。

    参数:
        start: 起始交易日。
        end: 截止交易日。
        factor_names: 本次实际读取的因子。
        universe: 股票池标识。
        db_path: 行情库路径。
        factor_dir: 因子数据根目录。
        benchmark_code: 基准代码；None 表示关闭。
        membership_end: 指数成分截止日；None 表示使用 ``end``。
        stock_codes: 固定的股票并集；None 时按股票池解析。
        industry_level: ``L1`` 表示加载行业约束，None 表示不加载。
        industry_report: True 表示加载行业用于报告；行业缺失时允许降级。
        impact_volatility_window: None 表示关闭冲击波动率；整数表示窗口长度。
    """
    if industry_level not in (None, "L1"):
        raise DataAlignmentError("行业快照只支持 None/L1")
    spec = resolve_universe(universe)
    if stock_codes is None and spec.is_index:
        stock_codes = universe_stocks(spec, start, end, db_path=db_path)
    stock_codes = None if stock_codes is None else tuple(sorted(set(stock_codes)))
    factor_names = tuple(dict.fromkeys(str(item) for item in factor_names))
    membership_end = membership_end or end
    root = Path(factor_dir or FACTOR_DIR)

    from backtest.data import PRICE_ADJUST, STATE_ADJUST, _load_versions

    try:
        with get_conn(db_path) as conn:
            date_conditions = [
                "trade_date >= CAST(? AS DATE)",
                "trade_date <= CAST(? AS DATE)",
            ]
            date_params: list[Any] = [start, end]
            stock_sql, stock_params = _stock_condition(stock_codes)
            market_conditions = list(date_conditions)
            market_params = list(date_params)
            if stock_sql:
                market_conditions.append(stock_sql)
                market_params.extend(stock_params)
            market = _sql_component(
                conn,
                "md.daily_bar",
                market_params,
                _market_columns(),
                _where(market_conditions),
                "行情",
            )
            calendar = _sql_component(
                conn,
                "md.calendar",
                date_params,
                (("trade_date", 'CAST("trade_date" AS DATE)', "DATE"),),
                _where([*date_conditions, "is_open = 1"]),
                "交易日历",
            )

            basic_conditions = ["list_date <= CAST(? AS DATE)"]
            basic_params: list[Any] = [end]
            if stock_sql:
                basic_conditions.append(stock_sql)
                basic_params.extend(stock_params)
            if _has_column(conn, "ref.stock_basic", "delist_date"):
                basic_conditions.append(
                    "(delist_date IS NULL OR delist_date >= CAST(? AS DATE))"
                )
                basic_params.append(start)
            basic = _sql_component(
                conn,
                "ref.stock_basic",
                basic_params,
                (
                    ("stock_code", '"stock_code"', "VARCHAR"),
                    ("market", '"market"', "VARCHAR"),
                    ("list_date", 'CAST("list_date" AS DATE)', "DATE"),
                    (
                        "delist_date",
                        'CAST("delist_date" AS DATE)'
                        if _has_column(conn, "ref.stock_basic", "delist_date")
                        else "NULL::DATE",
                        "DATE",
                    ),
                ),
                _where(basic_conditions),
                "股票基础表",
            )

            actions_conditions = [
                "ex_date >= CAST(? AS DATE)",
                "ex_date <= CAST(? AS DATE)",
            ]
            actions_params: list[Any] = [start, end]
            if stock_sql:
                actions_conditions.append(stock_sql)
                actions_params.extend(stock_params)
            actions = _sql_component(
                conn,
                "ref.corporate_action",
                actions_params,
                _corporate_action_columns(),
                _where(actions_conditions),
                "公司行动",
            )

            industry = {"enabled": False}
            if industry_level == "L1" or industry_report:
                industry_conditions = [
                    "(out_date IS NULL OR out_date >= CAST(? AS DATE))",
                    "in_date <= CAST(? AS DATE)",
                ]
                industry_params: list[Any] = [start, end]
                if stock_sql:
                    industry_conditions.append(stock_sql)
                    industry_params.extend(stock_params)
                try:
                    industry = {
                        "enabled": True,
                        "manifest_digest": _industry_manifest(conn),
                        **_sql_component(
                            conn,
                            "ref.sw_industry_member",
                            industry_params,
                            _industry_columns(),
                            _where(industry_conditions),
                            "申万行业",
                        ),
                    }
                except Exception as error:
                    if industry_level == "L1":
                        raise
                    industry = {"enabled": False, "reason": str(error)}

            impact_volatility: dict[str, Any] = {"enabled": False}
            if impact_volatility_window is not None:
                lookback_row = conn.execute(
                    "SELECT min(trade_date) FROM ("
                    "SELECT trade_date FROM md.calendar WHERE is_open = 1 "
                    "AND trade_date < CAST(? AS DATE) "
                    "ORDER BY trade_date DESC LIMIT ?"
                    ")",
                    [start, int(impact_volatility_window)],
                ).fetchone()
                lookback = (
                    str(pd.Timestamp(lookback_row[0]).date())
                    if lookback_row and lookback_row[0] is not None
                    else start
                )
                impact_conditions = [
                    "trade_date >= CAST(? AS DATE)",
                    "trade_date <= CAST(? AS DATE)",
                ]
                impact_params: list[Any] = [lookback, end]
                if stock_sql:
                    impact_conditions.append(stock_sql)
                    impact_params.extend(stock_params)
                impact_volatility = {
                    "enabled": True,
                    "window": int(impact_volatility_window),
                    **_sql_component(
                        conn,
                        "md.daily_hfq",
                        impact_params,
                        (
                            ("stock_code", '"stock_code"', "VARCHAR"),
                            ("trade_date", 'CAST("trade_date" AS DATE)', "DATE"),
                            ("ret", 'CAST("ret" AS DOUBLE)', "DOUBLE"),
                        ),
                        _where(impact_conditions),
                        "冲击波动率",
                    ),
                }

            benchmark: dict[str, Any] = {"enabled": False}
            if benchmark_code:
                benchmark = {
                    "enabled": True,
                    "code": str(benchmark_code),
                    "name": _index_name(conn, str(benchmark_code)),
                    "return_type": (
                        "total_return"
                        if benchmark_code in TOTAL_RETURN_BENCHMARKS
                        or "收益" in _index_name(conn, str(benchmark_code))
                        else "price"
                    ),
                    **_sql_component(
                        conn,
                        "md.index_daily",
                        [str(benchmark_code), start, end],
                        (
                            ("index_code", '"index_code"', "VARCHAR"),
                            ("trade_date", 'CAST("trade_date" AS DATE)', "DATE"),
                            ("close", 'CAST("close" AS DOUBLE)', "DOUBLE"),
                        ),
                        _where(
                            [
                                "index_code = ?",
                                "trade_date >= CAST(? AS DATE)",
                                "trade_date <= CAST(? AS DATE)",
                            ]
                        ),
                        "基准行情",
                    ),
                }

        membership: dict[str, Any] = {"enabled": False}
        if spec.is_index:
            rows = membership_rows(spec, membership_end, db_path=db_path)
            rows = rows.loc[rows["period"] >= pd.Period(start, freq="M")]
            membership = {
                "enabled": True,
                "index_code": spec.index_code,
                **_pandas_component(rows, include_index=False, label="指数成分"),
            }

        factor_components = {
            name: _factor_component(name, start, end, stock_codes, root)
            for name in factor_names
        }
    except DataAlignmentError:
        raise
    except Exception as error:
        raise DataAlignmentError(f"生成回测数据快照失败：{error}") from error

    stock_summary = _stock_set_summary(stock_codes)
    if stock_summary["all_a"]:
        stock_summary["count"] = int(basic["rows"])
    scope = {
        "start": str(start),
        "end": str(end),
        "universe": spec.key,
        "membership_end": str(membership_end),
        "stocks": stock_summary,
        "factors": list(factor_names),
        "benchmark": None if benchmark_code is None else str(benchmark_code),
        "industry_level": industry_level,
        "industry_report": bool(industry_report),
        "impact_volatility_window": impact_volatility_window,
        "price_adjust": PRICE_ADJUST,
        "state_adjust": STATE_ADJUST,
    }
    components = {
        "market": market,
        "calendar": calendar,
        "universe": {
            "stock_basic": basic,
            "membership": membership,
        },
        "factors": factor_components,
        "corporate_actions": actions,
        "industry": industry,
        "benchmark": benchmark,
        "impact_volatility": impact_volatility,
    }
    payload = {
        "schema_version": DATA_SNAPSHOT_SCHEMA_VERSION,
        "scope": scope,
        "components": components,
    }
    return {
        **payload,
        "data_id": _sha256_payload(payload),
        "provenance": {"versions": _load_versions(db_path, FACTOR_DB)},
    }


def _rules_component(strategy: StrategySpec) -> dict[str, Any]:
    """返回策略配置、交易规则和提供者源码的摘要内容。"""
    provider = (
        provider_snapshot_metadata(strategy.signal.provider)
        if strategy.signal.provider is not None
        else None
    )
    return {
        "strategy": strategy.to_dict(),
        "strategy_hash": strategy.config_hash,
        "board_rule_version": RULE_VERSION,
        "board_rules": rule_summary(),
        "cost_model": cost_model_summary(strategy.costs),
        "delisting_rule_version": DELISTING_RULE_VERSION,
        "provider": provider,
    }


def build_input_snapshot(
    strategy: StrategySpec,
    data_snapshot: dict[str, Any] | None,
    *,
    provenance: dict[str, Any] | None = None,
) -> InputSnapshot:
    """把数据摘要和策略规则合成完整输入快照。

    参数:
        strategy: 本次执行的策略配置。
        data_snapshot: :func:`build_data_snapshot` 的结果。
        provenance: 可选的构建清单等审计信息，不参与快照 ID。
    """
    if not data_snapshot or "components" not in data_snapshot:
        raise SpecError("缺少回测数据快照，无法生成输入快照")
    components = dict(data_snapshot["components"])
    components["rules"] = _rules_component(strategy)
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "scope": data_snapshot["scope"],
        "components": components,
    }
    snapshot_id = _sha256_payload(payload)
    source_provenance = dict(data_snapshot.get("provenance") or {})
    if data_snapshot.get("data_id"):
        source_provenance["data_id"] = str(data_snapshot["data_id"])
    if provenance:
        source_provenance.update(provenance)
    return InputSnapshot(
        snapshot_id=snapshot_id,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        scope=dict(data_snapshot["scope"]),
        components=components,
        provenance=source_provenance,
    )


def data_snapshot_from_market_data(data) -> dict[str, Any]:
    """从合成或外部 ``MarketData`` 生成调试用数据摘要。

    正常数据库取数使用 :func:`build_data_snapshot`。这里只用于测试夹具和
    不经过数据库的自定义面板，摘要算法单独标记为 ``pandas_hash_v1``。
    """
    from backtest.data import PANEL_FIELDS, PRICE_ADJUST, STATE_ADJUST

    index = pd.DatetimeIndex(data.index)
    market = {
        name: _pandas_component(getattr(data, name), label=f"行情 {name}")
        for name in PANEL_FIELDS
    }
    factors_payload = {
        name: _pandas_component(frame, label=f"因子 {name}")
        for name, frame in data.factors.items()
    }
    basic = pd.DataFrame(
        {
            "board": data.board,
            "list_date": data.list_date,
            "delist_date": (
                data.delist_date
                if data.delist_date is not None
                else pd.Series(pd.NaT, index=data.columns)
            ),
        }
    )
    universe = {
        "stock_basic": _pandas_component(basic, label="股票基础表"),
        "membership": (
            {"enabled": False}
            if data.membership is None
            else {
                "enabled": True,
                **_pandas_component(data.membership, label="指数成分"),
            }
        ),
    }
    industry = (
        {"enabled": False}
        if data.industry_l1 is None
        else {
            "enabled": True,
            "labels_sha256": _sha256_payload(
                {str(key): str(value) for key, value in (data.industry_labels or {}).items()}
            ),
            **_pandas_component(data.industry_l1, label="申万行业"),
        }
    )
    benchmark = (
        {"enabled": False}
        if data.benchmark is None
        else {
            "enabled": True,
            "info": data.benchmark_info or {},
            **_pandas_component(data.benchmark, label="基准行情"),
        }
    )
    actions = (
        {"enabled": False, "rows": 0}
        if data.corporate_actions is None
        else {
            "enabled": True,
            **_pandas_component(data.corporate_actions, include_index=False, label="公司行动"),
        }
    )
    impact_volatility = (
        {"enabled": False}
        if data.impact_volatility is None
        else {
            "enabled": True,
            **_pandas_component(data.impact_volatility, label="冲击波动率"),
        }
    )
    components = {
        "market": {
            "algorithm": PANDAS_HASH_ALGORITHM,
            "fields": market,
            "digest": _sha256_payload(market),
        },
        "calendar": _pandas_component(
            pd.Series(index, index=index), include_index=False, label="交易日历"
        ),
        "universe": universe,
        "factors": {
            "algorithm": PANDAS_HASH_ALGORITHM,
            "factors": factors_payload,
            "digest": _sha256_payload(factors_payload),
        },
        "corporate_actions": actions,
        "industry": industry,
        "benchmark": benchmark,
        "impact_volatility": impact_volatility,
    }
    scope = {
        "start": str(index.min().date()) if len(index) else None,
        "end": str(index.max().date()) if len(index) else None,
        "universe": "custom",
        "membership_end": str(index.max().date()) if len(index) else None,
        "stocks": {
            "all_a": False,
            "count": int(len(data.columns)),
            "sha256": _sha256_payload([str(item) for item in data.columns]),
        },
        "factors": list(data.factors),
        "benchmark": (
            None if data.benchmark_info is None else data.benchmark_info.get("code")
        ),
        "industry_level": None if data.industry_l1 is None else "L1",
        "industry_report": data.industry_l1 is not None,
        "impact_volatility_window": (
            None
            if data.impact_volatility is None
            else 20
        ),
        "price_adjust": PRICE_ADJUST,
        "state_adjust": STATE_ADJUST,
    }
    payload = {
        "schema_version": DATA_SNAPSHOT_SCHEMA_VERSION,
        "scope": scope,
        "components": components,
    }
    return {
        **payload,
        "data_id": _sha256_payload(payload),
        "provenance": {"versions": data.versions},
    }
