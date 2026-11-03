"""生成不依赖外部授权的确定性 A 股演示数据。

数据只用于验证读取、因子评估、回测和页面链路，不代表真实行情或投资结论。
输出目录与真实 ``data/`` 分离，生成完成后通过 ``QUANT_DATA_DIR`` 使用。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pandas as pd

from quant.schema import DAILY_COLUMNS, LABEL_HORIZONS


BOOLEAN_COLUMNS = (
    "is_suspended",
    "is_resume",
    "is_ex_div",
    "is_st",
    "is_delisting",
    "is_limit_up",
    "is_limit_down",
    "touched_limit_up",
    "touched_limit_down",
    "is_one_word_board",
    "is_price_limit_applicable",
    "can_buy_close",
    "can_sell_close",
    "can_buy_open_next",
    "can_sell_open_next",
)
DATE_COLUMNS = ("trade_date", "list_date", "delist_date")


def _stock_codes(count: int) -> list[str]:
    """生成上海和深圳主板代码。"""

    shanghai = min(count, max(1, count * 3 // 4))
    codes = [f"SH{600000 + index:06d}" for index in range(shanghai)]
    codes.extend(f"SZ{index + 1:06d}" for index in range(count - shanghai))
    return codes


def _create_table(
    conn: duckdb.DuckDBPyConnection,
    schema: str,
    name: str,
    frame: pd.DataFrame,
) -> None:
    """按统一日期和布尔类型创建表。"""

    columns = []
    for column in frame.columns:
        if column in DATE_COLUMNS:
            columns.append(f'CAST("{column}" AS DATE) AS "{column}"')
        elif column in BOOLEAN_COLUMNS:
            columns.append(f'CAST("{column}" AS BOOLEAN) AS "{column}"')
        else:
            columns.append(f'"{column}"')
    conn.register("_demo_frame", frame)
    conn.execute(
        f'CREATE TABLE {schema}.{name} AS SELECT {", ".join(columns)} FROM _demo_frame'
    )
    conn.unregister("_demo_frame")


def _daily_frame(
    dates: pd.DatetimeIndex,
    codes: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成同构日线长表和标签表。"""

    rng = np.random.default_rng(seed)
    market_return = rng.normal(0.0002, 0.012, len(dates))
    exposures = rng.normal(0.0, 0.35, (len(dates), len(codes)))
    returns = market_return[:, None] + exposures * 0.015 + rng.normal(
        0.0001, 0.009, (len(dates), len(codes))
    )
    close_wide = 12.0 * np.exp(np.cumsum(returns, axis=0))
    open_wide = close_wide * np.exp(rng.normal(0.0, 0.003, close_wide.shape))
    high_wide = np.maximum(open_wide, close_wide) * (1 + rng.uniform(0, 0.012, close_wide.shape))
    low_wide = np.minimum(open_wide, close_wide) * (1 - rng.uniform(0, 0.012, close_wide.shape))
    amount_wide = rng.lognormal(17.0, 0.5, close_wide.shape)

    close = pd.DataFrame(close_wide, index=dates, columns=codes)
    open_price = pd.DataFrame(open_wide, index=dates, columns=codes)
    high = pd.DataFrame(high_wide, index=dates, columns=codes)
    low = pd.DataFrame(low_wide, index=dates, columns=codes)
    amount = pd.DataFrame(amount_wide, index=dates, columns=codes)

    rows = []
    for code_index, code in enumerate(codes):
        code_close = close[code]
        code_open = open_price[code]
        pre_close = code_close.shift(1).fillna(code_close.iloc[0])
        ret = code_close.pct_change().fillna(0.0)
        for date_index, trade_date in enumerate(dates):
            rows.append(
                {
                    "stock_code": code,
                    "trade_date": trade_date,
                    "open": float(code_open.iloc[date_index]),
                    "high": float(high.loc[trade_date, code]),
                    "low": float(low.loc[trade_date, code]),
                    "close": float(code_close.iloc[date_index]),
                    "volume_share": float(amount.loc[trade_date, code] / code_close.iloc[date_index]),
                    "amount_cny": float(amount.loc[trade_date, code]),
                    "vwap": float((code_open.iloc[date_index] + code_close.iloc[date_index]) / 2),
                    "factor": 1.0,
                    "change": float(code_close.iloc[date_index] - pre_close.iloc[date_index]),
                    "pre_close": float(pre_close.iloc[date_index]),
                    "ret": float(ret.iloc[date_index]),
                    "status": "trading",
                    "is_suspended": False,
                    "is_resume": False,
                    "is_ex_div": False,
                    "is_st": False,
                    "is_delisting": False,
                    "list_date": pd.Timestamp("2018-01-02"),
                    "delist_date": pd.NaT,
                    "pe_ttm": float(12 + (code_index % 17)),
                    "pb": float(1.2 + (code_index % 9) / 10),
                    "ps_ttm": float(1.5 + (code_index % 5) / 4),
                    "dv_ttm": float(1.0 + (code_index % 4) / 2),
                    "total_mv": float(2e9 + code_index * 4e7),
                    "circ_mv": float(1.5e9 + code_index * 3e7),
                    "turnover_rate": float(0.5 + (code_index % 7) / 10),
                    "limit_ratio": 0.1,
                    "limit_up": float(round(pre_close.iloc[date_index] * 1.1, 2)),
                    "limit_down": float(round(pre_close.iloc[date_index] * 0.9, 2)),
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "touched_limit_up": False,
                    "touched_limit_down": False,
                    "is_one_word_board": False,
                    "is_price_limit_applicable": True,
                    "can_buy_close": True,
                    "can_sell_close": True,
                    "can_buy_open_next": True,
                    "can_sell_open_next": True,
                    "pre_close_source": "synthetic",
                    "st_source": "synthetic",
                }
            )
    daily = pd.DataFrame(rows)

    labels: dict[str, pd.Series] = {}
    for horizon in LABEL_HORIZONS:
        labels[f"fwd_ret_close_{horizon}"] = (
            close.shift(-horizon) / close - 1
        ).stack(future_stack=True)
        labels[f"fwd_ret_open_{horizon}"] = (
            open_price.shift(-(horizon + 1)) / open_price.shift(-1) - 1
        ).stack(future_stack=True)
    label_frame = pd.DataFrame(labels).reset_index()
    label_frame.columns = ["trade_date", "stock_code", *labels]
    return daily, label_frame


def _index_frames(
    dates: pd.DatetimeIndex,
    daily: pd.DataFrame,
    codes: list[str],
) -> tuple[pd.DataFrame, ...]:
    """生成基准指数、基础信息和月末成分。"""

    close = daily.groupby("trade_date")["close"].mean().sort_index()
    index_daily = pd.DataFrame(
        {
            "index_code": "000300.SH",
            "trade_date": close.index,
            "open": close.to_numpy(),
            "high": close.to_numpy() * 1.005,
            "low": close.to_numpy() * 0.995,
            "close": close.to_numpy(),
            "pre_close": close.shift(1).fillna(close.iloc[0]).to_numpy(),
            "pct_chg": close.pct_change().fillna(0.0).to_numpy() * 100,
        }
    )
    index_basic = pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "index_name": "合成演示指数",
                "market": "演示市场",
                "publisher": "local",
                "base_date": pd.Timestamp("2020-12-31"),
                "base_point": 1000.0,
                "list_date": pd.Timestamp("2021-01-04"),
            }
        ]
    )
    month_ends = pd.Series(dates, index=dates).groupby(dates.to_period("M")).max()
    index_weight = pd.concat(
        [
            pd.DataFrame(
                {
                    "index_code": "000300.SH",
                    "con_code": codes,
                    "trade_date": pd.Timestamp(day),
                    "weight": 1.0 / len(codes),
                }
            )
            for day in month_ends
        ],
        ignore_index=True,
    )
    industries = pd.DataFrame(
        [
            {
                "stock_code": code,
                "l1_code": f"L{index % 6 + 1}",
                "l1_name": f"演示行业{index % 6 + 1}",
                "classification": "SW2021",
                "in_date": pd.Timestamp("2018-01-02"),
                "out_date": pd.NaT,
                "carry_forward": False,
            }
            for index, code in enumerate(codes)
        ]
    )
    return index_daily, index_basic, index_weight, industries


def build_demo(output: Path, stocks: int, start: str, end: str, seed: int, force: bool) -> dict:
    """写入演示数据库和因子数据，返回摘要。"""

    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"演示目录非空：{output}；需要覆盖时显式传 --force")
        shutil.rmtree(output)
    database_dir = output / "databases"
    factor_dir = output / "factors"
    database_dir.mkdir(parents=True, exist_ok=True)
    factor_dir.mkdir(parents=True, exist_ok=True)
    db_path = database_dir / "quant.duckdb"
    factor_db = database_dir / "factors.duckdb"

    dates = pd.bdate_range(start=start, end=end)
    if len(dates) < 100:
        raise ValueError("演示区间至少需要 100 个工作日")
    codes = _stock_codes(stocks)
    daily, labels = _daily_frame(dates, codes, seed)
    index_daily, index_basic, index_weight, industries = _index_frames(dates, daily, codes)
    stock_basic = pd.DataFrame(
        [
            {
                "stock_code": code,
                "name": f"演示股票{index + 1:03d}",
                "industry": f"演示行业{index % 6 + 1}",
                "market": "主板",
                "exchange": code[:2],
                "list_status": "L",
                "list_date": pd.Timestamp("2018-01-02"),
                "delist_date": pd.NaT,
            }
            for index, code in enumerate(codes)
        ]
    )

    conn = duckdb.connect(str(db_path))
    try:
        for schema in ("md", "ref", "fin", "evt", "meta"):
            conn.execute(f"CREATE SCHEMA {schema}")
        _create_table(conn, "md", "calendar", pd.DataFrame({"trade_date": dates, "is_open": 1}))
        _create_table(conn, "md", "daily_bar", daily.loc[:, DAILY_COLUMNS])
        _create_table(conn, "md", "daily_hfq", daily.loc[:, DAILY_COLUMNS])
        _create_table(conn, "md", "daily_qfq", daily.loc[:, DAILY_COLUMNS])
        _create_table(conn, "md", "fwd_return", labels)
        _create_table(conn, "md", "index_daily", index_daily)
        conn.execute(
            "CREATE TABLE md.suspend_segment "
            "(stock_code VARCHAR, start_date DATE, end_date DATE)"
        )
        _create_table(conn, "ref", "stock_basic", stock_basic)
        _create_table(conn, "ref", "index_basic", index_basic)
        _create_table(conn, "ref", "index_weight", index_weight)
        _create_table(conn, "ref", "sw_industry_member", industries)
        conn.execute(
            "CREATE TABLE ref.corporate_action ("
            "stock_code VARCHAR, ex_date DATE, record_date DATE, pay_date DATE, "
            "cash_div DOUBLE, stock_div DOUBLE, rights_ratio DOUBLE, rights_price DOUBLE)"
        )
        generated_at = datetime.now(timezone.utc).isoformat()
        manifest = pd.DataFrame(
            [
                {
                    "data_version": "synthetic_demo_v1",
                    "generated_at": generated_at,
                    "payload": json.dumps(
                        {
                            "shape": {
                                "n_stocks": len(codes),
                                "n_rows": len(daily),
                                "n_days": len(dates),
                            },
                            "synthetic": True,
                        }
                    ),
                }
            ]
        )
        _create_table(conn, "meta", "manifest", manifest)
        _create_table(
            conn,
            "meta",
            "sw_industry_manifest",
            pd.DataFrame(
                [
                    {
                        "source": "synthetic",
                        "api": "demo",
                        "digest": "demo",
                        "min_date": dates.min(),
                        "max_date": dates.max(),
                        "source_updated_at": generated_at,
                        "built_at": generated_at,
                        "classifications": "SW2021",
                        "switch_date": pd.Timestamp("2021-01-04"),
                        "rows": len(industries),
                        "source_rows": len(industries),
                        "stocks": len(codes),
                        "history_rows": len(industries),
                    }
                ]
            ),
        )
        _create_table(
            conn,
            "meta",
            "corporate_action_manifest",
            pd.DataFrame(
                [
                    {
                        "source": "synthetic",
                        "built_at": generated_at,
                        "rows": 0,
                    }
                ]
            ),
        )
        _create_table(
            conn,
            "meta",
            "financial_manifest",
            pd.DataFrame(columns=["table_name", "rows", "built_at"]),
        )
    finally:
        conn.close()

    from scripts.build_factors.build import build as build_factors

    factor_result = build_factors(
        ["momentum"],
        start,
        end,
        db_file=db_path,
        factor_dir=factor_dir,
        catalog_file=factor_db,
    )
    return {
        "output": str(output),
        "database": str(db_path),
        "factor_database": str(factor_db),
        "stocks": len(codes),
        "trading_days": len(dates),
        "daily_rows": len(daily),
        "factors": factor_result["factor_count"],
    }


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="生成合成 A 股演示数据")
    parser.add_argument("--output", default="data/demo", help="输出目录")
    parser.add_argument("--stocks", type=int, default=120, help="股票数量")
    parser.add_argument("--start", default="2021-01-04", help="起始日")
    parser.add_argument("--end", default="2023-12-29", help="截止日")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--force", action="store_true", help="覆盖已有演示目录")
    args = parser.parse_args(argv)
    result = build_demo(
        Path(args.output),
        args.stocks,
        args.start,
        args.end,
        args.seed,
        args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"使用：QUANT_DATA_DIR={args.output} .venv/bin/python examples/run_demo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

