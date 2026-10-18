"""原始价执行账本。

VectorBT 只适合处理普通买卖，无法表达 A 股现金分红、送转和按持仓批次结算的
红利税。本模块提供与现有执行器兼容的轻量账本：外层仍然使用原来的规则过滤、
目标股数分配和固定点迭代，这里只负责把每日订单和公司行动记成真实现金与持股。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from backtest.costs import CostSpec, FeeCharge, order_cost_frame, round_cent
from backtest.logging_conf import get_logger
from backtest.trading_rules import rule_for

logger = get_logger("backtest.ledger")

ORDER_COLUMNS = [
    "Order Id", "Column", "Timestamp", "Size", "Price", "Fees", "Side",
    "Board", "Order Valid", "reference_price", "gross_amount", "commission",
    "stamp_tax", "transfer_fee", "slippage_rate", "slippage_cost", "total_cost",
    "forced_exit",
]
POSITION_COLUMNS = [
    "Position Id",
    "Column",
    "Size",
    "Entry Timestamp",
    "Avg Entry Price",
    "Entry Fees",
    "Exit Timestamp",
    "Avg Exit Price",
    "Exit Fees",
    "PnL",
    "Return",
    "Direction",
    "Status",
]
ACTION_COLUMNS = [
    "trade_date",
    "stock_code",
    "kind",
    "share_delta",
    "cash_delta",
    "gross_dividend",
    "dividend_tax",
    "tax_rate",
    "fractional_cash",
    "source",
    "source_event_key",
]
WRITEOFF_COLUMNS = [
    "trade_date",
    "stock_code",
    "delist_date",
    "size",
    "price",
    "value",
    "reference_price",
    "reason",
]


@dataclass
class TaxLot:
    """一笔买入形成的税务批次。"""

    stock_code: str
    shares: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_fees: float
    dividend_gross: float = 0.0


@dataclass
class LedgerSnapshot:
    """跨分块交接的账本状态。"""

    cash: float
    shares: pd.Series
    lots: pd.DataFrame
    receivables: pd.DataFrame
    last_prices: pd.Series

    def to_frames(self) -> tuple[pd.Series, pd.DataFrame]:
        """返回可直接交给下一块的持股与批次表。"""
        return self.shares.copy(), self.lots.copy()


def _matrix(
    value: Any,
    index: pd.Index,
    columns: pd.Index,
    default: float = 0.0,
    dtype: str = "float64",
) -> pd.DataFrame:
    """把标量或宽表整理成指定形状。"""
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=index, columns=columns).astype(dtype)
    if isinstance(value, pd.Series):
        return pd.DataFrame(
            np.repeat(value.reindex(index).to_numpy(dtype=dtype)[:, None], len(columns), axis=1),
            index=index,
            columns=columns,
        )
    return pd.DataFrame(float(value), index=index, columns=columns, dtype=dtype)


def _tax_rate(entry: pd.Timestamp, sell: pd.Timestamp) -> float:
    """按历史政策和持有期给出红利税税率。"""
    return CostSpec().dividend_tax_rate(entry, sell)


def _lot_frame(lots: dict[str, list[TaxLot]]) -> pd.DataFrame:
    """把内存批次转成可跨块保存的长表。"""
    rows = []
    for stock, items in lots.items():
        for lot in items:
            if lot.shares <= 1e-9:
                continue
            rows.append(
                {
                    "stock_code": stock,
                    "shares": float(lot.shares),
                    "entry_date": lot.entry_date,
                    "entry_price": float(lot.entry_price),
                    "entry_fees": float(lot.entry_fees),
                    "dividend_gross": float(lot.dividend_gross),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "stock_code",
            "shares",
            "entry_date",
            "entry_price",
            "entry_fees",
            "dividend_gross",
        ],
    )


def _load_lots(frame: pd.DataFrame | None) -> dict[str, list[TaxLot]]:
    """从跨块长表恢复批次。"""
    result: dict[str, list[TaxLot]] = {}
    if frame is None or frame.empty:
        return result
    for row in frame.itertuples(index=False):
        result.setdefault(str(row.stock_code), []).append(
            TaxLot(
                stock_code=str(row.stock_code),
                shares=float(row.shares),
                entry_date=pd.Timestamp(row.entry_date),
                entry_price=float(row.entry_price),
                entry_fees=float(row.entry_fees),
                dividend_gross=float(row.dividend_gross),
            )
        )
    return result


def _receivable_frame(rows: list[dict]) -> pd.DataFrame:
    """整理待收分红。"""
    return pd.DataFrame(
        rows,
        columns=["stock_code", "pay_date", "amount", "source_event_key"],
    )


class LedgerPortfolio:
    """与 VectorBT 结果接口保持一致的原始价账本。"""

    def __init__(
        self,
        index: pd.Index,
        columns: pd.Index,
        cash: pd.Series,
        assets: pd.DataFrame,
        opening_cash: pd.Series,
        opening_assets: pd.DataFrame,
        values: pd.Series,
        receivables: pd.Series,
        orders: pd.DataFrame,
        positions: pd.DataFrame,
        actions: pd.DataFrame,
        lots: dict[str, list[TaxLot]],
        receivable_rows: list[dict],
        last_prices: pd.Series,
        dividend_income: pd.Series,
        dividend_tax: pd.Series,
        dividend_tax_pending: pd.Series,
        fractional_cash: pd.Series,
        writeoff_size: pd.Series,
        writeoff_value: pd.Series,
        valuation: pd.DataFrame,
        sizes: pd.DataFrame,
        writeoff_rows: list[dict],
    ) -> None:
        self.index = pd.DatetimeIndex(index)
        self.columns = columns
        self._cash = cash
        self._assets = assets
        self._opening_cash = opening_cash
        self._opening_assets = opening_assets
        self._values = values
        self._receivables = receivables
        self._orders = orders
        self._positions = positions
        self._actions = actions
        self._lots = lots
        self._receivable_rows = receivable_rows
        self._last_prices = last_prices
        self._dividend_income = dividend_income
        self._dividend_tax = dividend_tax
        self._dividend_tax_pending = dividend_tax_pending
        self._fractional_cash = fractional_cash
        self._writeoff_size = writeoff_size
        self._writeoff_value = writeoff_value
        self._valuation = valuation
        self._sizes = sizes
        self._writeoff_rows = writeoff_rows
        self.orders = SimpleNamespace(records_readable=orders)
        self.positions = SimpleNamespace(records_readable=positions)

    @classmethod
    def from_orders(
        cls,
        close: pd.DataFrame,
        size: pd.DataFrame | float,
        *,
        size_type: str = "amount",
        price: pd.DataFrame | None = None,
        slippage: float = 0.0,
        size_granularity: int = 100,
        init_cash: float = 1_000_000.0,
        cash_sharing: bool = True,
        call_seq: str = "auto",
        corporate_actions: pd.DataFrame | None = None,
        dividend_tax_mode: str = "fifo_deferred",
        costs: CostSpec | None = None,
        boards: pd.Series | None = None,
        allow_board_adjustment: bool = False,
        initial_state: LedgerSnapshot | None = None,
        warmup_rows: int = 0,
        writeoffs: pd.DataFrame | None = None,
        forced_exit: pd.DataFrame | None = None,
        **_: Any,
    ) -> "LedgerPortfolio":
        """执行一次账本回测。

        ``size_type`` 支持 ``targetpercent``、``targetamount`` 和 ``amount``。
        ``targetamount`` 表示目标持股数，``amount`` 表示有符号订单股数。
        ``boards`` 是按股票代码索引的板块序列；默认严格拒绝非法申报。
        ``allow_board_adjustment`` 仅供引擎固定点迭代使用，目标与真实开盘持股不一致时
        先按板块归一化，差异会由引擎的现金裁剪校验发现并触发下一轮。
        ``writeoffs`` 是退市零价值核销候选；``forced_exit`` 标记整理期内应审计为
        强制退出的成交格。
        """
        if not isinstance(close, pd.DataFrame):
            raise TypeError("账本需要日期 × 股票宽表 close")
        index = pd.DatetimeIndex(close.index)
        columns = pd.Index(close.columns)
        if boards is None:
            raise ValueError("账本需要以股票代码为索引的板块序列")
        writeoff_frame = (
            pd.DataFrame(columns=WRITEOFF_COLUMNS)
            if writeoffs is None
            else writeoffs.copy()
        )
        if not writeoff_frame.empty:
            writeoff_frame["trade_date"] = pd.to_datetime(writeoff_frame["trade_date"])
            writeoff_frame["delist_date"] = pd.to_datetime(
                writeoff_frame["delist_date"], errors="coerce"
            )
        forced_exit_panel = (
            pd.DataFrame(False, index=index, columns=columns)
            if forced_exit is None
            else forced_exit.reindex(index=index, columns=columns).fillna(False).astype(bool)
        )
        aligned_boards = boards.reindex(columns)
        missing_boards = aligned_boards.isna() | (
            aligned_boards.astype(str).str.strip() == ""
        )
        if missing_boards.any():
            sample = [str(item) for item in columns[missing_boards.to_numpy()][:5]]
            raise ValueError(f"账本缺少股票板块：{sample}")
        rules = [rule_for(str(item)) for item in aligned_boards]
        valuation = close.astype("float64").ffill()
        order_price = (
            valuation.copy()
            if price is None
            else price.reindex(index=index, columns=columns).astype("float64")
        )
        targets = _matrix(size, index, columns, np.nan)
        cost_model = costs or CostSpec.zero()
        effective_slippage = (
            float(cost_model.slippage) if costs is not None else float(slippage)
        )
        actions = (
            pd.DataFrame(columns=ACTION_COLUMNS)
            if corporate_actions is None
            else corporate_actions.copy()
        )
        if not actions.empty:
            for name in ("ex_date", "record_date", "pay_date"):
                if name in actions:
                    actions[name] = pd.to_datetime(actions[name], errors="coerce").dt.normalize()

        cash = float(init_cash if initial_state is None else initial_state.cash)
        shares = np.zeros(len(columns), dtype="float64")
        lots: dict[str, list[TaxLot]] = {}
        receivable_rows: list[dict] = []
        if initial_state is not None:
            shares = (
                initial_state.shares.reindex(columns).fillna(0.0).to_numpy(dtype="float64")
            )
            lots = _load_lots(initial_state.lots)
            if not initial_state.receivables.empty:
                receivable_rows = initial_state.receivables.to_dict("records")

        cash_values = np.zeros(len(index), dtype="float64")
        opening_cash = np.zeros(len(index), dtype="float64")
        values = np.zeros(len(index), dtype="float64")
        receivable_values = np.zeros(len(index), dtype="float64")
        dividend_income = np.zeros(len(index), dtype="float64")
        dividend_tax = np.zeros(len(index), dtype="float64")
        dividend_pending = np.zeros(len(index), dtype="float64")
        fractional_cash = np.zeros(len(index), dtype="float64")
        writeoff_size = np.zeros(len(index), dtype="float64")
        writeoff_value = np.zeros(len(index), dtype="float64")
        assets = np.zeros((len(index), len(columns)), dtype="float64")
        opening_assets = np.zeros((len(index), len(columns)), dtype="float64")
        order_rows: list[dict] = []
        position_rows: list[dict] = []
        action_rows: list[dict] = []
        writeoff_rows: list[dict] = []
        order_id = 0
        last_prices = pd.Series(np.nan, index=columns, dtype="float64")
        col_lookup = {str(code): position for position, code in enumerate(columns)}

        by_ex: dict[pd.Timestamp, list[dict]] = {}
        for event in actions.to_dict("records") if not actions.empty else []:
            if pd.notna(event.get("ex_date")):
                by_ex.setdefault(pd.Timestamp(event["ex_date"]), []).append(event)
        writeoffs_by_day: dict[pd.Timestamp, list[pd.Series]] = {}
        for item in writeoff_frame.to_dict("records") if not writeoff_frame.empty else []:
            writeoffs_by_day.setdefault(pd.Timestamp(item["trade_date"]), []).append(item)

        def apply_writeoffs(row: int, day: pd.Timestamp) -> None:
            """在当日订单执行后清理退市残仓。"""
            for item in writeoffs_by_day.get(day, []):
                key = str(item["stock_code"])
                position = col_lookup.get(key)
                if position is None:
                    continue
                quantity = float(shares[position])
                if quantity <= 1e-9:
                    continue
                reference = item.get("reference_price", np.nan)
                reference = (
                    float(reference)
                    if pd.notna(reference) and np.isfinite(float(reference))
                    else float(last_prices.get(key, np.nan))
                )
                if not np.isfinite(reference):
                    reference = 0.0
                remove_lots_writeoff(day, key, quantity)
                shares[position] = 0.0
                writeoff_size[row] += quantity
                writeoff_value[row] += quantity * reference
                writeoff_rows.append(
                    {
                        "trade_date": day,
                        "stock_code": key,
                        "delist_date": item.get("delist_date", pd.NaT),
                        "size": quantity,
                        "price": 0.0,
                        "value": 0.0,
                        "reference_price": reference,
                        "reason": "zero_writeoff",
                    }
                )

        def add_action(
            day: pd.Timestamp,
            stock: str,
            kind: str,
            *,
            share_delta: float = 0.0,
            cash_delta: float = 0.0,
            gross_dividend: float = 0.0,
            tax: float = 0.0,
            tax_rate: float = 0.0,
            fractional: float = 0.0,
            source: str = "",
            event_key: str = "",
        ) -> None:
            action_rows.append(
                {
                    "trade_date": day,
                    "stock_code": stock,
                    "kind": kind,
                    "share_delta": float(share_delta),
                    "cash_delta": float(cash_delta),
                    "gross_dividend": float(gross_dividend),
                    "dividend_tax": float(tax),
                    "tax_rate": float(tax_rate),
                    "fractional_cash": float(fractional),
                    "source": source,
                    "source_event_key": event_key,
                }
            )

        def receive_due(day: pd.Timestamp) -> float:
            """把到账日不晚于当天的分红从应收转为现金。"""
            nonlocal cash, receivable_rows
            received = 0.0
            kept = []
            for item in receivable_rows:
                pay_date = pd.Timestamp(item["pay_date"]).normalize()
                if pay_date <= day:
                    amount = float(item["amount"])
                    cash += amount
                    received += amount
                    dividend_income[index.get_loc(day)] += amount
                    add_action(
                        day,
                        str(item["stock_code"]),
                        "dividend_payment",
                        cash_delta=amount,
                        source="dividend",
                        event_key=str(item.get("source_event_key", "")),
                    )
                else:
                    kept.append(item)
            receivable_rows = kept
            return received

        def apply_share_action(day: pd.Timestamp, event: dict, price_value: float) -> None:
            """按除权日开盘前调整持股和批次。"""
            nonlocal cash, shares
            key = str(event["stock_code"])
            if key not in col_lookup:
                return
            position = col_lookup[key]
            old = float(shares[position])
            if old <= 0:
                return
            multiplier = float(event["share_multiplier"])
            exact = old * multiplier
            new = float(np.floor(exact + 1e-9))
            delta = new - old
            fractional = max(exact - new, 0.0)
            cash_in_lieu = fractional * max(price_value, 0.0)
            if cash_in_lieu:
                cash += cash_in_lieu
                fractional_cash[index.get_loc(day)] += cash_in_lieu
            shares[position] = new
            if abs(delta) >= 1e-9:
                items = lots.get(key, [])
                old_total = sum(item.shares for item in items)
                if old_total > 0:
                    if delta > 0:
                        for item in items:
                            item.shares += delta * item.shares / old_total
                    else:
                        remaining = -delta
                        for item in items:
                            take = min(item.shares, remaining)
                            item.shares -= take
                            remaining -= take
                            if remaining <= 1e-9:
                                break
            add_action(
                day,
                key,
                "share_adjustment",
                share_delta=delta,
                cash_delta=cash_in_lieu,
                fractional=fractional,
                source=str(event.get("source", "")),
                event_key=str(event.get("source_event_key", "")),
            )

        def accrue_dividend(day: pd.Timestamp, event: dict) -> None:
            """登记日收盘按持股确认税前分红。"""
            key = str(event["stock_code"])
            if key not in col_lookup:
                return
            position = col_lookup[key]
            held = float(shares[position])
            gross = held * float(event["cash_div_per_share"])
            if gross <= 1e-12:
                return
            payable = pd.Timestamp(event.get("pay_date") or event["ex_date"]).normalize()
            receivable_rows.append(
                {
                    "stock_code": key,
                    "pay_date": payable,
                    "amount": gross,
                    "source_event_key": str(event.get("source_event_key", "")),
                }
            )
            items = lots.get(key, [])
            total = sum(item.shares for item in items)
            if total > 0:
                for item in items:
                    item.dividend_gross += gross * item.shares / total
            add_action(
                day,
                key,
                "dividend_accrual",
                gross_dividend=gross,
                source="dividend",
                event_key=str(event.get("source_event_key", "")),
            )

        def consume_lots(
            day: pd.Timestamp,
            key: str,
            quantity: float,
            exit_price: float,
            exit_fees: float,
        ) -> tuple[float, float]:
            """按 FIFO 消耗批次，返回本笔卖出的红利税和对应税前分红。"""
            items = lots.get(key, [])
            remaining = quantity
            tax = 0.0
            gross = 0.0
            while remaining > 1e-9 and items:
                item = items[0]
                take = min(item.shares, remaining)
                ratio = take / item.shares if item.shares else 0.0
                if dividend_tax_mode == "fifo_deferred":
                    dividend = item.dividend_gross * ratio
                    gross += dividend
                    tax += dividend * cost_model.dividend_tax_rate(item.entry_date, day)
                entry_fees = item.entry_fees * ratio
                entry_price = item.entry_price
                pnl = take * (exit_price - entry_price) - entry_fees - exit_fees * take / quantity
                position_rows.append(
                    {
                        "Position Id": len(position_rows),
                        "Column": key,
                        "Size": take,
                        "Entry Timestamp": item.entry_date,
                        "Avg Entry Price": entry_price,
                        "Entry Fees": entry_fees,
                        "Exit Timestamp": day,
                        "Avg Exit Price": exit_price,
                        "Exit Fees": exit_fees * take / quantity,
                        "PnL": pnl,
                        "Return": pnl / (take * entry_price) if entry_price else 0.0,
                        "Direction": "Long",
                        "Status": "Closed",
                    }
                )
                item.dividend_gross *= 1.0 - ratio
                item.entry_fees -= entry_fees
                item.shares -= take
                remaining -= take
                if item.shares <= 1e-9:
                    items.pop(0)
            return tax, gross

        def remove_lots_writeoff(day: pd.Timestamp, key: str, quantity: float) -> None:
            """按零价值关闭持仓批次；核销不是卖出，不结算红利税。"""
            items = lots.get(key, [])
            remaining = float(quantity)
            while remaining > 1e-9 and items:
                item = items[0]
                take = min(item.shares, remaining)
                ratio = take / item.shares if item.shares else 0.0
                entry_fees = item.entry_fees * ratio
                pnl = -take * item.entry_price - entry_fees
                position_rows.append(
                    {
                        "Position Id": len(position_rows),
                        "Column": key,
                        "Size": take,
                        "Entry Timestamp": item.entry_date,
                        "Avg Entry Price": item.entry_price,
                        "Entry Fees": entry_fees,
                        "Exit Timestamp": day,
                        "Avg Exit Price": 0.0,
                        "Exit Fees": 0.0,
                        "PnL": pnl,
                        "Return": -1.0 if item.entry_price else 0.0,
                        "Direction": "Long",
                        "Status": "Closed",
                    }
                )
                item.entry_fees -= entry_fees
                item.shares -= take
                remaining -= take
                if item.shares <= 1e-9:
                    items.pop(0)

        def order_charge(
            day: pd.Timestamp,
            key: str,
            board: str,
            side: str,
            quantity: float,
            base_price: float,
        ) -> FeeCharge:
            """按账本成交口径计算一笔订单的成本。"""
            return cost_model.charge(day, side, quantity, base_price, key, board)

        def buy_total_cost(quantity: float, charge: FeeCharge) -> float:
            """买入占用的现金，包含成交额和现金费用。"""
            return charge.fill_amount + charge.fees

        start_row = min(max(int(warmup_rows), 0), len(index))
        for row, day in enumerate(index):
            # 预热行只提供上一块的价格口径，不重复执行已经发生的公司行动。
            if row >= start_row:
                receive_due(day)
                day_price = (
                    float(order_price.iloc[row].dropna().median())
                    if order_price.iloc[row].notna().any()
                    else 0.0
                )
                for event in by_ex.get(day, []):
                    # 分红按除权日前持股确认，和价格在除权日的下跌同一天进入组合价值。
                    accrue_dividend(day, event)
                    event_code = str(event["stock_code"])
                    event_position = col_lookup.get(event_code)
                    if event_position is None:
                        continue
                    event_price = order_price.iloc[row, event_position]
                    apply_share_action(
                        day,
                        event,
                        float(event_price) if pd.notna(event_price) else day_price,
                    )
                receive_due(day)
            opening_cash[row] = cash
            opening_assets[row] = shares
            if row < start_row:
                assets[row] = shares
                cash_values[row] = cash
                receivable_values[row] = sum(float(item["amount"]) for item in receivable_rows)
                values[row] = cash + receivable_values[row] + float(
                    np.nansum(shares * valuation.iloc[row].to_numpy(dtype="float64"))
                )
                mark_row = valuation.iloc[row].to_numpy(dtype="float64")
                last_prices = pd.Series(
                    np.where(np.isfinite(mark_row), mark_row, last_prices.to_numpy()),
                    index=columns,
                )
                continue

            open_row = order_price.iloc[row].to_numpy(dtype="float64")
            weights = targets.iloc[row].to_numpy(dtype="float64")
            active = np.flatnonzero(np.isfinite(weights))
            if not len(active):
                apply_writeoffs(row, day)
                assets[row] = shares
                cash_values[row] = cash
                receivable_values[row] = sum(float(item["amount"]) for item in receivable_rows)
                values[row] = cash + receivable_values[row] + float(
                    np.nansum(shares * valuation.iloc[row].to_numpy(dtype="float64"))
                )
                mark_row = valuation.iloc[row].to_numpy(dtype="float64")
                last_prices = pd.Series(
                    np.where(np.isfinite(mark_row), mark_row, last_prices.to_numpy()),
                    index=columns,
                )
                continue

            market_value = float(np.nansum(shares * open_row))
            total_value = cash + market_value
            desired = shares.copy()
            if size_type == "targetamount":
                desired[active] = np.nan_to_num(weights[active], nan=0.0)
            elif size_type == "targetpercent":
                for position in active:
                    if not np.isfinite(open_row[position]) or open_row[position] <= 0:
                        continue
                    value = max(total_value, 0.0) * float(weights[position])
                    desired[position] = rules[position].normalize_target(
                        shares[position], value / open_row[position]
                    )
            else:
                desired[active] = shares[active] + np.nan_to_num(weights[active], nan=0.0)

            delta = desired - shares
            order_cells = [
                position
                for position in active
                if abs(delta[position]) >= 0.5
                and np.isfinite(open_row[position])
                and open_row[position] > 0
            ]
            # 卖出先于买入，保证卖出所得可用于同一天的买入。
            order_cells.sort(key=lambda position: (delta[position] > 0, delta[position]))
            for position in order_cells:
                key = str(columns[position])
                quantity = delta[position]
                base_price = open_row[position]
                rule = rules[position]
                if (
                    allow_board_adjustment
                    and size_type == "targetamount"
                    and not rule.is_valid_order(shares[position], quantity)
                ):
                    # 固定点第一轮可能用上一轮开盘状态算出目标；按真实开盘持股归一化后，
                    # 目标股数与计划不一致会由 _clipped_cells 记下，下一轮重新分配。
                    desired[position] = rule.normalize_target(shares[position], desired[position])
                    quantity = desired[position] - shares[position]
                    if abs(quantity) < 0.5:
                        continue
                if not rule.is_valid_order(shares[position], quantity):
                    raise ValueError(
                        f"非法板块订单：{day.date()} {key} {rule.board} "
                        f"持仓 {shares[position]:.0f} 股，申报 {quantity:.0f} 股"
                    )
                if quantity < 0:
                    sell_quantity = abs(float(quantity))
                    if not np.isfinite(sell_quantity) or sell_quantity <= 0.0:
                        continue
                    charge = order_charge(
                        day, key, rule.board, "Sell", sell_quantity, base_price
                    )
                    fill_price = charge.fill_price
                    cash += charge.fill_amount - charge.fees
                    shares[position] -= sell_quantity
                    raw_tax, gross_dividend = consume_lots(
                        day, key, sell_quantity, fill_price, charge.fees
                    )
                    tax = round_cent(raw_tax) if dividend_tax_mode == "fifo_deferred" else 0.0
                    if tax > 1e-12:
                        cash -= tax
                        dividend_tax[row] += tax
                        add_action(
                            day, key, "dividend_tax", cash_delta=-tax, tax=tax,
                            gross_dividend=gross_dividend,
                            tax_rate=tax / gross_dividend if gross_dividend > 1e-12 else 0.0,
                            source="fifo", event_key="",
                        )
                    order_rows.append(
                        {
                            "Order Id": order_id,
                            "Column": key,
                            "Timestamp": day,
                            "Size": -sell_quantity,
                            "Price": fill_price,
                            "Fees": charge.fees,
                            "Side": "Sell",
                            "Board": rule.board,
                            "Order Valid": True,
                            "reference_price": charge.reference_price,
                            "gross_amount": charge.gross_amount,
                            "commission": charge.commission,
                            "stamp_tax": charge.stamp_tax,
                            "transfer_fee": charge.transfer_fee,
                            "slippage_rate": charge.slippage_rate,
                            "slippage_cost": charge.slippage_cost,
                            "total_cost": charge.total_cost,
                            "forced_exit": bool(forced_exit_panel.iat[row, position]),
                        }
                    )
                    order_id += 1
                else:
                    # 目标分配通常已经留好现金；这里仍按现金约束截断，防止坏配置穿透账本。
                    affordable = quantity
                    charge = order_charge(
                        day, key, rule.board, "Buy", affordable, base_price
                    )
                    if quantity > 0 and buy_total_cost(affordable, charge) > cash + 1e-9:
                        max_order = rule.floor_buy_quantity(
                            max(cash, 0.0)
                            / (
                                base_price
                                * (1.0 + effective_slippage)
                                * (1.0 + float(cost_model.commission_rate))
                            )
                        )
                        affordable = float(min(quantity, max_order))
                        while affordable > 0:
                            charge = order_charge(
                                day, key, rule.board, "Buy", affordable, base_price
                            )
                            if buy_total_cost(affordable, charge) <= cash + 1e-9:
                                break
                            affordable = float(rule.previous_buy_target(0.0, affordable))
                        while affordable > 0:
                            next_order = float(rule.previous_buy_target(0.0, affordable))
                            step = affordable - next_order
                            candidate = affordable + step
                            candidate_charge = order_charge(
                                day, key, rule.board, "Buy", candidate, base_price
                            )
                            if candidate > quantity or buy_total_cost(
                                candidate, candidate_charge
                            ) > cash + 1e-9:
                                break
                            affordable = candidate
                        affordable = min(affordable, quantity)
                    if affordable <= 0:
                        continue
                    if not rule.is_valid_buy(affordable):
                        raise ValueError(
                            f"现金裁剪生成了非法板块订单：{day.date()} {key} "
                            f"{rule.board} {affordable:.0f} 股"
                        )
                    charge = order_charge(
                        day, key, rule.board, "Buy", affordable, base_price
                    )
                    fill_price = charge.fill_price
                    cash -= charge.fill_amount + charge.fees
                    shares[position] += affordable
                    lots.setdefault(key, []).append(
                        TaxLot(
                            stock_code=key,
                            shares=affordable,
                            entry_date=day,
                            entry_price=fill_price,
                            entry_fees=charge.fees,
                        )
                    )
                    order_rows.append(
                        {
                            "Order Id": order_id,
                            "Column": key,
                            "Timestamp": day,
                            "Size": affordable,
                            "Price": fill_price,
                            "Fees": charge.fees,
                            "Side": "Buy",
                            "Board": rule.board,
                            "Order Valid": True,
                            "reference_price": charge.reference_price,
                            "gross_amount": charge.gross_amount,
                            "commission": charge.commission,
                            "stamp_tax": charge.stamp_tax,
                            "transfer_fee": charge.transfer_fee,
                            "slippage_rate": charge.slippage_rate,
                            "slippage_cost": charge.slippage_cost,
                            "total_cost": charge.total_cost,
                            "forced_exit": False,
                        }
                    )
                    order_id += 1

            apply_writeoffs(row, day)
            assets[row] = shares
            cash_values[row] = cash
            receivable_values[row] = sum(float(item["amount"]) for item in receivable_rows)
            mark_row = valuation.iloc[row].to_numpy(dtype="float64")
            values[row] = cash + receivable_values[row] + float(np.nansum(shares * mark_row))
            last_prices = pd.Series(
                np.where(np.isfinite(mark_row), mark_row, last_prices.to_numpy()),
                index=columns,
            )

        # 区间末仍未卖出的批次保留为 Open 持仓。
        for key, items in lots.items():
            last = last_prices.get(key, np.nan)
            for item in items:
                if item.shares <= 1e-9:
                    continue
                exit_price = float(last) if np.isfinite(last) else item.entry_price
                pnl = item.shares * (exit_price - item.entry_price) - item.entry_fees
                position_rows.append(
                    {
                        "Position Id": len(position_rows),
                        "Column": key,
                        "Size": item.shares,
                        "Entry Timestamp": item.entry_date,
                        "Avg Entry Price": item.entry_price,
                        "Entry Fees": item.entry_fees,
                        "Exit Timestamp": pd.NaT,
                        "Avg Exit Price": exit_price,
                        "Exit Fees": 0.0,
                        "PnL": pnl,
                        "Return": pnl / (item.shares * item.entry_price) if item.entry_price else 0.0,
                        "Direction": "Long",
                        "Status": "Open",
                    }
                )
                if dividend_tax_mode == "fifo_deferred":
                    dividend_pending[-1] += item.dividend_gross * cost_model.dividend_tax_rate(
                        item.entry_date, index[-1]
                    )
            if dividend_tax_mode == "fifo_deferred":
                dividend_pending[-1] = round_cent(dividend_pending[-1])

        orders = pd.DataFrame(order_rows, columns=ORDER_COLUMNS)
        positions = pd.DataFrame(position_rows, columns=POSITION_COLUMNS)
        actions_frame = pd.DataFrame(action_rows, columns=ACTION_COLUMNS)
        cash_series = pd.Series(cash_values, index=index, name="cash")
        opening_cash_series = pd.Series(opening_cash, index=index, name="opening_cash")
        values_series = pd.Series(values, index=index, name="portfolio_value")
        receivable_series = pd.Series(receivable_values, index=index, name="dividend_receivable")
        dividend_income_series = pd.Series(dividend_income, index=index)
        dividend_tax_series = pd.Series(dividend_tax, index=index)
        pending_series = pd.Series(dividend_pending, index=index)
        fractional_series = pd.Series(fractional_cash, index=index)
        writeoff_size_series = pd.Series(writeoff_size, index=index)
        writeoff_value_series = pd.Series(writeoff_value, index=index)
        assets_frame = pd.DataFrame(assets, index=index, columns=columns)
        opening_assets_frame = pd.DataFrame(opening_assets, index=index, columns=columns)
        return cls(
            index=index,
            columns=columns,
            cash=cash_series,
            assets=assets_frame,
            opening_cash=opening_cash_series,
            opening_assets=opening_assets_frame,
            values=values_series,
            receivables=receivable_series,
            orders=orders,
            positions=positions,
            actions=actions_frame,
            lots=lots,
            receivable_rows=receivable_rows,
            last_prices=last_prices,
            dividend_income=dividend_income_series,
            dividend_tax=dividend_tax_series,
            dividend_tax_pending=pending_series,
            fractional_cash=fractional_series,
            writeoff_size=writeoff_size_series,
            writeoff_value=writeoff_value_series,
            valuation=valuation,
            sizes=targets,
            writeoff_rows=writeoff_rows,
        )

    def value(self) -> pd.Series:
        """逐日组合价值，包含应收分红。"""
        return self._values.copy()

    def cash(self) -> pd.Series:
        """逐日收盘现金。"""
        return self._cash.copy()

    def assets(self) -> pd.DataFrame:
        """逐日收盘持股数。"""
        return self._assets.copy()

    def opening_cash(self) -> pd.Series:
        """逐日开盘可用现金。"""
        return self._opening_cash.copy()

    def opening_assets(self) -> pd.DataFrame:
        """逐日开盘前持股数，已经包含当日公司行动。"""
        return self._opening_assets.copy()

    def asset_value(self, group_by: bool = False) -> pd.DataFrame:
        """逐日逐股持仓市值。"""
        return self._assets * self._valuation.reindex_like(self._assets)

    def receivables(self) -> pd.Series:
        """逐日尚未到账的分红。"""
        return self._receivables.copy()

    def dividend_income(self) -> pd.Series:
        """逐日到账的税前分红。"""
        return self._dividend_income.copy()

    def dividend_tax(self) -> pd.Series:
        """逐日结算的红利税。"""
        return self._dividend_tax.copy()

    def dividend_tax_pending(self) -> pd.Series:
        """逐日区间末仍未卖出的批次对应红利税估计。"""
        return self._dividend_tax_pending.copy()

    def fractional_cash(self) -> pd.Series:
        """逐日零头折现金额。"""
        return self._fractional_cash.copy()

    def writeoff_records(self) -> pd.DataFrame:
        """返回逐笔退市零价值核销明细。"""
        return pd.DataFrame(self._writeoff_rows, columns=WRITEOFF_COLUMNS)

    def corporate_action_records(self) -> pd.DataFrame:
        """返回公司行动明细。"""
        frame = self._actions.copy()
        if "event_key" in frame:
            frame = frame.rename(columns={"event_key": "source_event_key"})
        return frame.reindex(columns=ACTION_COLUMNS)

    def snapshot(self) -> LedgerSnapshot:
        """给出跨块交接状态。"""
        return LedgerSnapshot(
            cash=float(self._cash.iloc[-1]) if len(self._cash) else 0.0,
            shares=self._assets.iloc[-1].astype("float64"),
            lots=_lot_frame(self._lots),
            receivables=_receivable_frame(self._receivable_rows),
            last_prices=self._last_prices.astype("float64"),
        )

    def daily(self) -> pd.DataFrame:
        """返回账本层面的日结果。"""
        holdings_value = (self._assets * self._valuation.reindex_like(self._assets)).sum(axis=1)
        frame = pd.DataFrame(
            {
                "portfolio_value": self._values,
                "cash": self._cash,
                "holdings_value": holdings_value,
                "dividend_income": self._dividend_income,
                "dividend_tax": self._dividend_tax,
                "dividend_receivable": self._receivables,
                "fractional_cash": self._fractional_cash,
                "writeoff_size": self._writeoff_size,
                "writeoff_value": self._writeoff_value,
            },
            index=self.index,
        )
        frame["return"] = frame["portfolio_value"].pct_change().fillna(0.0)
        frame["nav"] = frame["portfolio_value"] / float(frame["portfolio_value"].iloc[0])
        frame["drawdown"] = frame["nav"] / frame["nav"].cummax() - 1.0
        costed = order_cost_frame(self._orders, frame.index)
        for name in costed.columns:
            frame[name] = costed[name]
        frame["cost"] = (
            frame["commission"]
            + frame["stamp_tax"]
            + frame["transfer_fee"]
            + self._dividend_tax
        )
        frame["total_cost"] = frame["cost"] + frame["slippage_cost"]
        if not self._orders.empty:
            stamps = pd.DatetimeIndex(self._orders["Timestamp"])
            traded = (
                self._orders["Size"].abs() * self._orders["Price"].astype(float)
            ).groupby(stamps).sum()
            frame["turnover"] = (
                traded.reindex(frame.index).fillna(0.0)
                / frame["portfolio_value"].shift(1).replace(0, np.nan)
            )
        else:
            frame["turnover"] = 0.0
        return frame
