"""A 股交易成本模型。

资金分配、成交账本和结果核对都从这里取费率，避免同一笔订单在不同模块按不同
口径计费。费率表只记录公开的生效日期，实际计费按成交日、买卖方向和交易所解析。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any

import numpy as np
import pandas as pd

from backtest.errors import ExecutionError, SpecError
from backtest.trading_rules import BOARD_BSE, BOARD_CHINEXT, BOARD_MAIN, BOARD_STAR


COST_MODEL_VERSION = "cn_a_share_v1"
PAR_VALUE_CNY = 1.0


@dataclass(frozen=True)
class StampTaxTier:
    """印花税的一个生效区间。费率使用小数表示。"""

    effective_from: str
    buy_rate: float
    sell_rate: float


@dataclass(frozen=True)
class TransferFeeTier:
    """过户费的一个生效区间。

    ``basis`` 为 ``amount`` 时按成交金额收取，为 ``par_value`` 时按成交面额收取。
    ``markets`` 取 ``SH``、``SZ``、``BJ``。
    """

    effective_from: str
    rate: float
    basis: str
    markets: tuple[str, ...]
    minimum: float = 0.0


@dataclass(frozen=True)
class DividendTaxTier:
    """红利税政策区间；``flat`` 表示不区分持有期。"""

    effective_from: str
    short_rate: float
    medium_rate: float
    long_rate: float
    flat: bool = False


STAMP_TAX_SCHEDULE = (
    StampTaxTier("1998-06-12", 0.004, 0.004),
    StampTaxTier("2001-11-16", 0.002, 0.002),
    StampTaxTier("2005-01-24", 0.001, 0.001),
    StampTaxTier("2007-05-30", 0.003, 0.003),
    StampTaxTier("2008-04-24", 0.001, 0.001),
    StampTaxTier("2008-09-19", 0.0, 0.001),
    StampTaxTier("2023-08-28", 0.0, 0.0005),
)

TRANSFER_FEE_SCHEDULE = (
    # 2015-08-01 前沪市按成交面额、深市按成交金额；沪市面额口径最低 1 元。
    TransferFeeTier("2000-01-01", 0.0005, "par_value", ("SH",), 1.0),
    TransferFeeTier("2000-01-01", 0.0000255, "amount", ("SZ",), 0.0),
    TransferFeeTier("2012-06-01", 0.000375, "par_value", ("SH",), 1.0),
    TransferFeeTier("2012-09-01", 0.0003, "par_value", ("SH",), 1.0),
    TransferFeeTier("2015-08-01", 0.00002, "amount", ("SH", "SZ"), 0.0),
    TransferFeeTier("2021-11-15", 0.000025, "amount", ("BJ",), 0.0),
    TransferFeeTier("2022-04-29", 0.00001, "amount", ("SH", "SZ", "BJ"), 0.0),
)

DIVIDEND_TAX_SCHEDULE = (
    DividendTaxTier("2000-01-01", 0.20, 0.20, 0.20, flat=True),
    DividendTaxTier("2005-06-13", 0.10, 0.10, 0.10, flat=True),
    DividendTaxTier("2013-01-01", 0.20, 0.10, 0.05),
    DividendTaxTier("2015-09-08", 0.20, 0.10, 0.0),
)


def _tier_date(value: str) -> pd.Timestamp:
    """把生效日期解析成规范化的时间戳。"""
    return pd.Timestamp(value).normalize()


def _validate_schedule(items, label: str) -> None:
    """校验费率表按日期递增且费率非负。"""
    dates = [_tier_date(item.effective_from) for item in items]
    if dates != sorted(dates):
        raise SpecError(f"{label} 必须按生效日期递增")
    if not hasattr(items[0], "markets") and len(set(dates)) != len(dates):
        raise SpecError(f"{label} 生效日期不能重复")
    for item in items:
        values = [
            getattr(item, name)
            for name in ("buy_rate", "sell_rate", "rate", "short_rate", "medium_rate", "long_rate", "minimum")
            if hasattr(item, name)
        ]
        if any(float(value) < 0 for value in values):
            raise SpecError(f"{label} 费率不能为负数")
        if hasattr(item, "basis") and item.basis not in ("amount", "par_value"):
            raise SpecError(f"{label} basis 只支持 amount/par_value")


_validate_schedule(STAMP_TAX_SCHEDULE, "印花税费率表")
_validate_schedule(TRANSFER_FEE_SCHEDULE, "过户费费率表")
_validate_schedule(DIVIDEND_TAX_SCHEDULE, "红利税费率表")


def _pick_tier(items, day: pd.Timestamp, label: str):
    """返回不晚于 ``day`` 的最后一档费率。"""
    stamp = pd.Timestamp(day).normalize()
    picked = None
    for item in items:
        if _tier_date(item.effective_from) <= stamp:
            picked = item
        else:
            break
    if picked is None:
        raise ExecutionError(f"{label} 没有覆盖交易日期 {stamp.date()}")
    return picked


@lru_cache(maxsize=8192)
def _stamp_rates(day: pd.Timestamp) -> tuple[float, float]:
    """按成交日返回印花税买入、卖出税率。"""
    tier = _pick_tier(STAMP_TAX_SCHEDULE, day, "印花税费率表")
    return float(tier.buy_rate), float(tier.sell_rate)


@lru_cache(maxsize=8192)
def _transfer_terms(day: pd.Timestamp, exchange: str) -> tuple[float, str, float] | None:
    """按成交日和交易所返回过户费费率、口径和最低收费。"""
    stamp = pd.Timestamp(day).normalize()
    picked = None
    for item in TRANSFER_FEE_SCHEDULE:
        if _tier_date(item.effective_from) <= stamp and exchange in item.markets:
            picked = item
    if picked is None:
        raise ExecutionError(
            f"过户费费率表没有覆盖 {exchange} 市场在 {stamp.date()} 的交易"
        )
    return float(picked.rate), picked.basis, float(picked.minimum)


@lru_cache(maxsize=8192)
def _dividend_tier(day: pd.Timestamp) -> DividendTaxTier:
    """按卖出日返回红利税政策。"""
    return _pick_tier(DIVIDEND_TAX_SCHEDULE, day, "红利税费率表")


def round_cent(value: float) -> float:
    """按分四舍五入（half-up），用于单笔费用。"""
    number = float(value)
    if not math.isfinite(number):
        raise ExecutionError(f"费用不是有限数：{value!r}")
    return math.floor(number * 100.0 + 0.5 + 1e-9) / 100.0


def round_cent_array(values: np.ndarray) -> np.ndarray:
    """按分四舍五入（half-up），用于向量化费用。"""
    array = np.asarray(values, dtype="float64")
    if not np.isfinite(array).all():
        raise ExecutionError("费用矩阵包含非有限数")
    return np.floor(array * 100.0 + 0.5 + 1e-9) / 100.0


def exchange_for(stock_code: str, board: str | None = None) -> str:
    """从标准股票代码解析交易所；没有前缀时按板块兜底。"""
    code = str(stock_code).upper()
    if code.startswith("SH"):
        return "SH"
    if code.startswith("SZ"):
        return "SZ"
    if code.startswith("BJ"):
        return "BJ"
    if board == BOARD_BSE:
        return "BJ"
    if board in (BOARD_MAIN, BOARD_STAR):
        return "SH"
    if board == BOARD_CHINEXT:
        return "SZ"
    raise ExecutionError(f"无法解析股票 {stock_code!r} 的交易所（板块 {board!r}）")


@dataclass(frozen=True)
class FeeCharge:
    """一笔成交的成本分项。"""

    reference_price: float
    fill_price: float
    gross_amount: float
    fill_amount: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    fees: float
    slippage_rate: float
    slippage_cost: float
    impact_rate: float
    impact_cost: float
    total_cost: float


@dataclass(frozen=True)
class ChargeArrays:
    """日期 × 股票单元格的成本分项矩阵。"""

    reference_price: np.ndarray
    fill_price: np.ndarray
    gross_amount: np.ndarray
    fill_amount: np.ndarray
    commission: np.ndarray
    stamp_tax: np.ndarray
    transfer_fee: np.ndarray
    fees: np.ndarray
    slippage_rate: float
    slippage_cost: np.ndarray
    impact_rate: np.ndarray
    impact_cost: np.ndarray
    total_cost: np.ndarray


@dataclass(frozen=True)
class CostSpec:
    """A 股交易成本参数。

    ``stamp_tax_rate`` 为 ``None`` 时按历史分段；旧策略 JSON 里的固定值继续作为
    仅卖出固定税率，保证旧配置可以重放。费率均为小数。

    底层 ``CostSpec`` 默认不启用冲击成本，便于旧口径和低层调用保持原样；
    新建 ``StrategySpec`` 会显式使用平方根冲击模型。
    """

    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_tax_rate: float | None = None
    slippage: float = 0.0005
    impact_model: str = "none"
    impact_coefficient: float = 0.5
    impact_volatility_window: int = 20
    impact_volatility_floor: float = 0.0
    impact_max_rate: float = 0.05
    dividend_tax_mode: str = "fifo_deferred"
    transfer_fee_enabled: bool = True
    fee_rounding: str = "cent"

    def __post_init__(self) -> None:
        values = (self.commission_rate, self.minimum_commission, self.slippage)
        if any(float(value) < 0 for value in values):
            raise SpecError("交易成本不能为负数")
        if self.stamp_tax_rate is not None and float(self.stamp_tax_rate) < 0:
            raise SpecError("交易成本不能为负数")
        if self.dividend_tax_mode not in ("fifo_deferred", "disabled"):
            raise SpecError("costs.dividend_tax_mode 只支持 fifo_deferred/disabled")
        if self.impact_model not in ("none", "sqrt"):
            raise SpecError("costs.impact_model 只支持 none/sqrt")
        for name in (
            "impact_coefficient",
            "impact_volatility_floor",
            "impact_max_rate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SpecError(f"costs.{name} 必须是数字")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise SpecError(f"costs.{name} 必须是非负有限数")
        if type(self.impact_volatility_window) is not int or self.impact_volatility_window <= 0:
            raise SpecError("costs.impact_volatility_window 必须是正整数")
        if float(self.impact_volatility_floor) >= 1.0:
            raise SpecError("costs.impact_volatility_floor 必须在 [0, 1) 之间")
        if float(self.impact_max_rate) > 1.0:
            raise SpecError("costs.impact_max_rate 必须在 [0, 1] 之间")
        if not isinstance(self.transfer_fee_enabled, bool):
            raise SpecError("costs.transfer_fee_enabled 必须是布尔值")
        if self.fee_rounding not in ("cent", "none"):
            raise SpecError("costs.fee_rounding 只支持 cent/none")

    @classmethod
    def zero(cls) -> "CostSpec":
        """返回零成本配置，供测试、探针和关闭成本的调用方使用。"""
        return cls(
            commission_rate=0.0,
            minimum_commission=0.0,
            stamp_tax_rate=0.0,
            slippage=0.0,
            impact_model="none",
            impact_coefficient=0.0,
            transfer_fee_enabled=False,
            fee_rounding="none",
        )

    def stamp_rates(self, day) -> tuple[float, float]:
        """返回买入、卖出印花税率。"""
        if self.stamp_tax_rate is not None:
            return 0.0, float(self.stamp_tax_rate)
        return _stamp_rates(pd.Timestamp(day).normalize())

    def transfer_terms(self, day, exchange: str) -> tuple[float, str, float] | None:
        """返回过户费费率、口径和最低收费；关闭时返回 None。"""
        if not self.transfer_fee_enabled:
            return None
        return _transfer_terms(pd.Timestamp(day).normalize(), str(exchange))

    def dividend_tax_rate(self, entry_date, sell_date) -> float:
        """按卖出日政策和持有期给出红利税税率。"""
        if self.dividend_tax_mode == "disabled":
            return 0.0
        entry = pd.Timestamp(entry_date).normalize()
        sell = pd.Timestamp(sell_date).normalize()
        tier = _dividend_tier(sell)
        if tier.flat:
            return float(tier.short_rate)
        if sell <= entry + pd.DateOffset(months=1):
            return float(tier.short_rate)
        if sell <= entry + pd.DateOffset(years=1):
            return float(tier.medium_rate)
        return float(tier.long_rate)

    def _round(self, values: np.ndarray) -> np.ndarray:
        """按配置对费用矩阵取整。"""
        return round_cent_array(values) if self.fee_rounding == "cent" else values

    def charge_arrays(
        self,
        day,
        side: np.ndarray,
        shares: np.ndarray,
        reference_price: np.ndarray,
        stock_codes,
        boards,
        exchanges=None,
        amount=None,
        volatility=None,
    ) -> ChargeArrays:
        """向量化计算一组成交的费用。

        参数:
            day: 成交日。
            side: 有符号方向，1 买入、-1 卖出、0 无成交。
            shares: 成交股数，取绝对值使用。
            reference_price: 未含滑点的执行价。
            stock_codes: 与成交数组同形状的股票代码。
            boards: 与成交数组同形状的板块名称。
            exchanges: 可选的预解析交易所矩阵，避免逐次重复解析代码前缀。
            amount: 可选的当日成交额矩阵，用于平方根冲击模型。
            volatility: 可选的执行日前 20 日复权收益波动率矩阵。
        """
        directions = np.asarray(side, dtype="int8")
        quantities = np.abs(np.asarray(shares, dtype="float64"))
        reference = np.asarray(reference_price, dtype="float64")
        codes = np.asarray(stock_codes, dtype=object)
        board_values = np.asarray(boards, dtype=object)
        if not (
            directions.shape == quantities.shape == reference.shape == codes.shape == board_values.shape
        ):
            raise ExecutionError("成本计算输入形状不一致")
        amount_values = (
            np.full(directions.shape, np.nan, dtype="float64")
            if amount is None
            else np.asarray(amount, dtype="float64")
        )
        volatility_values = (
            np.full(directions.shape, np.nan, dtype="float64")
            if volatility is None
            else np.asarray(volatility, dtype="float64")
        )
        if amount_values.shape != directions.shape or volatility_values.shape != directions.shape:
            raise ExecutionError("冲击成本输入形状与成交数组不一致")
        active = (
            (directions != 0)
            & (quantities > 0)
            & np.isfinite(reference)
            & (reference > 0)
        )
        safe_reference = np.where(np.isfinite(reference), reference, 0.0)
        slippage = float(self.slippage)
        gross_amount = np.where(active, quantities * safe_reference, 0.0)
        if self.impact_model == "sqrt":
            valid_impact = (
                active
                & np.isfinite(amount_values)
                & (amount_values > 0)
                & np.isfinite(volatility_values)
            )
            participation = np.divide(
                gross_amount,
                amount_values,
                out=np.zeros_like(gross_amount),
                where=valid_impact,
            )
            sigma = np.maximum(volatility_values, float(self.impact_volatility_floor))
            impact_rate = np.where(
                valid_impact,
                np.minimum(
                    float(self.impact_coefficient) * sigma * np.sqrt(participation),
                    float(self.impact_max_rate),
                ),
                0.0,
            )
        else:
            impact_rate = np.zeros_like(gross_amount)
        fill_price = np.where(
            active,
            safe_reference
            * (1.0 + directions.astype("float64") * (slippage + impact_rate)),
            0.0,
        )
        fill_amount = np.where(active, quantities * fill_price, 0.0)
        buy_rate, sell_rate = self.stamp_rates(day)
        commission_raw = np.maximum(
            float(self.minimum_commission),
            fill_amount * float(self.commission_rate),
        )
        commission = np.where(active, commission_raw, 0.0)
        stamp_rate = np.where(directions < 0, sell_rate, buy_rate)
        stamp_tax = np.where(active, fill_amount * stamp_rate, 0.0)
        transfer_fee = np.zeros_like(fill_amount)
        if self.transfer_fee_enabled:
            if exchanges is None:
                exchange_values = np.array(
                    [
                        exchange_for(code, board) if is_active else ""
                        for code, board, is_active in zip(
                            codes.ravel(), board_values.ravel(), active.ravel()
                        )
                    ],
                    dtype=object,
                ).reshape(codes.shape)
            else:
                exchange_values = np.asarray(exchanges, dtype=object)
                if exchange_values.shape != codes.shape:
                    raise ExecutionError("预解析交易所矩阵形状与成交数组不一致")
            for exchange in ("SH", "SZ", "BJ"):
                mask = active & (exchange_values == exchange)
                if not mask.any():
                    continue
                terms = self.transfer_terms(day, exchange)
                if terms is None:
                    continue
                rate, basis, minimum = terms
                if basis == "amount":
                    raw = fill_amount * rate
                else:
                    raw = quantities * PAR_VALUE_CNY * rate
                transfer_fee = np.where(mask, np.maximum(raw, minimum), transfer_fee)
        commission = self._round(commission)
        stamp_tax = self._round(stamp_tax)
        transfer_fee = self._round(transfer_fee)
        fees = commission + stamp_tax + transfer_fee
        slippage_cost = np.where(active, quantities * safe_reference * slippage, 0.0)
        impact_cost = np.where(active, gross_amount * impact_rate, 0.0)
        total_cost = fees + slippage_cost + impact_cost
        return ChargeArrays(
            reference_price=safe_reference,
            fill_price=fill_price,
            gross_amount=gross_amount,
            fill_amount=fill_amount,
            commission=commission,
            stamp_tax=stamp_tax,
            transfer_fee=transfer_fee,
            fees=fees,
            slippage_rate=slippage,
            slippage_cost=slippage_cost,
            impact_rate=impact_rate,
            impact_cost=impact_cost,
            total_cost=total_cost,
        )

    def charge(
        self,
        day,
        side: str | int,
        shares: float,
        reference_price: float,
        stock_code: str,
        board: str | None = None,
        amount: float | None = None,
        volatility: float | None = None,
    ) -> FeeCharge:
        """计算一笔成交的成本分项。"""
        if isinstance(side, str):
            normalized = side.strip().lower()
            if normalized == "buy":
                direction = 1
            elif normalized == "sell":
                direction = -1
            else:
                raise ExecutionError(f"不支持的成交方向：{side!r}")
        else:
            direction = int(side)
            if direction not in (-1, 0, 1):
                raise ExecutionError(f"不支持的成交方向：{side!r}")
        arrays = self.charge_arrays(
            day,
            np.array([direction], dtype="int8"),
            np.array([float(shares)], dtype="float64"),
            np.array([float(reference_price)], dtype="float64"),
            np.array([stock_code], dtype=object),
            np.array([board], dtype=object),
            amount=None if amount is None else np.array([float(amount)], dtype="float64"),
            volatility=(
                None
                if volatility is None
                else np.array([float(volatility)], dtype="float64")
            ),
        )
        return FeeCharge(
            reference_price=float(arrays.reference_price[0]),
            fill_price=float(arrays.fill_price[0]),
            gross_amount=float(arrays.gross_amount[0]),
            fill_amount=float(arrays.fill_amount[0]),
            commission=float(arrays.commission[0]),
            stamp_tax=float(arrays.stamp_tax[0]),
            transfer_fee=float(arrays.transfer_fee[0]),
            fees=float(arrays.fees[0]),
            slippage_rate=float(arrays.slippage_rate),
            slippage_cost=float(arrays.slippage_cost[0]),
            impact_rate=float(arrays.impact_rate[0]),
            impact_cost=float(arrays.impact_cost[0]),
            total_cost=float(arrays.total_cost[0]),
        )


def order_cost_frame(orders: pd.DataFrame | None, index: pd.Index) -> pd.DataFrame:
    """把成交明细聚合成逐日成本分项。

    老订单表没有分项时，把 ``Fees`` 整体记入佣金，保证旧结果仍能派生。
    """
    columns = [
        "commission",
        "stamp_tax",
        "transfer_fee",
        "slippage_cost",
        "impact_cost",
    ]
    frame = pd.DataFrame(0.0, index=pd.DatetimeIndex(index), columns=columns)
    if orders is None or orders.empty:
        return frame
    date_column = "trade_date" if "trade_date" in orders.columns else "Timestamp"
    if date_column not in orders.columns:
        return frame
    dates = pd.DatetimeIndex(orders[date_column])
    if "commission" in orders.columns:
        for name in columns:
            values = (
                orders[name].astype("float64")
                if name in orders.columns
                else pd.Series(0.0, index=orders.index)
            )
            frame[name] = values.groupby(dates).sum().reindex(frame.index).fillna(0.0)
    elif "Fees" in orders.columns:
        frame["commission"] = (
            orders["Fees"].astype("float64").groupby(dates).sum()
            .reindex(frame.index).fillna(0.0)
        )
    return frame


def cost_model_summary(spec: CostSpec) -> dict[str, Any]:
    """返回可写入运行元数据的成本口径摘要。"""
    return {
        "version": COST_MODEL_VERSION,
        "commission_rate": float(spec.commission_rate),
        "minimum_commission": float(spec.minimum_commission),
        "slippage": float(spec.slippage),
        "impact_model": spec.impact_model,
        "impact_coefficient": float(spec.impact_coefficient),
        "impact_volatility_window": int(spec.impact_volatility_window),
        "impact_volatility_floor": float(spec.impact_volatility_floor),
        "impact_max_rate": float(spec.impact_max_rate),
        "stamp_tax_rate_override": spec.stamp_tax_rate,
        "transfer_fee_enabled": bool(spec.transfer_fee_enabled),
        "fee_rounding": spec.fee_rounding,
        "par_value_cny": PAR_VALUE_CNY,
        "stamp_tax_tiers": [asdict(item) for item in STAMP_TAX_SCHEDULE],
        "transfer_fee_tiers": [asdict(item) for item in TRANSFER_FEE_SCHEDULE],
        "dividend_tax_tiers": [asdict(item) for item in DIVIDEND_TAX_SCHEDULE],
        "dividend_tax_mode": spec.dividend_tax_mode,
    }
