"""策略配置和回测运行参数。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from backtest.errors import SpecError


def _date(value: str, name: str) -> str:
    """校验并返回 ISO 日期。"""
    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as error:
        raise SpecError(f"{name} 必须是 YYYY-MM-DD：{value!r}") from error
    return value


@dataclass(frozen=True)
class SignalSpec:
    """因子截面信号定义。"""

    factors: tuple[str, ...] = ("MOM_20",)
    weights: tuple[float, ...] = (1.0,)
    winsorize: bool = False
    standardize: bool = True

    def __post_init__(self) -> None:
        if not self.factors or any(not isinstance(item, str) or not item for item in self.factors):
            raise SpecError("signal.factors 必须是非空因子名序列")
        if len(self.weights) != len(self.factors):
            raise SpecError("signal.weights 数量必须与 factors 一致")
        if any(not isinstance(item, (int, float)) for item in self.weights):
            raise SpecError("signal.weights 必须是数字")
        if not any(float(item) != 0 for item in self.weights):
            raise SpecError("signal.weights 不能全部为 0")


@dataclass(frozen=True)
class PortfolioSpec:
    """组合选择和权重定义。

    ``weighting`` 支持 ``equal``（等权）与 ``market_cap``（选中股票内按当日总市值归一）。
    """

    selection: str = "top_n"
    top_n: int = 30
    weighting: str = "equal"
    long_only: bool = True

    def __post_init__(self) -> None:
        if self.selection != "top_n":
            raise SpecError("第一版只支持 selection='top_n'")
        if type(self.top_n) is not int or self.top_n <= 0:
            raise SpecError("portfolio.top_n 必须是正整数")
        if self.weighting not in ("equal", "market_cap"):
            raise SpecError("weighting 只支持 equal/market_cap")
        if self.long_only is not True:
            raise SpecError("第一版只支持 long_only=True")


@dataclass(frozen=True)
class RebalanceSpec:
    """调仓和执行时点。"""

    frequency: str = "monthly"
    signal_at: str = "close"
    execute_at: str = "next_open"

    def __post_init__(self) -> None:
        if self.frequency not in ("daily", "weekly", "monthly"):
            raise SpecError("frequency 只支持 daily/weekly/monthly")
        if self.signal_at != "close" or self.execute_at != "next_open":
            raise SpecError("第一版固定为收盘生成信号、下一交易日开盘执行")


@dataclass(frozen=True)
class ExecutionSpec:
    """A 股交易限制。

    ``enforce_t_plus_1`` 在当前执行口径下结构性满足：订单最早在信号日次一交易日
    开盘执行，同一天不会对同一只股票既买又卖，因此该开关不会产生额外拒单。
    """

    lot_size: int = 100
    reject_suspended: bool = True
    reject_limit_up_buy: bool = True
    reject_limit_down_sell: bool = True
    enforce_t_plus_1: bool = True

    def __post_init__(self) -> None:
        if type(self.lot_size) is not int or self.lot_size <= 0:
            raise SpecError("execution.lot_size 必须是正整数")
        if self.lot_size != 100:
            raise SpecError("第一版固定使用 100 股交易单位")


@dataclass(frozen=True)
class CostSpec:
    """交易成本参数。费率使用小数表示。"""

    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    slippage: float = 0.0005

    def __post_init__(self) -> None:
        values = (self.commission_rate, self.minimum_commission, self.stamp_tax_rate, self.slippage)
        if any(float(item) < 0 for item in values):
            raise SpecError("交易成本不能为负数")


@dataclass(frozen=True)
class BenchmarkSpec:
    """基准指数配置。"""

    index_code: str | None = "000300.SH"


@dataclass(frozen=True)
class StrategySpec:
    """可复现的策略定义。"""

    name: str = "MOM_20 Top30"
    version: int = 1
    universe: str = "all_a"
    signal: SignalSpec = field(default_factory=SignalSpec)
    portfolio: PortfolioSpec = field(default_factory=PortfolioSpec)
    rebalance: RebalanceSpec = field(default_factory=RebalanceSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    costs: CostSpec = field(default_factory=CostSpec)
    benchmark: BenchmarkSpec = field(default_factory=BenchmarkSpec)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SpecError("策略名称不能为空")
        if type(self.version) is not int or self.version <= 0:
            raise SpecError("策略 version 必须是正整数")
        if self.universe != "all_a" and not self.universe.startswith("index:"):
            raise SpecError("universe 必须是 all_a 或 index:<指数代码>")

    def to_dict(self) -> dict[str, Any]:
        """转换为稳定的嵌套字典。"""
        return asdict(self)

    def to_json(self) -> str:
        """序列化为稳定 JSON。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        """返回配置内容的 SHA-256 前 16 位。"""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategySpec":
        """从嵌套字典恢复策略。"""
        if not isinstance(payload, dict):
            raise SpecError("策略配置必须是对象")
        known = {"name", "version", "universe", "signal", "portfolio", "rebalance", "execution", "costs", "benchmark"}
        unknown = set(payload) - known
        if unknown:
            raise SpecError(f"策略配置包含未知字段：{sorted(unknown)}")
        def section(name: str, kind):
            value = payload.get(name, {})
            if not isinstance(value, dict):
                raise SpecError(f"策略配置 {name} 必须是对象")
            if kind is SignalSpec:
                value = dict(value)
                if "factors" in value:
                    value["factors"] = tuple(value["factors"])
                if "weights" in value:
                    value["weights"] = tuple(value["weights"])
            return kind(**value)
        return cls(
            name=payload.get("name", cls.name),
            version=payload.get("version", cls.version),
            universe=payload.get("universe", cls.universe),
            signal=section("signal", SignalSpec),
            portfolio=section("portfolio", PortfolioSpec),
            rebalance=section("rebalance", RebalanceSpec),
            execution=section("execution", ExecutionSpec),
            costs=section("costs", CostSpec),
            benchmark=section("benchmark", BenchmarkSpec),
        )


@dataclass(frozen=True)
class BacktestParams:
    """一次回测的区间和初始资金。"""

    start: str
    end: str
    init_cash: float = 1_000_000.0

    def __post_init__(self) -> None:
        start = _date(self.start, "start")
        end = _date(self.end, "end")
        if start > end:
            raise SpecError("start 不能晚于 end")
        if not isinstance(self.init_cash, (int, float)) or self.init_cash <= 0:
            raise SpecError("init_cash 必须是正数")
