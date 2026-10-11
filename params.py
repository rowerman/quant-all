"""评估参数：一次评估请求的全部输入。

参数在这里集中定义与校验，指标层只读不改，保证同一份参数算出的结果可复现。
"""

from __future__ import annotations

from dataclasses import dataclass

from calculate.errors import EvalParamError

# `md.fwd_return` 提供的标签周期与价格通道
HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 15, 20)
# 多周期衰减（V11-3）用的档位：库里的 15 档不纳入，避免曲线过密
DECAY_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)
PRICE_CHANNELS: tuple[str, ...] = ("close", "open")
QUANTILE_CHOICES: tuple[int, ...] = (5, 10, 20)

# 单次请求的时间跨度上限（交易日数），与面板行数上限一起防止把内存打满
MAX_TRADE_DAYS = 2400


@dataclass(frozen=True)
class EvalParams:
    """一次因子评估的输入参数。

    参数:
        factor: 因子名，必须在 `factors` 注册表里且已落盘。
        start: 起始交易日，闭区间，ISO 格式。
        end: 截止交易日，闭区间，ISO 格式。
        universe: 股票池标识，取值见 `calculate.universe.list_universes`。
        horizon: 标签周期，取值见 ``HORIZONS``。
        price: 标签价格通道，``close`` 或 ``open``。
        quantiles: 分层组数，取值见 ``QUANTILE_CHOICES``。
        winsorize: 算 IC 时是否对因子与标签做截面 3 倍 MAD 去极值。
    """

    factor: str
    start: str
    end: str
    universe: str = "all_a"
    horizon: int = 5
    price: str = "close"
    quantiles: int = 10
    winsorize: bool = True

    def __post_init__(self) -> None:
        if not self.factor:
            raise EvalParamError("因子名不能为空")
        if self.start > self.end:
            raise EvalParamError(f"起始日 {self.start} 晚于截止日 {self.end}")
        if self.horizon not in HORIZONS:
            raise EvalParamError(f"不支持的标签周期 {self.horizon}，可选：{list(HORIZONS)}")
        if self.price not in PRICE_CHANNELS:
            raise EvalParamError(f"不支持的价格通道 {self.price!r}，可选：{list(PRICE_CHANNELS)}")
        if self.quantiles not in QUANTILE_CHOICES:
            raise EvalParamError(f"不支持的分组数 {self.quantiles}，可选：{list(QUANTILE_CHOICES)}")

    @property
    def label_column(self) -> str:
        """面板标签列对应的 ``md.fwd_return`` 列名。"""
        return self.label_column_for(self.horizon)

    def label_column_for(self, horizon: int) -> str:
        """取指定周期的标签列名，净值曲线等需要换周期时用。

        参数:
            horizon: 标签周期。
        """
        return f"fwd_ret_{self.price}_{horizon}"

    def horizon_label_column(self, horizon: int) -> str:
        """把衰减分析用到的标签档位映射到评估面板的列名。

        参数:
            horizon: 标签周期，必须落在 ``DECAY_HORIZONS`` 里。
        """
        if horizon not in DECAY_HORIZONS:
            raise EvalParamError(
                f"衰减分析只支持 {list(DECAY_HORIZONS)} 这几档，收到 {horizon}"
            )
        if horizon == self.horizon:
            return "label"
        if horizon == 1:
            return "label_1d"
        return f"label_h{horizon}"

    def extra_label_specs(self) -> tuple[tuple[int, str], ...]:
        """除 ``label`` 与 ``label_1d`` 之外，面板还要带的标签档位与列名。"""
        return tuple(
            (horizon, f"label_h{horizon}")
            for horizon in DECAY_HORIZONS
            if horizon not in (1, self.horizon)
        )

    def extra_label_columns(self) -> tuple[str, ...]:
        """除 ``label`` 与 ``label_1d`` 之外，面板还要带的多周期标签列。"""
        return tuple(column for _, column in self.extra_label_specs())

    def describe(self) -> dict:
        """把参数摊平成可序列化的字典，供 meta 事件回显。"""
        return {
            "factor": self.factor,
            "start": self.start,
            "end": self.end,
            "universe": self.universe,
            "horizon": self.horizon,
            "price": self.price,
            "label": self.label_column,
            "quantiles": self.quantiles,
            "winsorize": self.winsorize,
        }
