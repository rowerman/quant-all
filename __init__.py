"""基于 VectorBT 的 A 股日频回测适配层。"""

from backtest.builtin import BUILTINS, MOM20_ID, MULTIFACTOR_ID
from backtest.data import MarketData, load_market_data
from backtest.engine import BacktestResult, run_backtest
from backtest.errors import (
    BacktestError,
    DataAlignmentError,
    ExecutionError,
    ResourceLimitError,
    RunCancelled,
    SpecError,
)
from backtest.metrics import summarize
from backtest.orders import OrderPlan, build_order_plan, execution_targets, liquidation_cells
from backtest.pipeline import RunRequest, iter_run, run
from backtest.signals import build_target_weights, eligible_mask, rebalance_mask
from backtest.specs import (
    BacktestParams,
    BenchmarkSpec,
    CostSpec,
    ExecutionSpec,
    PortfolioSpec,
    RebalanceSpec,
    RiskSpec,
    SignalSpec,
    StrategySpec,
)
from backtest.store import BacktestStore
from backtest.universe import list_universes, membership_frame, resolve_universe

__all__ = [
    "BUILTINS",
    "MOM20_ID",
    "MULTIFACTOR_ID",
    "BacktestError",
    "BacktestParams",
    "BacktestResult",
    "BacktestStore",
    "BenchmarkSpec",
    "CostSpec",
    "DataAlignmentError",
    "ExecutionError",
    "ExecutionSpec",
    "MarketData",
    "OrderPlan",
    "PortfolioSpec",
    "RebalanceSpec",
    "RiskSpec",
    "ResourceLimitError",
    "RunCancelled",
    "RunRequest",
    "SignalSpec",
    "SpecError",
    "StrategySpec",
    "build_order_plan",
    "build_target_weights",
    "eligible_mask",
    "execution_targets",
    "iter_run",
    "liquidation_cells",
    "list_universes",
    "load_market_data",
    "membership_frame",
    "rebalance_mask",
    "resolve_universe",
    "run",
    "run_backtest",
    "summarize",
]

