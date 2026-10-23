"""Alpha 模型层路径与第一版研究契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from alpha.errors import AlphaConfigError
from alpha.errors import AlphaDataError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "db"
DB_FILE = DB_DIR / "quant.duckdb"
FACTOR_DB_FILE = DB_DIR / "factors.duckdb"
FACTOR_DIR = PROJECT_ROOT / "factor_data"
MODEL_DATA_DIR = PROJECT_ROOT / "model_data"
MODELS_DIR = MODEL_DATA_DIR / "models"
ARCHIVE_MODELS_DIR = MODEL_DATA_DIR / "archive" / "models"
RUNS_DIR = MODEL_DATA_DIR / "runs"
ACCEPTANCE_RUNS_DIR = RUNS_DIR / "acceptance"
EVALUATION_RUNS_DIR = RUNS_DIR / "evaluation"
FINAL_HOLDOUT_RUNS_DIR = RUNS_DIR / "final_holdout"
LOG_DIR = PROJECT_ROOT / "logs" / "alpha"


def _iso_date(value: str, name: str) -> str:
    """校验并返回 ISO 日期字符串。"""

    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as error:
        raise AlphaConfigError(f"{name} 必须是 YYYY-MM-DD：{value!r}") from error
    return value


@dataclass(frozen=True)
class ResearchContract:
    """第一版 alpha 研究的唯一口径。"""

    label: str = "fwd_ret_open_5"
    horizon: int = 5
    price: str = "open"
    universe: str = "all_a"
    rebalance: str = "weekly"
    top_n: int = 30
    weighting: str = "equal"
    benchmark: str = "H00300.CSI"
    dev_start: str = "2012-01-01"
    dev_end: str = "2023-12-31"
    holdout_start: str = "2024-01-01"
    holdout_end: str = "2026-10-15"
    min_listed_days: int = 60
    purge_days: int = 6
    embargo_days: int = 6

    def __post_init__(self) -> None:
        for name in ("dev_start", "dev_end", "holdout_start", "holdout_end"):
            _iso_date(getattr(self, name), name)
        if self.label != f"fwd_ret_{self.price}_{self.horizon}":
            raise AlphaConfigError(
                f"标签 {self.label} 与 price={self.price}、horizon={self.horizon} 不一致"
            )
        if self.price not in ("open", "close"):
            raise AlphaConfigError("price 只支持 open/close")
        if self.universe != "all_a":
            raise AlphaConfigError("第一版股票池固定为 all_a")
        if self.rebalance != "weekly":
            raise AlphaConfigError("第一版调仓固定为 weekly")
        if self.weighting != "equal":
            raise AlphaConfigError("第一版权重固定为 equal")
        if type(self.top_n) is not int or self.top_n <= 0:
            raise AlphaConfigError("top_n 必须是正整数")
        if self.dev_start > self.dev_end:
            raise AlphaConfigError("开发期起始日晚于截止日")
        if self.holdout_start <= self.dev_end:
            raise AlphaConfigError("最终测试集必须晚于开发期")
        if self.holdout_start > self.holdout_end:
            raise AlphaConfigError("最终测试集起始日晚于截止日")
        minimum_gap = self.horizon + 1
        if self.purge_days < minimum_gap:
            raise AlphaConfigError(f"purge_days 不能小于 {minimum_gap}")
        if self.embargo_days < minimum_gap:
            raise AlphaConfigError(f"embargo_days 不能小于 {minimum_gap}")

    def to_dict(self) -> dict:
        """转换为稳定字典。"""

        return asdict(self)


DEFAULT_CONTRACT = ResearchContract()


def dataset_versions() -> dict:
    """读取行情与因子 manifest 版本。"""

    import quant.db

    data = quant.db.query(
        "SELECT data_version, generated_at FROM meta.manifest", path=DB_FILE
    )
    factor = quant.db.query(
        "SELECT factor_version, factor_count, rows, built_at FROM meta.factor_manifest",
        path=FACTOR_DB_FILE,
    )
    if data.empty or factor.empty:
        raise AlphaDataError("数据库缺少行情或因子 manifest")
    return {
        "data_version": str(data.iloc[0]["data_version"]),
        "data_generated_at": str(data.iloc[0]["generated_at"]),
        "factor_version": str(factor.iloc[0]["factor_version"]),
        "factor_count": int(factor.iloc[0]["factor_count"]),
        "factor_rows": int(factor.iloc[0]["rows"]),
        "factor_built_at": str(factor.iloc[0]["built_at"]),
    }
