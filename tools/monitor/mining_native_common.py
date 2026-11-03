"""挖掘候选相关性原生基准的固定参数。"""

SCHEMA_VERSION = "mining_native_baseline_v1"
BACKENDS = ("python", "native", "auto")
EXPRESSIONS = (
    "mean(close, 5) + rank(open)",
    "mean(close, 5) - open",
    "rank(mean(close, 5))",
    "mean(close, 5) + close",
)
START = "2024-03-01"
END = "2024-12-31"
QUICK_START = "2024-05-06"
QUICK_END = "2024-06-28"
MIN_CORRELATION_SPEEDUP = 1.20
MIN_TECHNICAL_IMPROVEMENT = 0.10
MAX_RSS_RATIO = 1.20
REPEATS = 3
