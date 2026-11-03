"""已注册提供者的结构化参数说明。

参数结构只服务新建策略页，不参与提供者执行。按 ``(name, version)`` 索引可以避免
修改提供者源码，从而保持已登记源码哈希和既有策略版本不变。
"""

from alpha.evaluation_plan import current_model_ids

SCHEMA_MODEL_ID, SCHEMA_FALLBACK_ID = current_model_ids()

PARAM_SCHEMAS = {
    ("ml_rank", "1"): (
        {
            "name": "model_id",
            "label": "模型 ID",
            "type": "string",
            "default": SCHEMA_MODEL_ID,
            "description": "使用哪个已登记模型的样本外预测。",
        },
        {
            "name": "top_n",
            "label": "持仓只数",
            "type": "integer",
            "default": 30,
            "min": 1,
            "step": 1,
            "description": "4 周平滑后每周等权持有的股票数。",
        },
    ),
    ("ml_rank", "2"): (
        {
            "name": "model_id",
            "label": "模型 ID",
            "type": "string",
            "default": SCHEMA_MODEL_ID,
            "description": "使用哪个已登记模型的样本外预测。",
        },
        {
            "name": "top_n",
            "label": "持仓只数",
            "type": "integer",
            "default": 30,
            "min": 1,
            "step": 1,
            "description": "4 周平滑后等权持有的股票数。",
        },
        {
            "name": "smoothing_weeks",
            "label": "平滑周数",
            "type": "integer",
            "default": 4,
            "min": 1,
            "step": 1,
            "description": "使用 1 表示原始分数，使用 4 表示当前选中的四周 rank 平滑。",
        },
    ),
    ("ml_rank", "3"): (
        {
            "name": "lifecycle_id",
            "label": "生命周期 ID",
            "type": "string",
            "default": "p5_3_dd7c9b4bf235",
            "description": "P5-3 生成且通过完整哈希校验的生命周期包。",
        },
        {
            "name": "model_id",
            "label": "主模型 ID",
            "type": "string",
            "default": SCHEMA_MODEL_ID,
            "description": "normal 和 reduced 状态使用的主模型。",
        },
        {
            "name": "fallback_model_id",
            "label": "回滚模型 ID",
            "type": "string",
            "default": SCHEMA_FALLBACK_ID,
            "description": "rollback 状态使用的兜底模型。",
        },
        {
            "name": "top_n",
            "label": "持仓只数",
            "type": "integer",
            "default": 30,
            "min": 1,
            "step": 1,
            "description": "按活动模型四周平滑分数等权持有的股票数。",
        },
        {
            "name": "smoothing_weeks",
            "label": "平滑周数",
            "type": "integer",
            "default": 4,
            "min": 1,
            "step": 1,
            "description": "活动模型分数的因果 rank 平滑周数。",
        },
    ),
    ("momentum_timing", "1"): (
        {
            "name": "top_n",
            "label": "持仓只数",
            "type": "integer",
            "default": 30,
            "min": 1,
            "step": 1,
            "description": "广度达标时等权持有的股票数。",
        },
        {
            "name": "min_breadth",
            "label": "最低市场广度",
            "type": "percent",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "description": "MOM_20 大于 0 的可交易股票占比达到该值才建仓。",
        },
    ),
    ("momentum_trailing_stop", "1"): (
        {
            "name": "top_n",
            "label": "持仓只数",
            "type": "integer",
            "default": 30,
            "min": 1,
            "step": 1,
            "description": "每周按 MOM_20 排名持有的股票数。",
        },
        {
            "name": "stop_loss",
            "label": "固定止损",
            "type": "percent",
            "default": 0.10,
            "min": 0.000001,
            "max": 0.999999,
            "step": 0.005,
            "description": "持仓浮亏达到该比例时清仓。",
        },
        {
            "name": "trailing_stop",
            "label": "移动止损",
            "type": "percent",
            "default": 0.15,
            "min": 0.000001,
            "max": 0.999999,
            "step": 0.005,
            "description": "收盘价从建仓以来最高价回撤达到该比例时清仓。",
        },
        {
            "name": "cooldown_days",
            "label": "再入场冷却天数",
            "type": "integer",
            "default": 10,
            "min": 0,
            "step": 1,
            "description": "清仓后多少个自然日内不再买回同一只股票。",
        },
    ),
}
