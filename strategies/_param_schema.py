"""已注册提供者的结构化参数说明。

参数结构只服务新建策略页，不参与提供者执行。按 ``(name, version)`` 索引可以避免
修改提供者源码，从而保持已登记源码哈希和既有策略版本不变。
"""

PARAM_SCHEMAS = {
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
