"""P5-4 模型卡与正式产物的一致性测试。"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from common.paths import MODEL_DATA_DIR

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "lightgbm_ensemble_label_rank_6aea5c9299d5"
CARD = ROOT / "docs" / "research" / "modeling" / "alpha" / f"model_card_{MODEL_ID}.md"


def _read_card() -> str:
    """读取正式模型卡。"""

    return CARD.read_text(encoding="utf-8")


def test_model_card_has_required_sections_and_registered_model():
    """模型卡覆盖 P5-4 必需内容，并指向已登记正式模型。"""

    text = _read_card()
    headings = (
        "## 1. 用途与结论先看",
        "## 2. 输入、输出与使用边界",
        "## 3. 数据与特征",
        "## 4. 训练与验证",
        "## 5. 历史表现",
        "## 6. 容量、监控与生命周期",
        "## 7. 复现与产物定位",
        "## 8. 已知限制与失效风险",
    )
    for heading in headings:
        assert heading in text

    registry = json.loads((MODEL_DATA_DIR / "index.json").read_text(encoding="utf-8"))
    registered = {item["model_id"] for item in registry["models"]}
    assert MODEL_ID in registered
    assert f"模型 ID：`{MODEL_ID}`" in text

    readme = (ROOT / "docs" / "research" / "modeling" / "alpha" / "README.md").read_text(encoding="utf-8")
    assert CARD.name in readme
    assert "P5-4：正式主模型已生成" in readme


def test_model_card_records_official_ids_and_key_metrics():
    """模型卡保留可追溯的正式 ID 和关键指标。"""

    text = _read_card()
    official_ids = (
        "6a59680fd71c50a8",
        "p3_4_8152f500bb86",
        "p3_5_5c02d3520f25",
        "p4_4a_23c7e101aee9",
        "p4_4b1_e9c029453c2d",
        "p4_4b2_52b6a7b661ff",
        "p5_2_0a9e6523e9e7",
        "p5_3_dd7c9b4bf235",
        "p5_3_4f91d5b68fba",
    )
    for official_id in official_ids:
        assert official_id in text

    from alpha.artifacts import model_dir

    metrics = json.loads(
        (model_dir(MODEL_ID) / "metrics.json").read_text(encoding="utf-8")
    )
    key_values = (
        f"{metrics['test_model']['rank_ic_mean']:.4f}",
        f"{metrics['test_model']['icir']:.4f}",
    )
    for value in key_values:
        assert value in text

    capacity_path = (
        MODEL_DATA_DIR
        / "runs"
        / "acceptance"
        / "p4_4b2_52b6a7b661ff"
        / "capacity_curve.csv"
    )
    with capacity_path.open(encoding="utf-8", newline="") as handle:
        capacity = {float(row["init_cash"]): row for row in csv.DictReader(handle)}
    expected_capacity = {
        1_000_000.0: ("28.35%", "29.68%", "-28.13%"),
        100_000_000.0: ("10.45%", "11.78%", "-32.90%"),
    }
    for capital, expected in expected_capacity.items():
        row = capacity[capital]
        actual = (
            f"{float(row['annual_return']) * 100:.2f}%",
            f"{float(row['excess_annual_return']) * 100:.2f}%",
            f"{float(row['max_drawdown']) * 100:.2f}%",
        )
        assert actual == expected
        for value in actual:
            assert value in text

    lifecycle_metrics = json.loads(
        (
            MODEL_DATA_DIR
            / "runs"
            / "acceptance"
            / "p5_3_4f91d5b68fba"
            / "ML_LIFECYCLE"
            / "metrics.json"
        ).read_text(encoding="utf-8")
    )["metrics"]
    for key in ("annual_return", "excess_annual_return", "max_drawdown"):
        assert f"{lifecycle_metrics[key] * 100:.2f}%" in text


def test_model_card_local_links_exist():
    """模型卡引用的本地文档和产物路径都存在。"""

    text = _read_card()
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    local_links = [
        link.split("#", 1)[0]
        for link in links
        if link and not link.startswith(("http://", "https://"))
    ]
    assert local_links
    missing = [link for link in local_links if not (CARD.parent / link).exists()]
    assert missing == []
