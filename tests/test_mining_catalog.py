"""S3 字段与算子目录的单元和 schema 一致性测试。"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from common.paths import MINING_CATALOG_FILE, PROJECT_ROOT
from factors.sources import SOURCES
from mining.catalog import (
    load_catalog,
    render_catalog_markdown,
    validate_catalog,
)
from mining.contracts import FrozenContext
from mining.errors import CatalogDriftError


def test_frozen_catalog_has_expected_scope_and_sorted_entries():
    """目录固定首批字段、来源和全部算子。"""

    catalog = load_catalog()
    assert catalog.schema_version == "mining_catalog_v1"
    assert len(catalog.fields) == 95
    assert len(catalog.allowed_fields) == 84
    assert len(catalog.operators) == 34
    assert [field.name for field in catalog.fields] == sorted(
        field.name for field in catalog.fields
    )
    assert [operator.name for operator in catalog.operators] == sorted(
        operator.name for operator in catalog.operators
    )
    assert {field.source for field in catalog.fields} == {"daily_hfq", *SOURCES}


def test_catalog_rejects_unsafe_daily_fields_and_accepts_required_sources():
    """键、日期、标签和未复权价格不能进入表达式。"""

    catalog = load_catalog()
    for name in ("stock_code", "trade_date", "status", "change", "pre_close", "limit_up"):
        with pytest.raises(CatalogDriftError, match="未获准"):
            catalog.field(name)
    assert catalog.field("close").dimension == "price"
    assert catalog.field("fi_roe").available_at == "next_trading_day"
    assert catalog.field("idx_ret_300").source == "index_000300"


def test_catalog_hash_and_frozen_mappings_are_stable():
    """同一目录重复加载得到相同哈希和冻结上下文映射。"""

    first = load_catalog()
    second = load_catalog()
    assert first.catalog_hash == second.catalog_hash
    assert first.frozen_field_mappings() == second.frozen_field_mappings()
    assert first.frozen_operator_mappings() == second.frozen_operator_mappings()
    assert len(first.frozen_field_mappings()) == 84

    context = FrozenContext(
        data_fingerprint="a" * 64,
        field_catalog=first.frozen_field_mappings(),
        operator_catalog=first.frozen_operator_mappings(),
    )
    changed = FrozenContext(
        data_fingerprint="a" * 64,
        field_catalog=first.frozen_field_mappings()[1:],
        operator_catalog=first.frozen_operator_mappings(),
    )
    assert context.field_catalog_hash != changed.field_catalog_hash


def test_catalog_matches_real_database_schema():
    """所有允许字段与真实行情库及 factors.sources 一致。"""

    result = validate_catalog()
    assert result.table_count == 6
    assert result.field_count == 95
    assert result.operator_count == 34
    assert len(result.catalog_hash) == 64


def test_catalog_markdown_is_current():
    """可审查文档包含全部字段、算子和当前哈希。"""

    catalog = load_catalog()
    expected = render_catalog_markdown(catalog)
    path = PROJECT_ROOT / "docs/research/mining/catalog.md"
    assert path.read_text(encoding="utf-8") == expected
    assert catalog.catalog_hash in expected
    for field in catalog.fields:
        assert f"`{field.name}`" in expected
    for operator in catalog.operators:
        assert f"`{operator.name}`" in expected


def test_catalog_rejects_schema_drift(tmp_path: Path):
    """行情库缺列时只读核对直接失败。"""

    db_path = tmp_path / "quant.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute("CREATE SCHEMA md")
    connection.execute("CREATE TABLE md.daily_hfq(stock_code VARCHAR)")
    connection.close()
    with pytest.raises(CatalogDriftError, match="行情库缺少"):
        validate_catalog(db_path=db_path)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": "other", "data_version": "v"},
        {
            "schema_version": "mining_catalog_v1",
            "data_version": "v",
            "field_groups": [],
            "operators": {},
        },
    ],
)
def test_catalog_rejects_malformed_manifest(tmp_path: Path, payload):
    """目录版本、字段组和算子组必须完整。"""

    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CatalogDriftError):
        load_catalog(path)


def test_default_catalog_path_is_versioned_config():
    """目录从集中配置路径读取，不散落到运行目录。"""

    assert MINING_CATALOG_FILE == PROJECT_ROOT / "configs/mining/catalog.json"
    assert MINING_CATALOG_FILE.is_file()
