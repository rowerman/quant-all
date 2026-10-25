"""因子读取接口的单元测试。"""

import duckdb
import pytest

import quant
from scripts.build_factors import build as build_module
from scripts.config import DB_FILE
from scripts.errors import FactorDataError, FactorRegistryError

pytestmark = pytest.mark.skipif(not DB_FILE.exists(), reason="需要真实主库 data/databases/quant.duckdb")

YEAR = 2024


@pytest.fixture(scope="module")
def factor_dir(tmp_path_factory):
    """构建动量族与估值族 2024 年数据，供读取接口测试。"""
    root = tmp_path_factory.mktemp("factor_load")
    build_module.build(
        ["momentum", "valuation"], f"{YEAR}-01-01", f"{YEAR}-12-31",
        factor_dir=root, catalog_file=root / "factors.duckdb",
    )
    return root


def test_load_factor_panel_returns_requested_columns(factor_dir):
    """长表返回主键 + 指定因子列，并按交易日、股票排序。"""
    frame = quant.load_factor_panel(
        start=f"{YEAR}-01-02", end=f"{YEAR}-01-10",
        names=["MOM_20", "EP_TTM"], factor_dir=factor_dir,
    )
    assert list(frame.columns) == ["stock_code", "trade_date", "MOM_20", "EP_TTM"]
    assert frame["trade_date"].is_monotonic_increasing
    assert len(frame) > 1000


def test_load_factor_panel_filters_stocks(factor_dir):
    """股票池参数要生效。"""
    frame = quant.load_factor_panel(
        start=f"{YEAR}-01-02", end=f"{YEAR}-01-10",
        names=["MOM_20"], stocks=["SH600000"], factor_dir=factor_dir,
    )
    assert set(frame["stock_code"]) == {"SH600000"}


def test_load_factor_panel_by_family(factor_dir):
    """只给族也能取到该族全部因子。"""
    frame = quant.load_factor_panel(
        start=f"{YEAR}-01-02", end=f"{YEAR}-01-05",
        families=["valuation"], factor_dir=factor_dir,
    )
    assert {"EP_TTM", "BP", "LN_MV"} <= set(frame.columns)


def test_load_factor_panel_all_families(factor_dir):
    """names 与 families 都为空时取全部已构建的族。"""
    frame = quant.load_factor_panel(
        start=f"{YEAR}-01-02", end=f"{YEAR}-01-05", factor_dir=factor_dir,
    )
    assert {"MOM_20", "EP_TTM"} <= set(frame.columns)


def test_load_factor_wide_shape(factor_dir):
    """宽表是交易日 × 股票，列数等于当日股票数。"""
    wide = quant.load_factor_wide(
        "MOM_20", start=f"{YEAR}-01-02", end=f"{YEAR}-01-05", factor_dir=factor_dir,
    )
    assert wide.index.name == "trade_date"
    assert wide.shape[0] == 4
    assert wide.shape[1] > 1000


def test_unknown_factor_name_is_reported(factor_dir):
    """未知因子名要报错并列出可用因子。"""
    with pytest.raises(FactorRegistryError, match="未知因子"):
        quant.load_factor_panel(names=["NOT_A_FACTOR"], factor_dir=factor_dir)


def test_unbuilt_family_is_reported(factor_dir):
    """没落盘的族要给出「已构建了哪些族」的提示。"""
    with pytest.raises(FactorDataError, match="还没有落盘数据"):
        quant.load_factor_panel(families=["technical"], factor_dir=factor_dir)


def test_unknown_family_is_reported(factor_dir):
    """非法族名要报错。"""
    with pytest.raises(FactorDataError, match="未知因子族"):
        quant.load_factor_panel(families=["no_such_family"], factor_dir=factor_dir)


def test_missing_factor_dir_is_reported(tmp_path):
    """目录不存在时给出可执行的提示。"""
    with pytest.raises(FactorDataError, match="scripts.build_factors.build"):
        quant.load_factor_panel(start=f"{YEAR}-01-02", factor_dir=tmp_path / "empty")


def test_load_factor_list_reads_catalog(factor_dir):
    """因子清单来自目录库，含已实现与待补齐两类。"""
    frame = quant.load_factor_list(db_path=factor_dir / "factors.duckdb")
    assert {"name", "family", "status", "formula"} <= set(frame.columns)
    statuses = set(frame["status"])
    assert statuses == {"implemented", "planned"}
    implemented = frame.loc[frame["status"] == "implemented", "name"].tolist()
    assert "MOM_20" in implemented


def test_load_factor_list_exposes_formula_and_steps(factor_dir):
    """清单要能回答「这个因子怎么算」：公式、白话步骤、实现表达式都要有。"""
    frame = quant.load_factor_list(family="momentum", db_path=factor_dir / "factors.duckdb")
    row = frame.loc[frame["name"] == "MOM_20"].iloc[0]
    assert row["formula"] == "close_t / close_(t-20) - 1"
    assert len(row["steps"].split("\n")) >= 2
    assert "lag(close, 20)" in row["expr"]


def test_load_factor_list_family_filter(factor_dir):
    """按族过滤清单。"""
    frame = quant.load_factor_list(family="momentum", db_path=factor_dir / "factors.duckdb")
    assert set(frame["family"]) == {"momentum"}


def test_load_factor_list_missing_catalog(tmp_path):
    """目录库不存在时要提示先构建。"""
    with pytest.raises(FileNotFoundError, match="scripts.build_factors.build"):
        quant.load_factor_list(db_path=tmp_path / "none.duckdb")


def test_built_families_lists_disk_state(factor_dir):
    """已落盘族清单来自磁盘，而不是注册表。"""
    assert quant.built_families(factor_dir) == ("momentum", "valuation")
