"""因子注册表与规划清单的单元测试。"""

import pytest

import factors
from factors.planned import PLANNED
from factors.registry import FactorRegistryError, FactorSpec, required_lookback


def test_all_families_have_factors():
    """内置因子族都要有已注册因子。"""
    assert set(factors.used_families()) == set(factors.FAMILIES)


def test_every_factor_has_formula_and_description():
    """公式与说明是文档的来源，不能留空。"""
    for spec in factors.all_factors():
        assert spec.formula.strip(), spec.name
        assert spec.description.strip(), spec.name


def test_duplicate_name_is_rejected():
    """同名因子默认不允许重复注册。"""
    with pytest.raises(FactorRegistryError, match="重复注册"):
        factors.define(
            name="MOM_20", family="momentum", formula="x", description="x", steps="x",
            layers="close", lookback=0,
        )


def test_bad_name_is_rejected():
    """因子名必须是大写字母开头。"""
    with pytest.raises(FactorRegistryError, match="不合规范"):
        FactorSpec(name="mom20", family="momentum", formula="x", description="x",
                   steps=("x",), layers=("close",), lookback=0)


def test_unknown_family_is_rejected():
    """族必须来自固定清单。"""
    with pytest.raises(FactorRegistryError, match="未知因子族"):
        FactorSpec(name="X_1", family="not_a_family", formula="x", description="x",
                   steps=("x",), layers=("close",), lookback=0)


def test_unknown_source_is_rejected():
    """引用了没定义的数据源要报错，而不是构建时才炸。"""
    with pytest.raises(FactorRegistryError, match="未知数据源"):
        FactorSpec(name="X_2", family="market", formula="x", description="x",
                   steps=("x",), layers=("idx_ret_300",), lookback=0,
                   extra_sources=("index_missing",))


def test_lookback_must_cover_window():
    """声明的回看窗口小于表达式实际需要时要报错，这是跨年取错数的根源。"""
    with pytest.raises(FactorRegistryError, match="至少需要回看"):
        FactorSpec(name="X_3", family="momentum", formula="x", description="x",
                   steps=("x",), layers=("close / lag(close, 60) over w - 1",), lookback=20)


def test_required_lookback_reads_windows_and_frames():
    """回看推导要同时看懂 lag 与显式窗口帧。"""
    assert required_lookback(("lag(close, 20) over w",)) == 20
    assert required_lookback(("avg(close) over (partition by stock_code order by trade_date "
                              "rows between 59 preceding and current row)",)) == 60
    assert required_lookback(("ln(close)",)) == 0
    assert required_lookback(("percent_rank() over (partition by trade_date order by close)",)) == 0


def test_unknown_factor_lookup_lists_available():
    """查不到的因子名要把可选清单报出来。"""
    with pytest.raises(FactorRegistryError, match="未知因子"):
        factors.get("NO_SUCH_FACTOR")


def test_unregister_removes_factor():
    """注销后同名因子可以重新注册，测试之间互不污染。"""
    factors.define(name="TMP_FACTOR", family="momentum", formula="x", description="x",
                   steps="x", layers="close", lookback=0)
    assert factors.get("TMP_FACTOR").name == "TMP_FACTOR"
    factors.unregister("TMP_FACTOR")
    with pytest.raises(FactorRegistryError):
        factors.get("TMP_FACTOR")


def test_steps_are_required_and_non_empty():
    """白话步骤是文档的来源，不能缺；空字符串也不行。"""
    with pytest.raises(FactorRegistryError, match="白话计算步骤"):
        FactorSpec(name="X_4", family="momentum", formula="x", description="x",
                   steps=(), layers=("close",), lookback=0)
    with pytest.raises(FactorRegistryError, match="白话计算步骤"):
        FactorSpec(name="X_5", family="momentum", formula="x", description="x",
                   steps=("  ",), layers=("close",), lookback=0)


def test_every_factor_documents_its_steps():
    """29 个已实现因子都要有能看的步骤，且不能只有一句敷衍。"""
    for spec in factors.all_factors():
        assert len(spec.steps) >= 2, f"{spec.name} 只有一步，读者看不出计算过程"
        assert all(len(step.strip()) >= 3 for step in spec.steps), spec.name


def test_planned_list_is_consistent():
    """规划清单要够 100 条、族合法、不重名，且不与已实现因子冲突。"""
    assert len(PLANNED) >= 100
    names = [item["name"] for item in PLANNED]
    assert len(names) == len(set(names))
    families = {family for family in factors.FAMILIES}
    for item in PLANNED:
        assert item["family"] in families, item
        assert item["formula"].strip(), item
    implemented = {spec.name for spec in factors.all_factors()}
    assert implemented <= set(names), "已实现因子也应该出现在规划清单里"


def test_each_family_matches_planned_family():
    """已实现因子所在的族必须与规划清单一致，避免文档分组错乱。"""
    planned_family = {item["name"]: item["family"] for item in PLANNED}
    for spec in factors.all_factors():
        assert planned_family[spec.name] == spec.family, spec.name
