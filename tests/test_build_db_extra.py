"""指数扩展表建库测试。"""

import duckdb
import pandas as pd
import pytest

from scripts.build_db import build
from scripts.errors import DbWriteError


def test_build_index_basic_accepts_missing_optional_columns(tmp_path, monkeypatch):
    """中转服务不返回 fullname 等可选列时仍应正常建表。"""
    frame = pd.DataFrame(
        {
            "ts_code": ["H00300.CSI"],
            "name": ["300收益"],
            "market": ["CSI"],
            "publisher": ["中证指数有限公司"],
            "category": ["规模指数"],
            "base_date": ["20041231"],
            "base_point": [1000.0],
            "list_date": ["20050408"],
        }
    )
    frame.to_parquet(tmp_path / "index_basic.parquet", index=False)
    monkeypatch.setattr(build, "TUSHARE_DIR", tmp_path)
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA ref")
    build.build_index_basic(conn)
    row = conn.execute(
        "SELECT index_code, index_name, fullname, index_type, exp_date FROM ref.index_basic"
    ).fetchone()
    assert row == ("H00300.CSI", "300收益", None, None, None)


def test_build_sw_industry_creates_point_in_time_member_table(tmp_path, monkeypatch):
    """两套分类按 2021-12-13 切换，SW2021 缺失时沿用最近 SW2014 行业。"""
    root = tmp_path / "raw_industry"
    sw2014 = root / "sw2014"
    sw2021 = root / "sw2021"
    for directory, classification, l1, l2, l3, names in (
        (sw2014, "SW2014", "801010.SI", "801016.SI", "850111.SI", ("农林牧渔", "种植业", "种子")),
        (sw2021, "SW2021", "801020.SI", "801026.SI", "850211.SI", ("基础化工", "化学制品", "涂料")),
    ):
        (directory / "history_members").mkdir(parents=True)
        (directory / "members").mkdir()
        pd.DataFrame(
            {
                "index_code": [l1, l2, l3],
                "industry_name": list(names),
                "parent_code": ["0", "110000", "110100"],
                "level": ["L1", "L2", "L3"],
                "industry_code": ["110000", "110100", "110101"],
                "is_pub": ["1", "1", "1"],
                "src": [classification] * 3,
            }
        ).to_parquet(directory / "classify.parquet", index=False)
    pd.DataFrame(
        {
            "index_code": ["850111.SI", "850111.SI"],
            "con_code": ["000001.SZ", "000003.SZ"],
            "in_date": ["20150101", "20150101"],
            "out_date": ["20211210", None],
            "is_new": ["N", "Y"],
        }
    ).to_parquet(sw2014 / "history_members" / "l3_code=850111.SI.parquet", index=False)
    pd.DataFrame(
        {
            "index_code": ["850211.SI"],
            "con_code": ["000001.SZ"],
            "in_date": ["20211213"],
            "out_date": [None],
            "is_new": ["Y"],
        }
    ).to_parquet(sw2021 / "history_members" / "l3_code=850211.SI.parquet", index=False)
    pd.DataFrame(
        {
            "l3_code": ["850211.SI"], "ts_code": ["000002.SZ"],
            "in_date": ["20220101"], "out_date": [None], "is_new": ["Y"],
        }
    ).to_parquet(sw2021 / "members" / "l3_code=850211.SI.parquet", index=False)
    monkeypatch.setattr(build, "TUSHARE_DIR", tmp_path)
    monkeypatch.setattr(
        build, "SW_INDUSTRY_DIRS", {"SW2014": sw2014, "SW2021": sw2021}
    )
    monkeypatch.setattr(build, "SW_INDUSTRY_REPAIR_DIR", root / "repair")
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA ref")
    conn.execute("CREATE SCHEMA meta")
    build.build_sw_industry(conn)
    rows = conn.execute(
        "SELECT stock_code, classification, l1_code, in_date, out_date, carry_forward "
        "FROM ref.sw_industry_member ORDER BY stock_code, in_date"
    ).fetchall()
    assert rows == [
        (
            "SZ000001", "SW2014", "801010.SI",
            __import__("datetime").date(2015, 1, 1),
            __import__("datetime").date(2021, 12, 10), False,
        ),
        (
                "SZ000001", "SW2014", "801010.SI",
                __import__("datetime").date(2021, 12, 11),
                __import__("datetime").date(2021, 12, 12), True,
            ),
            (
            "SZ000001", "SW2021", "801020.SI",
            __import__("datetime").date(2021, 12, 13), None, False,
        ),
        (
            "SZ000002", "SW2021", "801020.SI",
            __import__("datetime").date(2022, 1, 1), None, False,
        ),
        (
            "SZ000003", "SW2014", "801010.SI",
            __import__("datetime").date(2015, 1, 1),
            __import__("datetime").date(2021, 12, 12), False,
        ),
        (
            "SZ000003", "SW2014", "801010.SI",
            __import__("datetime").date(2021, 12, 13), None, True,
        ),
    ]
    manifest = conn.execute(
        "SELECT classifications, switch_date, repair_rows, carry_forward_rows, "
        "carry_forward_stocks, history_rows, current_snapshot_rows "
        "FROM meta.sw_industry_manifest"
    ).fetchone()
    assert manifest[0] == "SW2014,SW2021"
    assert manifest[1] == __import__("datetime").date(2021, 12, 13)
    assert manifest[2] == 0
    assert manifest[3] == 2
    assert manifest[4] == 2
    assert manifest[5] == 3
    assert manifest[6] == 1


def test_sw_industry_rejects_cross_l1_overlap():
    """同一股票同日命中两个一级行业时不能继续建库。"""
    frame = pd.DataFrame(
        {
            "stock_code": ["SZ000001", "SZ000001"],
            "l1_code": ["A", "B"],
            "l3_code": ["A1", "B1"],
            "in_date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
            "out_date": pd.to_datetime(["2024-01-31", None]),
        }
    )
    with pytest.raises(DbWriteError, match="区间冲突"):
        build._validate_sw_industry_intervals(frame)


def test_sw_industry_canonicalization_prefers_finite_history_then_latest_start():
    """同日起历史记录先于开放快照，后开始的记录覆盖较早长区间。"""
    frame = pd.DataFrame(
        {
            "stock_code": ["SZ000001"] * 3,
            "l1_code": ["CURRENT", "OLD", "NEW"],
            "l2_code": ["C2", "O2", "N2"],
            "l3_code": ["C3", "O3", "N3"],
            "in_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2021-01-01"]),
            "out_date": pd.to_datetime([None, "2020-06-30", "2021-06-30"]),
        }
    )
    result, resolved, gap_segments, gap_days = build._canonicalize_sw_industry_intervals(frame)
    assert resolved == 2
    assert gap_segments == 0
    assert gap_days == 0
    assert result[["l1_code", "in_date", "out_date"]].to_dict("records") == [
        {
            "l1_code": "OLD",
            "in_date": pd.Timestamp("2020-01-01"),
            "out_date": pd.Timestamp("2020-06-30"),
        },
        {
            "l1_code": "CURRENT",
            "in_date": pd.Timestamp("2020-07-01"),
            "out_date": pd.Timestamp("2020-12-31"),
        },
        {
            "l1_code": "NEW",
            "in_date": pd.Timestamp("2021-01-01"),
            "out_date": pd.Timestamp("2021-06-30"),
        },
        {
            "l1_code": "CURRENT",
            "in_date": pd.Timestamp("2021-07-01"),
            "out_date": pd.NaT,
        },
    ]


def test_sw_industry_canonicalization_fills_bounded_same_industry_gap():
    """前后行业完全相同时填平中间空档，不改变行业归属。"""
    frame = pd.DataFrame(
        {
            "stock_code": ["SZ000001", "SZ000001"],
            "l1_code": ["A", "A"], "l2_code": ["A2", "A2"], "l3_code": ["A3", "A3"],
            "in_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "out_date": pd.to_datetime(["2020-01-15", None]),
            "source_api": ["index_member", "index_member_all"],
        }
    )
    result, _, gap_segments, gap_days = build._canonicalize_sw_industry_intervals(frame)
    assert len(result) == 1
    assert result.iloc[0]["in_date"] == pd.Timestamp("2020-01-01")
    assert pd.isna(result.iloc[0]["out_date"])
    assert result.iloc[0]["source_api"] == "same_industry_gap_fill"
    assert (gap_segments, gap_days) == (1, 16)


def test_sw_industry_unique_name_alias_maps_legacy_l3_code():
    """SW2021 目录缺旧代码时，可用唯一三级行业名称映射。"""
    hierarchy = pd.DataFrame(
        {
            "classification": ["SW2021", "SW2021"],
            "l3_code": ["850401.SI", "850402.SI"],
            "l3_name": ["特钢Ⅲ", "普钢Ⅲ"],
        }
    )
    assert build._resolve_sw_l3_code(
        "850412.SI", "SW2021", hierarchy, {"850412.SI": "申万特钢Ⅲ"}
    ) == "850401.SI"
    assert build._resolve_sw_l3_code(
        "850499.SI", "SW2021", hierarchy, {"850499.SI": "申万特钢"}
    ) == "850401.SI"


def test_sw_industry_canonicalization_keeps_classifications_separate():
    """同一股票在新旧分类下各有一条区间时不能被当成来源重叠。"""
    frame = pd.DataFrame(
        {
            "stock_code": ["SZ000001", "SZ000001"],
            "classification": ["SW2014", "SW2021"],
            "l1_code": ["A", "B"], "l2_code": ["A2", "B2"], "l3_code": ["A3", "B3"],
            "in_date": pd.to_datetime(["2020-01-01", "2021-12-13"]),
            "out_date": pd.to_datetime([None, None]),
        }
    )
    result, resolved, _, _ = build._canonicalize_sw_industry_intervals(frame)
    assert resolved == 0
    assert result[["classification", "l1_code"]].to_dict("records") == [
        {"classification": "SW2014", "l1_code": "A"},
        {"classification": "SW2021", "l1_code": "B"},
    ]


def test_sw_industry_carry_forward_stops_when_sw2021_starts():
    """SW2021 出现后不再沿用 SW2014，且切换前后区间不重叠。"""
    members = pd.DataFrame(
        {
            "stock_code": ["SZ000001", "SZ000001"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "classification": ["SW2014", "SW2021"],
            "l1_code": ["A", "B"], "l1_name": ["甲", "乙"],
            "l2_code": ["A2", "B2"], "l2_name": ["甲2", "乙2"],
            "l3_code": ["A3", "B3"], "l3_name": ["甲3", "乙3"],
            "in_date": pd.to_datetime(["2020-01-01", "2022-07-01"]),
            "out_date": pd.to_datetime([None, None]),
            "source_in_date": pd.to_datetime(["2020-01-01", "2022-07-01"]),
            "source_out_date": pd.to_datetime([None, None]),
            "source_l3_code": ["A3", "B3"],
            "source_api": ["index_member", "index_member"],
            "is_new": ["Y", "Y"],
        }
    )
    result = build._apply_sw_version_windows(members).sort_values("in_date")
    assert result[["classification", "carry_forward", "in_date", "out_date"]].to_dict(
        "records"
    ) == [
        {
            "classification": "SW2014", "carry_forward": False,
            "in_date": pd.Timestamp("2020-01-01"),
            "out_date": pd.Timestamp("2021-12-12"),
        },
        {
            "classification": "SW2014", "carry_forward": True,
            "in_date": pd.Timestamp("2021-12-13"),
            "out_date": pd.Timestamp("2022-06-30"),
        },
        {
            "classification": "SW2021", "carry_forward": False,
            "in_date": pd.Timestamp("2022-07-01"),
            "out_date": pd.NaT,
        },
    ]


def test_sw_industry_uses_early_sw2021_when_no_sw2014_record():
    """科创板等无 SW2014 记录的股票，切换日前可使用早期 SW2021 区间。"""
    members = pd.DataFrame(
        {
            "stock_code": ["SH688001"],
            "ts_code": ["688001.SH"],
            "classification": ["SW2021"],
            "l1_code": ["A"], "l1_name": ["甲"],
            "l2_code": ["A2"], "l2_name": ["甲2"],
            "l3_code": ["A3"], "l3_name": ["甲3"],
            "in_date": pd.to_datetime(["2020-01-01"]),
            "out_date": pd.to_datetime([None]),
            "source_in_date": pd.to_datetime(["2020-01-01"]),
            "source_out_date": pd.to_datetime([None]),
            "source_l3_code": ["A3"], "source_api": ["index_member_all"],
            "is_new": ["Y"],
        }
    )
    result = build._apply_sw_version_windows(members)
    assert result[["classification", "carry_forward", "source_api", "in_date"]].to_dict(
        "records"
    ) == [
        {
            "classification": "SW2021", "carry_forward": False,
            "source_api": "early_sw2021", "in_date": pd.Timestamp("2020-01-01"),
        },
        {
            "classification": "SW2021", "carry_forward": False,
            "source_api": "index_member_all", "in_date": pd.Timestamp("2021-12-13"),
        },
    ]


def test_sw_industry_backfills_earliest_record_to_list_date():
    """最早行业记录晚于上市日时，向前补到上市日并标记来源。"""
    members = pd.DataFrame(
        {
            "stock_code": ["SH688005"],
            "ts_code": ["688005.SH"],
            "classification": ["SW2021"],
            "l1_code": ["A"], "l1_name": ["甲"],
            "l2_code": ["A2"], "l2_name": ["甲2"],
            "l3_code": ["A3"], "l3_name": ["甲3"],
            "in_date": pd.to_datetime(["2021-07-30"]),
            "out_date": pd.to_datetime([None]),
            "source_in_date": pd.to_datetime(["2021-07-30"]),
            "source_out_date": pd.to_datetime([None]),
            "source_l3_code": ["A3"], "source_api": ["index_member_all"],
            "is_new": ["Y"],
        }
    )
    result = build._apply_sw_version_windows(
        members, {"SH688005": pd.Timestamp("2019-07-22")}
    )
    assert result.iloc[0]["in_date"] == pd.Timestamp("2019-07-22")
    assert result.iloc[0]["out_date"] == pd.Timestamp("2021-07-29")
    assert result.iloc[0]["source_api"] == "backfill_earliest_sw2021"
    assert bool(result.iloc[0]["carry_forward"]) is True
