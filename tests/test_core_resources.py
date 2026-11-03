"""核心财务、事件和行业资源下载器测试。"""

from pathlib import Path

import pandas as pd
import pytest

from scripts.errors import SourceError
from scripts.sources import core_resources as core
from tools.data_audit import audit_core_resources


def test_quarter_periods_stop_at_cutoff():
    """季度列表不能包含截止日之后尚未结束的季度。"""
    assert core.quarter_periods(2025, "20261015")[-3:] == [
        "20260331", "20260630", "20260930"
    ]
    assert "20261231" not in core.quarter_periods(2025, "20261015")


def test_audit_resource_is_annual_only():
    """审计意见按年报拉取，季度参数会被服务端判为用法错误。"""
    resource = next(item for item in core.PERIOD_RESOURCES if item.api == "fina_audit")
    assert resource.annual_only is True
    assert resource.min_period == "19971231"


def test_fetch_paged_uses_offsets_and_deduplicates(monkeypatch):
    """分页应持续到短页，并清掉跨页完全重复记录。"""
    calls = []

    def fake_call(api, **params):
        calls.append(params["offset"])
        if params["offset"] == 0:
            return pd.DataFrame({"ts_code": ["A", "B"], "ann_date": ["1", "2"]})
        if params["offset"] == 2:
            return pd.DataFrame({"ts_code": ["B"], "ann_date": ["2"]})
        return pd.DataFrame()

    monkeypatch.setattr(core.tushare_src, "call", fake_call)
    result = core.fetch_paged("income_vip", {"period": "20231231"}, page_size=2)
    assert calls == [0, 2]
    assert result["ts_code"].tolist() == ["A", "B"]


def test_fetch_paged_rejects_repeated_full_page(monkeypatch):
    """服务端忽略 offset 时要报错，不能把截断数据标成完成。"""
    page = pd.DataFrame({"ts_code": ["A", "B"], "ann_date": ["1", "2"]})
    monkeypatch.setattr(core.tushare_src, "call", lambda api, **params: page.copy())
    with pytest.raises(SourceError, match="分页重复"):
        core.fetch_paged("income_vip", {"period": "20231231"}, page_size=2)


def test_disclosure_uses_single_large_page_and_checks_cap(monkeypatch):
    """披露日接口忽略 offset，因此只允许单次大页且必须检查硬上限。"""
    seen = []

    def fake_call(api, **params):
        seen.append(params)
        return pd.DataFrame({"ts_code": ["A"], "end_date": ["20231231"]})

    monkeypatch.setattr(core.tushare_src, "call", fake_call)
    assert len(core.fetch_disclosure("20231231")) == 1
    assert seen == [{"end_date": "20231231", "limit": 10000}]

    monkeypatch.setattr(
        core.tushare_src,
        "call",
        lambda api, **params: pd.DataFrame({"x": range(core.DISCLOSURE_HARD_CAP)}),
    )
    with pytest.raises(SourceError, match="命中 6000 行上限"):
        core.fetch_disclosure("20231231")


def test_period_resource_is_resumable(tmp_path, monkeypatch):
    """已完成分片在重跑时必须跳过，不再次调用远端接口。"""
    resource = core.PeriodResource("income_vip", "period", tmp_path / "income", 100)
    monkeypatch.setattr(core, "PERIOD_RESOURCES", (resource,))
    calls = []

    def fake_fetch(api, params, page_size):
        calls.append((api, params["period"]))
        return pd.DataFrame({"ts_code": ["600000.SH"], "ann_date": ["20240401"]})

    monkeypatch.setattr(core, "fetch_paged", fake_fetch)
    manifest = core.ResourceManifest(tmp_path / "manifest.json")
    first = core.fetch_period_resources(["20231231"], manifest)
    second = core.fetch_period_resources(["20231231"], manifest)

    assert first["failed"] == 0
    assert second["failed"] == 0
    assert calls == [("income_vip", "20231231")]
    assert (tmp_path / "income" / "period=20231231.parquet").exists()


def test_manifest_complete_requires_nonempty_product(tmp_path):
    """有行数的完成项若文件丢失，重跑时不能继续跳过。"""
    manifest = core.ResourceManifest(tmp_path / "manifest.json")
    path = tmp_path / "part.parquet"
    manifest.update("x:1", status="complete", rows=1, path=str(path))
    assert not manifest.is_complete("x:1", path)
    manifest.update("x:0", status="complete", rows=0, path=None)
    assert manifest.is_complete("x:0", tmp_path / "missing.parquet")


def test_sw_history_rejects_cross_industry_rows(tmp_path, monkeypatch):
    """单个三级行业分片不能混入其它行业成员。"""
    classifications = {
        "L1": pd.DataFrame({"index_code": ["801010.SI"], "level": ["L1"]}),
        "L2": pd.DataFrame({"index_code": ["801011.SI"], "level": ["L2"]}),
        "L3": pd.DataFrame(
            {"index_code": ["850111.SI"], "level": ["L3"], "is_pub": ["1"]}
        ),
    }

    def fake_call(api, **params):
        if api == "index_classify":
            frame = classifications[params["level"]].copy()
            frame["industry_name"] = "测试"
            frame["parent_code"] = "0"
            frame["industry_code"] = frame["index_code"]
            frame["src"] = params["src"]
            return frame
        return pd.DataFrame(
            {
                "index_code": ["850112.SI"], "con_code": ["000001.SZ"],
                "in_date": ["20200101"], "out_date": [None], "is_new": ["Y"],
            }
        )

    monkeypatch.setattr(core.tushare_src, "call", fake_call)
    industry = tmp_path / "raw_industry" / "sw2021"
    monkeypatch.setattr(core, "INDUSTRY_DIRS", {"SW2021": industry})
    monkeypatch.setattr(core, "TUSHARE_DIR", tmp_path)
    manifest = core.ResourceManifest(tmp_path / "manifest.json")
    result = core.fetch_sw_history(manifest, "SW2021")
    assert result["failures"] == ["index_member:850111.SI"]
    assert manifest.payload["items"]["index_member:850111.SI"]["status"] == "failed"


def test_sw_history_downloads_only_published_l3_and_resumes(tmp_path, monkeypatch):
    """正式历史链只拉有效三级行业，并能从 manifest 断点续跑。"""
    classifications = {
        "L1": pd.DataFrame(
            {"index_code": ["801010.SI"], "level": ["L1"], "is_pub": ["1"]}
        ),
        "L2": pd.DataFrame(
            {"index_code": ["801011.SI"], "level": ["L2"], "is_pub": ["1"]}
        ),
        "L3": pd.DataFrame(
            {
                "index_code": ["850111.SI", "850112.SI"],
                "level": ["L3", "L3"],
                "is_pub": ["1", "0"],
            }
        ),
    }
    calls = []

    def fake_call(api, **params):
        if api == "index_classify":
            frame = classifications[params["level"]].copy()
            frame["industry_name"] = "测试"
            frame["parent_code"] = "0"
            frame["industry_code"] = frame["index_code"]
            frame["src"] = params["src"]
            return frame
        calls.append((api, params["index_code"]))
        return pd.DataFrame(
            {
                "index_code": [params["index_code"]], "con_code": ["000001.SZ"],
                "in_date": ["20200101"], "out_date": [None], "is_new": ["Y"],
            }
        )

    monkeypatch.setattr(core.tushare_src, "call", fake_call)
    industry = tmp_path / "raw_industry" / "sw2021"
    monkeypatch.setattr(core, "INDUSTRY_DIRS", {"SW2021": industry})
    monkeypatch.setattr(core, "TUSHARE_DIR", tmp_path)
    manifest = core.ResourceManifest(tmp_path / "manifest.json")
    core.fetch_sw_history(manifest, "SW2021")
    core.fetch_sw_history(manifest, "SW2021")
    assert calls == [("index_member", "850111.SI")]
    assert (industry / "history_members" / "l3_code=850111.SI.parquet").exists()


def test_sw_history_marks_empty_response_without_complete_file(tmp_path, monkeypatch):
    """空行业成员不能被标记成有产物的 complete。"""
    classifications = {
        "L1": pd.DataFrame({"index_code": ["801010.SI"], "level": ["L1"]}),
        "L2": pd.DataFrame({"index_code": ["801011.SI"], "level": ["L2"]}),
        "L3": pd.DataFrame(
            {"index_code": ["850111.SI"], "level": ["L3"], "is_pub": ["1"]}
        ),
    }

    def fake_call(api, **params):
        if api == "index_classify":
            frame = classifications[params["level"]].copy()
            frame["industry_name"] = "测试"
            frame["parent_code"] = "0"
            frame["industry_code"] = frame["index_code"]
            frame["src"] = params["src"]
            return frame
        return pd.DataFrame()

    industry = tmp_path / "raw_industry" / "sw2021"
    monkeypatch.setattr(core.tushare_src, "call", fake_call)
    monkeypatch.setattr(core, "INDUSTRY_DIRS", {"SW2021": industry})
    monkeypatch.setattr(core, "TUSHARE_DIR", tmp_path)
    manifest = core.ResourceManifest(tmp_path / "manifest.json")
    result = core.fetch_sw_history(manifest, "SW2021")
    assert result["empty"] == ["index_member:850111.SI"]
    assert manifest.payload["items"]["index_member:850111.SI"]["status"] == "empty"
    assert not (industry / "history_members" / "l3_code=850111.SI.parquet").exists()


def test_sw_repair_downloads_stock_history_and_resumes(tmp_path, monkeypatch):
    """逐股修复按 ts_code 落盘，并与行业分片使用不同 manifest key。"""
    calls = []

    def fake_call(api, **params):
        calls.append(params)
        return pd.DataFrame(
            {
                "index_code": ["850111.SI", "801010.SI"],
                "con_code": ["000001.SZ", "000001.SZ"],
                "in_date": ["20150101", "20150101"],
                "out_date": [None, None],
                "is_new": ["Y", "Y"],
            }
        )

    monkeypatch.setattr(core.tushare_src, "call", fake_call)
    monkeypatch.setattr(core, "INDUSTRY_REPAIR_DIR", tmp_path / "repair")
    monkeypatch.setattr(core, "TUSHARE_DIR", tmp_path)
    manifest = core.ResourceManifest(tmp_path / "manifest.json")
    first = core.fetch_sw_repair(["000001.SZ"], manifest)
    second = core.fetch_sw_repair(["000001.SZ"], manifest)
    assert first["failures"] == [] and second["failures"] == []
    assert calls == [{"ts_code": "000001.SZ"}]
    assert (tmp_path / "repair" / "ts_code=000001.SZ.parquet").exists()
    assert manifest.payload["items"]["index_member_ts_code:000001.SZ"]["status"] == "complete"


def test_dividend_batches_keep_all_stock_rows(tmp_path, monkeypatch):
    """分红批次应保留每只股票的记录并写入完成状态。"""
    monkeypatch.setattr(core, "EVENT_DIR", tmp_path / "events")
    monkeypatch.setattr(core, "TUSHARE_DIR", tmp_path)
    monkeypatch.setattr(core, "_stock_codes", lambda: ["000001.SZ", "600000.SH"])
    monkeypatch.setattr(
        core,
        "_dividend_worker",
        lambda code: (code, pd.DataFrame({"ts_code": [code], "ann_date": ["20240101"]}), ""),
    )
    manifest = core.ResourceManifest(tmp_path / "manifest.json")
    result = core.fetch_dividends(manifest)
    assert result["failures"] == []
    rows = []
    for path in (tmp_path / "events" / "dividend").glob("bucket=*.parquet"):
        rows.extend(pd.read_parquet(path)["ts_code"].tolist())
    assert set(rows) == {"000001.SZ", "600000.SH"}


def test_dividend_bucket_is_stable():
    """同一股票始终落同一两位十六进制桶。"""
    assert core.dividend_bucket("600000.SH") == core.dividend_bucket("600000.SH")
    assert len(core.dividend_bucket("600000.SH")) == 2


def test_stock_codes_skip_legacy_nonstandard_codes(tmp_path, monkeypatch):
    """T600018.SH 这类历史代码不能送进分红接口。"""
    monkeypatch.setattr(core, "TUSHARE_DIR", tmp_path)
    pd.DataFrame(
        {"ts_code": ["600018.SH", "T600018.SH", "000001.SZ", "920000.BJ"]}
    ).to_parquet(tmp_path / "stock_basic_all.parquet", index=False)
    assert core._stock_codes() == ["000001.SZ", "600018.SH", "920000.BJ"]


def test_core_paths_are_under_tushare_directory():
    """正式目录必须留在 data/source/tushare 下，避免散落到项目其它位置。"""
    assert Path(core.FINANCIAL_DIR).is_relative_to(core.TUSHARE_DIR)
    assert Path(core.EVENT_DIR).is_relative_to(core.TUSHARE_DIR)
    assert Path(core.INDUSTRY_DIR).is_relative_to(core.TUSHARE_DIR)
    assert Path(core.INDUSTRY_HISTORY_DIR).is_relative_to(core.INDUSTRY_DIR)


def test_audit_core_resources_checks_rows_and_duplicates(tmp_path, monkeypatch):
    """审计脚本应核对文件行数并统计完全重复行。"""
    part = tmp_path / "part.parquet"
    pd.DataFrame({"x": [1, 1]}).to_parquet(part, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"items":{"demo:1":{"api":"demo","status":"complete","rows":2,'
        '"path":"part.parquet","min_date":"20200101","max_date":"20200102"}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_core_resources, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(audit_core_resources, "TUSHARE_DIR", tmp_path)
    result = audit_core_resources.audit()
    assert result["failures"] == []
    assert result["resources"]["demo"]["exact_duplicates"] == 1


def test_audit_rejects_empty_recent_financial_partition(tmp_path, monkeypatch):
    """2005 年后的四张财务主表不应把瞬时空响应记成完成。"""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"items":{"income_vip:20240331":{"api":"income_vip",'
        '"partition":"20240331","status":"complete","rows":0,"path":null}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_core_resources, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(audit_core_resources, "TUSHARE_DIR", tmp_path)
    result = audit_core_resources.audit()
    assert result["failures"][0]["key"] == "income_vip:20240331"
