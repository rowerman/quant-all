"""V2 搜索实验 API 分流和脱敏测试。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.mining_search_service import MiningSearchWebService
from backend.mining_service import MiningWebService
from mining.search_context import FrozenSearchContext
from mining.search_progress import progress_for_search
from mining.search_run import SearchRunRequest
from mining.storage import MiningStore
from mining.storage_v2 import MiningV2Store


class StubSearchOrchestrator:
    def __init__(self, store: MiningV2Store) -> None:
        self.store = store
        self.closed = False

    def create_run(self, request, context, *, background=True):
        return self.store.create_run(request, context)

    def progress(self, run_id):
        return progress_for_search(self.store, run_id)

    def cancel(self, run_id):
        return self.store.transition_search_run(
            run_id,
            "cancelled",
            event_key="run_cancelled",
            stage="queued",
            stop_reason="用户取消",
        )

    def resume(self, run_id, *, background=True):
        return self.store.search_control_state(run_id)

    def close(self):
        self.closed = True


def _services(tmp_path: Path):
    v1_store = MiningStore(
        tmp_path / "factor_mining_v1.duckdb",
        tmp_path / "v1",
    )
    v2_store = MiningV2Store(
        tmp_path / "factor_mining_v2.duckdb",
        tmp_path / "v2",
    )
    search = MiningSearchWebService(
        store=v2_store,
        orchestrator=StubSearchOrchestrator(v2_store),
    )
    service = MiningWebService(
        store=v1_store,
        defaults=None,
        snapshot_check=lambda *args, **kwargs: None,
        search_service=search,
    )
    return service, v2_store


def test_v2_api_creates_lists_and_reads_safe_detail(tmp_path: Path):
    service, _store = _services(tmp_path)
    app = create_app(mining_service=service)
    with TestClient(app) as client:
        options = client.get("/api/mining/options").json()
        assert options["engine"] == "v2"
        assert options["search_v2"]["enabled"] is True

        created = client.post(
            "/api/mining/runs",
            json={
                "hypothesis": "短期反转可能带来开发期超额收益",
                "idempotency_key": "web-v2-search-0001",
                "budget": {
                    **options["search_v2"]["defaults"],
                    "max_search_turns": 2,
                },
            },
        )
        assert created.status_code == 200
        detail = created.json()
        assert detail["run"]["engine"] == "v2"
        assert detail["progress"]["status"] == "queued"
        assert detail["hypotheses"] == []
        assert detail["turns"] == []
        assert "constraints" not in detail["run"]

        listed = client.get("/api/mining/runs").json()
        assert listed[0]["run_id"] == detail["run"]["run_id"]
        assert listed[0]["engine"] == "v2"

        run_id = detail["run"]["run_id"]
        events = client.get(f"/api/mining/runs/{run_id}/events").json()
        assert events["events"][0]["event_type"] == "run_created"
        status = client.post(f"/api/mining/runs/{run_id}/cancel").json()
        assert status["status"] == "cancelled"

        candidate = client.get(
            f"/api/mining/runs/{run_id}/candidates/unknown"
        )
        assert candidate.status_code == 400


def test_v2_detail_does_not_require_v1_run(tmp_path: Path):
    service, store = _services(tmp_path)
    context = FrozenSearchContext(data_fingerprint="a" * 64)
    run = store.create_run(
        SearchRunRequest(
            "v2-detail-only-0001",
            "短期反转可能带来开发期超额收益",
        ),
        context,
    )

    detail = service.run_detail(run["run_id"])

    assert detail["run"]["run_id"] == run["run_id"]
    assert "payload" not in detail["run"]["context"]
    assert detail["attempts"]["proposed"] == 0
