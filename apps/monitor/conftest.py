"""把前后端包与仓库根加入导入路径。

前后端统一放在 ``apps/monitor/`` 下，``import backend`` 需要本目录，
而 ``backend`` 又依赖仓库根的 ``calculate``/``quant``/``backtest``：
放在这里的 conftest 会在收集 ``apps/monitor/**`` 的测试时自动生效。
"""

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
for _path in (_PROJECT_ROOT, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture(autouse=True)
def isolated_default_mining_runtime(monkeypatch, tmp_path):
    """未显式注入时，让默认监测站应用使用临时候选库。"""

    import backend.app as app_module
    from backend.mining_service import MiningWebService
    from mining.contract_requests import MiningConstraints
    from mining.orchestrator import MiningOrchestrator
    from mining.storage import MiningStore

    def factory():
        store = MiningStore(
            tmp_path / "factor_mining.duckdb",
            tmp_path / "factor_mining",
        )
        return MiningWebService(
            store=store,
            orchestrator=MiningOrchestrator(store=store),
            defaults=MiningConstraints(
                max_rounds=1,
                max_candidates=4,
                full_eval_top_k=2,
                max_llm_calls=4,
                min_quality_checks_for_failure_stop=4,
            ),
            reviewer="test-reviewer",
            snapshot_check=lambda _context, **_kwargs: None,
        )

    monkeypatch.setattr(app_module, "MiningWebService", factory)
