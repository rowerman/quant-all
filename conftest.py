"""项目级 pytest 分类规则。

默认测试只运行快速、隔离的单元测试。真实数据、原生扩展、容器和外部服务测试
通过显式标记运行，避免测试结果依赖某台开发机是否碰巧保留了数据或镜像。
"""

from __future__ import annotations

from pathlib import Path

import pytest


_NATIVE_TESTS = {
    "test_alpha_dataset.py",
    "test_alpha_preprocess.py",
    "test_cpp_binning.py",
    "test_cpp_daily_kernels.py",
    "test_cpp_model_preprocess.py",
    "test_cpp_native.py",
    "test_cpp_ols.py",
    "test_cpp_standardize.py",
    "test_daily_native_routing.py",
    "test_eval_correlation.py",
    "test_eval_exposure.py",
    "test_mining_evaluation_correlation.py",
    "test_modeling_preprocess_baseline.py",
    "test_native_kernels.py",
}

_REAL_DATA_TESTS = {
    "test_alpha_model_card.py",
    "test_alpha_registry.py",
    "test_app.py",
    "test_backtest_api.py",
    "test_consistency.py",
    "test_data.py",
    "test_factor_rank_provider.py",
    "test_factor_build.py",
    "test_factor_catalog.py",
    "test_factor_loaders.py",
    "test_factor_validate.py",
    "test_ml_rank_provider.py",
    "test_mining_batch_materialization.py",
    "test_mining_evaluation_consistency.py",
    "test_mining_orchestrator_real.py",
    "test_mining_publication.py",
    "test_mining_search_downstream.py",
    "test_mining_v3_downstream.py",
    "test_mining_v3_joint_api.py",
    "test_orders.py",
    "test_snapshot.py",
}

_DOCKER_TESTS = {
    "test_mining_sandbox.py",
    "test_mining_sandbox_research.py",
    "test_mining_sandbox_security.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """按测试模块补资源标记，不改变测试实现。"""

    for item in items:
        path = Path(str(item.path))
        name = path.name
        relative = path.as_posix()
        if name.startswith(("test_alpha_", "test_mining_")):
            item.add_marker(pytest.mark.integration)
        if "apps/monitor/backend/tests/" in relative:
            item.add_marker(pytest.mark.integration)
        if name in {"test_chunked.py", "test_engine.py"}:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        if "research/evaluation/calculate/tests/" in relative:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.real_data)
        if name in _REAL_DATA_TESTS:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.real_data)
        if name in _NATIVE_TESTS:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.native)
        if name in _DOCKER_TESTS:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.docker)
        if name == "test_mining_v3_joint_api.py":
            item.add_marker(pytest.mark.e2e)

