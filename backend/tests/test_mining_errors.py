"""监测站对挖掘错误码和 HTTP 状态的转换测试。"""

from __future__ import annotations

import pytest

from backend.errors import error_payload
from mining.errors import (
    ApprovalConflictError,
    HoldoutError,
    MiningBusyError,
    MiningContractError,
    MiningError,
    MiningEvaluationError,
    MiningNotFoundError,
    MiningPublicationError,
    MiningRecoveryError,
    MiningStateError,
    MiningStorageError,
    MiningWorkerError,
)


def test_mining_errors_preserve_code_status_and_request_id():
    """挖掘错误不能落入通用 500。"""

    status, payload = error_payload(
        MiningContractError("研究假设不合法"),
        "request-123",
    )
    assert status == 400
    assert payload == {
        "code": "MINING-101",
        "message": "研究假设不合法",
        "request_id": "request-123",
    }

    status, payload = error_payload(
        ApprovalConflictError("候选内容已变化"),
        "request-456",
    )
    assert status == 409
    assert payload["code"] == "MINING-108"


@pytest.mark.parametrize(
    "error,expected",
    [
        (MiningContractError("a"), 400),
        (MiningBusyError(), 429),
        (MiningWorkerError("a"), 503),
        (MiningEvaluationError("a"), 422),
        (HoldoutError("a"), 409),
        (MiningPublicationError("a"), 500),
        (MiningNotFoundError("a"), 404),
        (MiningStateError("a"), 409),
        (MiningStorageError("a"), 503),
        (MiningRecoveryError("a"), 409),
    ],
)
def test_all_mining_error_families_keep_http_mapping(error, expected):
    """每个错误族都有稳定 HTTP 状态。"""

    status, payload = error_payload(error, "request-id")
    assert status == expected
    assert isinstance(error, MiningError)
    assert payload["code"].startswith("MINING-")
