"""执行一次低预算真实 Worker 生成并保存脱敏验收结果。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from common.paths import EVIDENCE_DIR
from mining import FrozenContext, HttpWorkerClient, generate_worker_candidates, load_catalog
from mining_worker.protocol import WorkerGenerateRequest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        type=Path,
        default=EVIDENCE_DIR / "s4-worker-smoke.json",
    )
    parser.add_argument("--max-candidates", type=int, default=2)
    args = parser.parse_args()

    catalog = load_catalog()
    context = FrozenContext(
        data_fingerprint="0" * 64,
        field_catalog=catalog.frozen_field_mappings(),
        operator_catalog=catalog.frozen_operator_mappings(),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    request = WorkerGenerateRequest(
        request_id=f"s4-smoke-{stamp}",
        run_id="run_s4_smoke_20261024",
        round_index=1,
        hypothesis="短期价格反转可能带来可重复的截面超额收益",
        context_hash=context.context_hash,
        field_catalog_hash=context.field_catalog_hash,
        operator_catalog_hash=context.operator_catalog_hash,
        llm_model=context.llm_model,
        max_candidates=args.max_candidates,
        max_expression_depth=context.budget.max_expression_depth,
        direction=context.budget.direction,
        excluded_fields=context.budget.excluded_fields,
        fields=catalog.frozen_field_mappings(),
        operators=catalog.frozen_operator_mappings(),
    )
    client = HttpWorkerClient.from_environment()
    health = client.health()
    started = time.perf_counter()
    result = generate_worker_candidates(client, request)
    elapsed = round(time.perf_counter() - started, 3)
    if not result.accepted:
        raise RuntimeError("低成本 Worker 验收没有产生通过 S3 的候选")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "job_id": result.job_id,
        "model": request.llm_model,
        "rdagent_version": health.get("rdagent_version"),
        "worker_mode": health.get("mode"),
        "elapsed_seconds": elapsed,
        "accepted": len(result.accepted),
        "rejected": len(result.rejected),
        "candidate_names": [candidate.name for candidate in result.accepted],
        "rejection_codes": [item.error_code for item in result.rejected],
        "usage": dict(result.usage),
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
