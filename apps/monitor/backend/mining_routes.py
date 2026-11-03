"""自动因子挖掘 REST 与 SSE 路由。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Callable

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.errors import error_payload
from backend.mining_service import MiningWebService
from calculate.jsonutil import json_safe
from mining.errors import MiningError


def build_mining_router(
    service_provider: Callable[[], MiningWebService],
) -> APIRouter:
    """基于 Web 服务实例构造挖掘路由。"""

    router = APIRouter(prefix="/api/mining", tags=["mining"])

    def current_service() -> MiningWebService:
        """读取已启动的 Web 服务。"""

        service = service_provider()
        if service is None:
            raise RuntimeError("挖掘 Web 服务尚未启动")
        return service

    @router.get("/options")
    async def options():
        """返回挖掘表单默认值、限制和冻结口径。"""

        return json_safe(await asyncio.to_thread(current_service().options))

    @router.post("/runs")
    async def create_run(payload: dict):
        """创建并后台执行一次研究假设。"""

        return json_safe(
            await asyncio.to_thread(current_service().create_run, payload)
        )

    @router.post("/research-runs")
    async def create_research_run(payload: dict):
        """创建一次资料驱动运行。"""

        return json_safe(
            await asyncio.to_thread(
                current_service().create_research_run,
                payload,
            )
        )

    @router.post("/costeer-runs")
    async def create_costeer_run(payload: dict):
        """导入服务端 CoSTEER 人工提升记录并创建 V3 运行。"""

        return json_safe(
            await asyncio.to_thread(
                current_service().create_costeer_run,
                payload,
            )
        )

    @router.get("/costeer-promotions")
    async def costeer_promotions():
        """列出可导入的服务端人工提升记录。"""

        return json_safe(
            await asyncio.to_thread(current_service().costeer_promotions)
        )

    @router.get("/v3/status")
    async def v3_status():
        """返回 V3 运行、审核和发布监控摘要。"""

        return json_safe(
            await asyncio.to_thread(current_service().v3_status)
        )

    @router.get("/runs")
    async def list_runs(limit: int = 50):
        """列出近期挖掘运行。"""

        return json_safe(
            await asyncio.to_thread(current_service().list_runs, limit)
        )

    @router.get("/runs/{run_id}")
    async def run_detail(run_id: str):
        """读取运行、预算进度和候选摘要。"""

        return json_safe(
            await asyncio.to_thread(current_service().run_detail, run_id)
        )

    @router.get("/runs/{run_id}/events")
    async def list_events(run_id: str, after: int = 0, limit: int = 500):
        """读取不可变运行事件。"""

        return json_safe(
            await asyncio.to_thread(
                current_service().events,
                run_id,
                after=after,
                limit=limit,
            )
        )

    @router.get("/runs/{run_id}/stream")
    async def stream_events(request: Request, run_id: str, after: int = 0):
        """回放并持续推送事件；断开连接不取消运行。"""

        request.state.request_id = uuid.uuid4().hex
        service = current_service()
        await asyncio.to_thread(service.run_status, run_id)
        await asyncio.to_thread(service.events, run_id, after=after, limit=1)

        async def stream():
            """按序号持续读取持久化事件，并发送心跳。"""

            cursor = after
            last_heartbeat = time.monotonic()
            while True:
                if await request.is_disconnected():
                    return
                try:
                    page = await asyncio.to_thread(
                        service.events,
                        run_id,
                        after=cursor,
                        limit=1000,
                    )
                    status = await asyncio.to_thread(service.run_status, run_id)
                except MiningError as error:
                    _, payload = error_payload(error, request.state.request_id)
                    yield (
                        "event: error\ndata: "
                        + json.dumps(
                            payload,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n\n"
                    )
                    return
                for event in page["events"]:
                    cursor = int(event["sequence"])
                    yield (
                        f"id: {cursor}\nevent: mining\ndata: "
                        + json.dumps(
                            event,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n\n"
                    )
                if status["finished"]:
                    yield (
                        "event: done\ndata: "
                        + json.dumps(
                            {
                                **status,
                                "last_sequence": cursor,
                                "request_id": request.state.request_id,
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n\n"
                    )
                    return
                if time.monotonic() - last_heartbeat >= 10:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request.state.request_id,
            },
        )

    @router.post("/runs/{run_id}/cancel")
    async def cancel(run_id: str):
        """在安全阶段边界取消运行。"""

        return json_safe(
            await asyncio.to_thread(current_service().cancel, run_id)
        )

    @router.post("/runs/{run_id}/resume")
    async def resume(run_id: str):
        """显式恢复中断运行。"""

        return json_safe(
            await asyncio.to_thread(current_service().resume, run_id)
        )

    @router.get("/runs/{run_id}/candidates/{candidate_id}")
    async def candidate_detail(run_id: str, candidate_id: str):
        """读取候选定义、评估证据和审批资格。"""

        return json_safe(
            await asyncio.to_thread(
                current_service().candidate_detail,
                run_id,
                candidate_id,
            )
        )

    @router.post("/runs/{run_id}/candidates/{candidate_id}/decision")
    async def decide(run_id: str, candidate_id: str, payload: dict):
        """记录人工批准或拒绝，不执行正式发布。"""

        return json_safe(
            await asyncio.to_thread(
                current_service().decide,
                run_id,
                candidate_id,
                payload,
            )
        )

    @router.post("/runs/{run_id}/publication-batches", status_code=202)
    async def start_publication_batch(run_id: str, payload: dict):
        """整批构建并发布该运行内全部已批准候选。"""

        return json_safe(
            await asyncio.to_thread(
                current_service().start_publication_batch,
                run_id,
                payload,
            )
        )

    @router.get("/runs/{run_id}/publication-batches")
    async def list_publication_batches(run_id: str):
        """列出发布批次和状态。"""

        return json_safe(
            await asyncio.to_thread(
                current_service().publication_batches,
                run_id,
            )
        )

    @router.get("/runs/{run_id}/publication-batches/{batch_id}")
    async def publication_batch(run_id: str, batch_id: str):
        """读取单个发布批次进度。"""

        return json_safe(
            await asyncio.to_thread(
                current_service().publication_batch,
                run_id,
                batch_id,
            )
        )

    return router
