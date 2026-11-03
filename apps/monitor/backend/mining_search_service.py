"""监测站 V2 证据驱动搜索用例服务。"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from backend.mining_search_detail import build_search_run_detail
from backend.mining_search_review import MiningSearchReviewAdapter
from calculate.correlation import available_factors
from common.paths import (
    DB_FILE,
    FACTOR_DB_FILE,
    FACTOR_DIR,
    MINING_DB_FILE,
    MINING_DATA_DIR,
    PROJECT_ROOT,
    RESEARCH_MATERIALS_DIR,
)
from mining.analysis_tool_gateway import AnalysisToolGateway
from mining.catalog import load_catalog
from mining.errors import MiningContractError, MiningNotFoundError
from mining.free_exploration_contracts import EXPLORATION_SEED_MAX
from mining.logging_conf import format_event, get_mining_logger
from mining.publication_service import MiningPublicationService
from mining.research_material_io import (
    build_research_material_manifest,
    discover_research_material_paths,
)
from mining.research_material_parser import RESEARCH_PDF_PARSER_VERSION
from mining.research_factor_screening_contracts import ResearchConversionLimits
from mining.research_run_contracts import ResearchRunInput
from mining.search_budget import SearchBudget
from mining.search_context import FrozenSearchContext, load_search_config
from mining.search_orchestrator import MiningSearchOrchestrator
from mining.search_run import SearchRunRequest
from mining.search_contract_types import (
    FREE_EXPLORATION_INPUT_MODE,
    HYPOTHESIS_INPUT_MODE,
)
from mining.snapshot import (
    DEFAULT_DATA_MANIFEST,
    data_snapshot_fingerprint,
)
from mining.storage_v2 import MiningV2Store
from mining_worker.worker_config import load_worker_config

_LOGGER = get_mining_logger("mining.search.web")
_CREATE_FIELDS = {
    "hypothesis",
    "idempotency_key",
    "llm_profile",
    "budget",
    "input_mode",
    "exploration_seed",
}
_RESEARCH_CREATE_FIELDS = {
    "material_path",
    "idempotency_key",
    "llm_profile",
    "budget",
}
_END_STATUSES = {
    "review_ready",
    "failed",
    "cancelled",
    "stale",
}


class MiningSearchWebService:
    """为监测站提供 V2 运行和证据视图。"""

    def __init__(
        self,
        *,
        store: MiningV2Store | None = None,
        orchestrator: MiningSearchOrchestrator | None = None,
        publication_service: MiningPublicationService | None = None,
        db_file: Path | str = DB_FILE,
        factor_dir: Path | str = FACTOR_DIR,
        catalog_file: Path | str = FACTOR_DB_FILE,
        manifest_file: Path | str = DEFAULT_DATA_MANIFEST,
        snapshot_check: Callable[..., None] | None = None,
        material_root: Path | str | None = None,
    ) -> None:
        self.store = store or MiningV2Store(MINING_DB_FILE, MINING_DATA_DIR)
        tools_enabled = os.environ.get(
            "MINING_ANALYSIS_TOOLS_ENABLED",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.material_root = Path(
            material_root or RESEARCH_MATERIALS_DIR
        )
        self.orchestrator = orchestrator or MiningSearchOrchestrator(
            store=self.store,
            analysis_tool_gateway=(
                AnalysisToolGateway(store=self.store)
                if tools_enabled
                else None
            ),
            db_file=db_file,
            factor_dir=factor_dir,
            manifest_file=manifest_file,
            research_material_root=self.material_root,
        )
        self.db_file = Path(db_file)
        self.factor_dir = Path(factor_dir)
        self.catalog_file = Path(catalog_file)
        self.manifest_file = Path(manifest_file)
        self.review = MiningSearchReviewAdapter(
            store=self.store,
            publication_service=publication_service,
            db_file=self.db_file,
            factor_dir=self.factor_dir,
            catalog_file=self.catalog_file,
            manifest_file=self.manifest_file,
            snapshot_check=snapshot_check,
        )
        self.publication_service = self.review.publication_service
        self._initialized = False

    def initialize(self) -> None:
        """初始化 V2 存储并把遗留活动运行标记中断。"""

        if self._initialized:
            return
        interrupted = self.store.interrupt_active_search_runs()
        self.publication_service.initialize()
        self._initialized = True
        _LOGGER.info(
            format_event(
                {
                    "operation": "initialize_search_web",
                    "interrupted_runs": len(interrupted),
                }
            )
        )

    def close(self) -> None:
        """关闭 V2 后台执行器。"""

        self.publication_service.close()
        self.orchestrator.close()

    def options(self) -> dict[str, Any]:
        """返回 V2 预算、质量策略和启用状态。"""

        config = load_search_config()
        return {
            "defaults": config.budget.to_dict(),
            "control_policy": config.control_policy.to_dict(),
            "windows": config.windows.to_dict(),
            "input_modes": [
                {
                    "value": HYPOTHESIS_INPUT_MODE,
                    "label": "手工研究假设",
                },
                {
                    "value": FREE_EXPLORATION_INPUT_MODE,
                    "label": "自由探索",
                },
            ],
            "exploration": {
                "seed_min": 0,
                "seed_max": EXPLORATION_SEED_MAX,
                "seed_optional": True,
                "safety_notice": (
                    "自由探索只替代初始假设，不会扩大冻结预算，也不会把选择期、"
                    "最终研究期或模块测试段送入 Worker。"
                ),
            },
            "limits": {
                "max_search_turns": 6,
                "max_hypothesis_branches": 3,
                "max_refines_per_branch": 3,
                "max_candidates_per_turn": 4,
                "max_candidates_per_run": 24,
                "max_full_evaluations": 4,
            },
            "worker_configured": bool(
                os.environ.get("MINING_WORKER_URL", "").strip()
            ),
            "analysis_tools_enabled": (
                getattr(self.orchestrator, "analysis_tool_gateway", None)
                is not None
            ),
            "analysis_tool_catalog": (
                list(
                    getattr(
                        self.orchestrator,
                        "analysis_tool_gateway",
                    ).catalog()
                )
                if getattr(self.orchestrator, "analysis_tool_gateway", None)
                is not None
                else []
            ),
            "research": {
                "enabled": bool(
                    os.environ.get("MINING_WORKER_URL", "").strip()
                ),
                "material_root": self._public_material_root(),
                "max_documents": ResearchConversionLimits().max_documents,
                "max_factors": ResearchConversionLimits().max_factors,
                "prompt_versions": {
                    "extraction": "mining_research_extraction_prompt_v1",
                    "conversion": "mining_research_dsl_conversion_prompt_v1",
                },
            },
        }

    def create_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """创建 V2 实验运行。"""

        if not isinstance(payload, Mapping):
            raise MiningContractError("请求体必须是 JSON 对象")
        unknown = set(payload) - _CREATE_FIELDS
        if unknown:
            raise MiningContractError(f"未知 V2 请求字段：{sorted(unknown)}")
        idempotency_key = payload.get("idempotency_key")
        if not isinstance(idempotency_key, str):
            raise MiningContractError("idempotency_key 必须由页面提供")
        budget_payload = payload.get("budget")
        budget = (
            SearchBudget.from_dict(budget_payload)
            if budget_payload is not None
            else load_search_config().budget
        )
        request = SearchRunRequest(
            idempotency_key=idempotency_key,
            hypothesis=payload.get("hypothesis"),
            budget=budget,
            llm_profile=payload.get("llm_profile", "default"),
            input_mode=payload.get("input_mode", HYPOTHESIS_INPUT_MODE),
            exploration_seed=_exploration_seed(payload.get("exploration_seed")),
        )
        context = self._freeze_context(request)
        run = self.orchestrator.create_run(
            request,
            context,
            background=True,
        )
        return self.run_detail(run["run_id"])

    def create_research_run(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """冻结资料目录并创建后台资料驱动运行。"""

        if not isinstance(payload, Mapping):
            raise MiningContractError("请求体必须是 JSON 对象")
        unknown = set(payload) - _RESEARCH_CREATE_FIELDS
        if unknown:
            raise MiningContractError(
                f"未知资料运行请求字段：{sorted(unknown)}"
            )
        idempotency_key = payload.get("idempotency_key")
        if not isinstance(idempotency_key, str):
            raise MiningContractError("idempotency_key 必须由页面提供")
        material_path = payload.get("material_path")
        if not isinstance(material_path, str):
            raise MiningContractError("material_path 必须是相对路径")
        paths = discover_research_material_paths(
            self.material_root,
            material_path,
            recursive=True,
        )
        manifest = build_research_material_manifest(
            self.material_root,
            paths,
            parser_version=RESEARCH_PDF_PARSER_VERSION,
        )
        budget_payload = payload.get("budget")
        budget = (
            SearchBudget.from_dict(budget_payload)
            if budget_payload is not None
            else load_search_config().budget
        )
        request = SearchRunRequest(
            idempotency_key=idempotency_key,
            hypothesis=f"资料驱动研究：{len(paths)} 份研报资料",
            budget=budget,
            llm_profile=payload.get("llm_profile", "default"),
        )
        research_input = ResearchRunInput(
            run_id=request.run_id,
            idempotency_key=request.idempotency_key,
            material_path=material_path,
            manifest=manifest,
            limits=ResearchConversionLimits(),
        )
        context = self._freeze_context(request)
        run = self.orchestrator.create_run(
            request,
            context,
            background=True,
            research_input=research_input,
        )
        return self.run_detail(run["run_id"])

    def _freeze_context(
        self,
        request: SearchRunRequest,
    ) -> FrozenSearchContext:
        catalog = load_catalog()
        worker_config = load_worker_config()
        existing = tuple(
            sorted(
                name
                for names in available_factors(self.factor_dir).values()
                for name in names
            )
        )
        config = replace(load_search_config(), budget=request.budget)
        return FrozenSearchContext.from_config(
            config,
            data_fingerprint=data_snapshot_fingerprint(
                self.db_file,
                self.manifest_file,
            ),
            llm_profile=request.llm_profile,
            llm_model=worker_config.default_model,
            field_catalog=catalog.frozen_field_mappings(),
            operator_catalog=catalog.frozen_operator_mappings(),
            existing_factors=existing,
        )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """列出 V2 运行摘要。"""

        result = []
        for row in self.store.list_runs(limit):
            item = dict(row)
            is_research = bool(item.pop("is_research", False))
            item["source_mode"] = (
                "research" if is_research else item["input_mode"]
            )
            item["engine"] = "v2"
            result.append(item)
        return result

    def run_detail(self, run_id: str) -> dict[str, Any]:
        """返回 V2 假设树、turn、尝试和候选评分摘要。"""

        run = self.store.get_run(run_id)
        return build_search_run_detail(
            self.store,
            run_id,
            self._run_view(run),
        )

    def run_status(self, run_id: str) -> dict[str, Any]:
        """返回 SSE 终止判断所需状态。"""

        state = self.store.search_control_state(run_id)
        return {
            "run_id": run_id,
            "status": state["status"],
            "current_stage": state["interrupted_from"] or state["status"],
            "updated_at": state["updated_at"],
            "error_code": None,
            "error_message": state["stop_reason"],
            "finished": state["status"] in _END_STATUSES,
        }

    def events(
        self,
        run_id: str,
        *,
        after: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        """读取 V2 脱敏事件。"""

        rows = self.store.list_events(run_id, after)
        page = rows[: max(1, min(limit, 1000))]
        return {
            "events": page,
            "last_sequence": int(page[-1]["sequence"]) if page else after,
            "has_more": len(rows) > len(page),
        }

    def cancel(self, run_id: str) -> dict[str, Any]:
        """请求 V2 运行取消。"""

        self.orchestrator.cancel(run_id)
        return self.store.search_control_state(run_id)

    def resume(self, run_id: str) -> dict[str, Any]:
        """恢复中断 V2 运行。"""

        self.orchestrator.resume(run_id, background=True)
        return self.run_detail(run_id)

    def candidate_detail(
        self,
        run_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        """返回 V2 候选、分阶段证据、审批资格和发布状态。"""

        run = self.store.get_run(run_id)
        detail = {
            "run": self._run_view(run),
            **self.review.candidate_detail(run, candidate_id),
        }
        try:
            provenance = self.store.require_research_candidate(
                run_id,
                candidate_id,
                candidate_content_hash=detail["candidate"]["content_hash"],
            )
        except MiningNotFoundError:
            pass
        else:
            detail["research_source"] = provenance.to_dict()
        return detail

    def decide(
        self,
        run_id: str,
        candidate_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """记录 V2 候选最终审批，不触发正式发布。"""

        return self.review.decide(run_id, candidate_id, payload)

    def start_publication_batch(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """整批构建并发布 V2 运行内全部可发布候选。"""

        return self.review.start_publication_batch(run_id, payload)

    def publication_batch(
        self,
        run_id: str,
        batch_id: str,
    ) -> dict[str, Any]:
        """读取 V2 发布批次进度。"""

        return self.review.publication_batch(run_id, batch_id)

    def publication_batches(self, run_id: str) -> list[dict[str, Any]]:
        """列出 V2 运行的全部发布批次。"""

        return self.review.publication_batches(run_id)

    @staticmethod
    def _run_view(run: Mapping[str, Any]) -> dict[str, Any]:
        request = SearchRunRequest.from_dict(
            {
                "idempotency_key": run["idempotency_key"],
                **run["request"],
            }
        )
        is_free = request.input_mode == FREE_EXPLORATION_INPUT_MODE
        context = run["context"]
        windows = context.get("windows", {})
        return {
            "run_id": run["run_id"],
            "idempotency_key": run["idempotency_key"],
            "status": run["status"],
            "hypothesis": run["hypothesis"],
            "engine": "v2",
            "input_mode": request.input_mode,
            "exploration_seed": request.exploration_seed,
            "effective_exploration_seed": (
                request.effective_exploration_seed if is_free else None
            ),
            "seed_source": (
                (
                    "request"
                    if request.exploration_seed is not None
                    else "request_hash"
                )
                if is_free
                else None
            ),
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "context": {
                "context_id": context.get("context_id"),
                "context_hash": context.get("context_hash"),
                "data_version": context.get("data_version"),
                "data_fingerprint": context.get("data_fingerprint"),
                "label": context.get("label"),
                "universe": context.get("universe"),
                "price_channel": context.get("price_channel"),
                "development_start": windows.get("development_start"),
                "development_end": windows.get("development_end"),
                "llm_profile": context.get("llm_profile"),
                "llm_model": context.get("llm_model"),
                "field_catalog_hash": context.get("field_catalog_hash"),
                "operator_catalog_hash": context.get("operator_catalog_hash"),
                "budget": context.get("budget"),
                "created_at": context.get("created_at"),
            },
            "context_hash": run["context_hash"],
        }

    def _public_material_root(self) -> str:
        """返回不暴露主机绝对路径的资料根目录标签。"""

        try:
            return self.material_root.resolve().relative_to(
                PROJECT_ROOT.resolve()
            ).as_posix()
        except ValueError:
            return self.material_root.name


def _exploration_seed(value: Any) -> int | None:
    """把页面十进制种子转换为整数，避免浏览器丢失 64 位精度。"""

    if value is None:
        return None
    if isinstance(value, str):
        if not value or not value.isascii() or not value.isdecimal():
            raise MiningContractError(
                "exploration_seed 必须是 0～2^64-1 的整数"
            )
        return int(value)
    return value
