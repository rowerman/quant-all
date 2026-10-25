"""候选因子值的安全物化与目录隔离。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from calculate import FactorView, open_source
from common.paths import FACTOR_DIR
from mining.contract_candidates import CandidateDefinition
from mining.contract_requests import FrozenContext
from mining.contract_types import canonical_json
from mining.errors import CandidateError, MiningEvaluationError
from mining.expression_validation import ValidatedExpression, validate_expression
from mining.logging_conf import format_event, get_mining_logger
from scripts.build_factors.sql import family_query

_LOGGER = get_mining_logger("mining.evaluation.materialization")


@dataclass(frozen=True)
class CandidateComputation:
    """构造候选 SQL 所需的最小冻结定义，不进入正式注册表。"""

    name: str
    layers: tuple[str, ...]
    extra_sources: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedCandidate:
    """已校验并物化的候选因子。"""

    factor_view: FactorView
    computation: CandidateComputation
    artifact_sha256: str
    files: tuple[Path, ...]


def _validated_candidate(
    candidate: CandidateDefinition,
    context: FrozenContext,
) -> ValidatedExpression:
    """重新执行 S3 校验，拒绝落库后被修改或漂移的候选。"""

    validated = validate_expression(
        candidate.canonical_expression,
        direction=candidate.direction,
        max_depth=context.budget.max_expression_depth,
        excluded_fields=context.budget.excluded_fields,
    )
    if (
        validated.semantic_hash != candidate.semantic_hash
        or validated.factor_name != candidate.name
        or validated.lookback != candidate.lookback
        or validated.dependencies != candidate.dependencies
        or validated.output_type != candidate.output_type
        or validated.parameters != dict(candidate.parameters)
    ):
        raise CandidateError("候选表达式、稳定名或语义哈希与冻结定义不一致")
    return validated


def _artifact_hash(files: tuple[Path, ...], root: Path) -> str:
    """对按相对路径排序的因子文件计算树哈希。"""

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _identity(
    validated: ValidatedExpression,
    candidate: CandidateDefinition,
    context: FrozenContext,
    start: str,
    end: str,
) -> dict[str, object]:
    """返回候选物化可复用所需的稳定身份。"""

    return {
        "catalog_hash": validated.catalog_hash,
        "candidate_content_hash": candidate.content_hash,
        "candidate_semantic_hash": candidate.semantic_hash,
        "data_fingerprint": context.data_fingerprint,
        "end": end,
        "name": candidate.name,
        "start": start,
    }


def _read_manifest(path: Path) -> dict[str, object]:
    """读取候选物化 manifest。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise MiningEvaluationError(f"候选物化 manifest 缺失或损坏：{path}") from exc
    if not isinstance(payload, dict):
        raise MiningEvaluationError("候选物化 manifest 根节点必须是对象")
    return payload


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    """原子写入候选物化 manifest。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload), encoding="utf-8")
    os.replace(temporary, path)


def _candidate_files(root: Path) -> tuple[Path, ...]:
    """返回候选因子年分区文件。"""

    return tuple(sorted(root.glob("year=*/part.parquet")))


def _materialize_new(
    root: Path,
    *,
    computation: CandidateComputation,
    start: str,
    end: str,
) -> tuple[tuple[Path, ...], Path]:
    """在独立 staging 目录生成完整候选因子后一次替换。"""

    staging = root.parent / f".{root.name}.staging-{uuid.uuid4().hex}"
    combined = staging / "_candidate.parquet"
    staging.mkdir(parents=True)
    conn = open_source()
    try:
        query = family_query([computation], spine_start=start, spine_end=end)
        quoted = str(combined).replace("'", "''")
        conn.execute(
            f"COPY ({query}) TO '{quoted}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000)"
        )
        years = [
            int(row[0])
            for row in conn.execute(
                "SELECT DISTINCT year(trade_date) FROM read_parquet(?) ORDER BY 1",
                [str(combined)],
            ).fetchall()
        ]
        for year in years:
            target = staging / f"year={year}" / "part.parquet"
            target.parent.mkdir(parents=True)
            target_sql = str(target).replace("'", "''")
            conn.execute(
                f"COPY (SELECT * FROM read_parquet('{quoted}') "
                f"WHERE year(trade_date)={year}) TO '{target_sql}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000)"
            )
        combined.unlink()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        conn.close()
    files = _candidate_files(staging)
    if not files:
        shutil.rmtree(staging, ignore_errors=True)
        raise MiningEvaluationError("候选物化没有生成任何年分区")
    if root.exists():
        raise MiningEvaluationError(f"候选因子目录已被占用但不可复用：{root}")
    os.replace(staging, root)
    return _candidate_files(root), root


def materialize_candidate(
    candidate: CandidateDefinition,
    context: FrozenContext,
    candidate_root: Path | str,
    *,
    start: str,
    end: str,
) -> MaterializedCandidate:
    """校验并物化候选，拒绝任何写入正式因子目录的路径。"""

    base = Path(candidate_root).resolve()
    if base == FACTOR_DIR.resolve() or base.is_relative_to(FACTOR_DIR.resolve()):
        raise MiningEvaluationError("候选根目录不能位于正式因子目录内")
    validated = _validated_candidate(candidate, context)
    computation = CandidateComputation(
        name=candidate.name,
        layers=validated.layers,
        extra_sources=validated.extra_sources,
    )
    factor_root = base / "factor"
    manifest_path = factor_root / "materialization.json"
    identity = _identity(validated, candidate, context, start, end)
    if factor_root.exists():
        manifest = _read_manifest(manifest_path)
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise MiningEvaluationError("已有候选因子与当前定义或数据快照不一致")
        files = _candidate_files(factor_root)
        if not files:
            raise MiningEvaluationError("已有候选因子目录没有年分区")
        artifact_sha256 = _artifact_hash(files, factor_root)
        if manifest.get("artifact_sha256") != artifact_sha256:
            raise MiningEvaluationError("已有候选因子产物哈希不匹配")
    else:
        files, written_root = _materialize_new(
            factor_root,
            computation=computation,
            start=start,
            end=end,
        )
        artifact_sha256 = _artifact_hash(files, written_root)
        _write_manifest(
            manifest_path,
            {**identity, "artifact_sha256": artifact_sha256},
        )
    factor_view = FactorView(
        name=candidate.name,
        family="mining_candidate",
        formula=candidate.canonical_expression,
        description=candidate.description,
        steps=candidate.steps,
        direction=candidate.direction,
        lookback=candidate.lookback,
        fields=candidate.dependencies,
        extra_sources=validated.extra_sources,
        parquet_glob=str(factor_root / "year=*" / "part.parquet"),
        formal=False,
    )
    _LOGGER.info(
        format_event(
            {
                "operation": "materialize_candidate",
                "candidate_id": candidate.candidate_id,
                "artifact_sha256": artifact_sha256,
                "files": len(files),
            }
        )
    )
    return MaterializedCandidate(
        factor_view,
        computation,
        artifact_sha256,
        files,
    )
