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
from mining.errors import MiningEvaluationError
from mining.expression_validation import (
    ValidatedExpression,
    validate_candidate_expression,
)
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


def _candidate_base(candidate_root: Path | str) -> Path:
    """校验候选根目录并返回规范化路径。

    参数:
        candidate_root: 调用方给出的候选根目录。
    """

    base = Path(candidate_root).resolve()
    formal = FACTOR_DIR.resolve()
    if base == formal or base.is_relative_to(formal):
        raise MiningEvaluationError("候选根目录不能位于正式因子目录内")
    return base


def _computation(
    candidate: CandidateDefinition,
    validated: ValidatedExpression,
) -> CandidateComputation:
    """由已验证表达式构造候选计算定义。

    参数:
        candidate: 冻结候选定义。
        validated: 重新校验后的表达式。
    """

    return CandidateComputation(
        name=candidate.name,
        layers=validated.layers,
        extra_sources=validated.extra_sources,
    )


def _factor_view(
    candidate: CandidateDefinition,
    validated: ValidatedExpression,
    factor_root: Path,
) -> FactorView:
    """构造候选评估使用的只读因子视图。

    参数:
        candidate: 冻结候选定义。
        validated: 重新校验后的表达式。
        factor_root: 已发布因子目录。
    """

    return FactorView(
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


def _publish_staging(
    staging: Path,
    factor_root: Path,
    *,
    identity: dict[str, object],
    candidate: CandidateDefinition,
    computation: CandidateComputation,
    validated: ValidatedExpression,
) -> MaterializedCandidate:
    """写入 manifest，原子发布候选目录并重新校验产物。

    参数:
        staging: 包含完整年分区的临时目录。
        factor_root: 最终候选因子目录。
        identity: 参与复用判断的稳定身份。
        candidate/computation/validated: 候选定义与校验结果。
    """

    files = _candidate_files(staging)
    if not files:
        raise MiningEvaluationError("候选物化没有生成任何年分区")
    artifact_sha256 = _artifact_hash(files, staging)
    _write_manifest(
        staging / "materialization.json",
        {**identity, "artifact_sha256": artifact_sha256},
    )
    if factor_root.exists():
        raise MiningEvaluationError(f"候选因子目录已被占用但不可复用：{factor_root}")
    os.replace(staging, factor_root)
    published_files = _candidate_files(factor_root)
    published_hash = _artifact_hash(published_files, factor_root)
    manifest = _read_manifest(factor_root / "materialization.json")
    if (
        published_hash != artifact_sha256
        or manifest.get("artifact_sha256") != published_hash
    ):
        shutil.rmtree(factor_root, ignore_errors=True)
        raise MiningEvaluationError("候选因子发布后产物哈希或 manifest 不匹配")
    return MaterializedCandidate(
        factor_view=_factor_view(candidate, validated, factor_root),
        computation=computation,
        artifact_sha256=published_hash,
        files=published_files,
    )


def _log_materialized(
    candidate: CandidateDefinition,
    materialized: MaterializedCandidate,
    *,
    operation: str,
) -> None:
    """记录单个候选的物化结果。"""

    _LOGGER.info(
        format_event(
            {
                "operation": operation,
                "candidate_id": candidate.candidate_id,
                "artifact_sha256": materialized.artifact_sha256,
                "files": len(materialized.files),
            }
        )
    )


def _materialize_new(
    root: Path,
    *,
    computation: CandidateComputation,
    start: str,
    end: str,
) -> Path:
    """在独立 staging 目录生成完整候选因子。

    参数:
        root: 最终候选因子目录。
        computation: 单候选计算定义。
        start/end: 物化日期区间。
    """

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
    return staging


def materialize_candidate(
    candidate: CandidateDefinition,
    context: FrozenContext,
    candidate_root: Path | str,
    *,
    start: str,
    end: str,
) -> MaterializedCandidate:
    """校验并物化候选，拒绝任何写入正式因子目录的路径。"""

    base = _candidate_base(candidate_root)
    validated = validate_candidate_expression(candidate, context)
    computation = _computation(candidate, validated)
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
        factor_view = _factor_view(candidate, validated, factor_root)
        materialized = MaterializedCandidate(
            factor_view,
            computation,
            artifact_sha256,
            files,
        )
    else:
        staging = _materialize_new(
            factor_root,
            computation=computation,
            start=start,
            end=end,
        )
        try:
            materialized = _publish_staging(
                staging,
                factor_root,
                identity=identity,
                candidate=candidate,
                computation=computation,
                validated=validated,
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    _log_materialized(candidate, materialized, operation="materialize_candidate")
    return materialized
