"""正式 mined 族的暂存构建、校验、原子提交与回滚。"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping

import duckdb

import factors
from factors.published import MINED_FAMILY, PUBLISHED_SCHEMA_VERSION
from factors.registry import FactorSpec, required_lookback
from mining.contract_requests import FrozenContext
from mining.contract_types import FULL_FACTOR_START, utc_now_iso
from mining.errors import (
    PublicationConflictError,
    PublicationValidationError,
)
from mining.expression_validation import validate_expression
from mining.publication_commit import (
    StagedPublication,
    commit_publication,
    rollback_publication,
    tree_hash,
    write_json,
)
from mining.publication_validation import validate_full_tree, validate_no_lookahead
from scripts.build_factors import catalog
from scripts.build_factors.sql import coverage_query, family_query


def _candidate_spec(candidate: Mapping[str, Any], context: FrozenContext) -> FactorSpec:
    """从冻结候选重新校验并构造正式因子定义。"""

    definition = candidate["definition"]
    validated = validate_expression(
        definition["canonical_expression"],
        direction=definition["direction"],
        max_depth=context.budget.max_expression_depth,
        excluded_fields=context.budget.excluded_fields,
    )
    if validated.semantic_hash != candidate["semantic_hash"]:
        raise PublicationConflictError("候选语义哈希与表达式不一致")
    return FactorSpec(
        name=candidate["name"],
        family=MINED_FAMILY,
        formula=definition["canonical_expression"],
        description=definition["description"],
        steps=tuple(definition["steps"]),
        layers=tuple(validated.layers),
        lookback=max(int(validated.lookback), required_lookback(tuple(validated.layers))),
        direction=int(definition["direction"]),
        fields=tuple(validated.dependencies),
        extra_sources=tuple(validated.extra_sources),
    )


def _entry(
    spec: FactorSpec,
    candidate: Mapping[str, Any] | None = None,
    *,
    run_id: str | None = None,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """生成定义清单条目，并保留人工批准来源。"""

    payload = {
        "name": spec.name,
        "family": MINED_FAMILY,
        "formula": spec.formula,
        "description": spec.description,
        "steps": list(spec.steps),
        "layers": list(spec.layers),
        "lookback": spec.lookback,
        "direction": spec.direction,
        "fields": list(spec.fields),
        "extra_sources": list(spec.extra_sources),
    }
    if candidate is not None:
        payload.update(
            {
                "run_id": run_id,
                "candidate_id": candidate["candidate_id"],
                "candidate_content_hash": candidate["content_hash"],
                "semantic_hash": candidate["semantic_hash"],
                "approval_id": approval_id,
            }
        )
    return payload


def _write_year_files(
    conn: duckdb.DuckDBPyConnection,
    *,
    family_root: Path,
    specs: tuple[FactorSpec, ...],
    start: str,
    end: str,
) -> None:
    """一次计算完整定义集，再按年写入暂存目录。"""

    family_root.mkdir(parents=True)
    combined = family_root / "_combined.parquet"
    query = family_query(specs, spine_start=start, spine_end=end, db="src")
    escaped = str(combined).replace("'", "''")
    conn.execute(
        f"COPY ({query}) TO '{escaped}' "
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
        target = family_root / f"year={year}" / "part.parquet"
        target.parent.mkdir()
        conn.execute(
            f"COPY (SELECT * FROM read_parquet('{escaped}') "
            f"WHERE year(trade_date)={year}) TO '{str(target).replace(chr(39), chr(39) * 2)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000)"
        )
    combined.unlink()


def _coverage_rows(
    conn: duckdb.DuckDBPyConnection,
    specs: tuple[FactorSpec, ...],
    family_root: Path,
) -> tuple[dict[str, Any], ...]:
    """计算暂存 mined 族的逐因子逐年覆盖率。"""

    frame = conn.execute(
        coverage_query(specs, family_root.parent, MINED_FAMILY)
    ).df()
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "name": row.name,
                "family": MINED_FAMILY,
                "year": int(row.year),
                "rows": int(row.rows),
                "non_null": int(row.non_null),
                "min_value": row.min_value,
                "max_value": row.max_value,
                "mean_value": row.mean_value,
                "std_value": row.std_value,
                "p01": row.p01,
                "p99": row.p99,
            }
        )
    return tuple(rows)


def _stage_catalog(
    *,
    source_catalog: Path,
    target_catalog: Path,
    specs: tuple[FactorSpec, ...],
    coverage: tuple[dict[str, Any], ...],
    factor_dir: Path,
    batch_id: str,
    start: str,
    end: str,
) -> None:
    """复制正式目录库，在副本中合并新定义、覆盖率和视图。"""

    shutil.copy2(source_catalog, target_catalog)
    with catalog.connect(target_catalog) as conn:
        catalog.ensure_catalog(conn)
        names = [spec.name for spec in specs]
        placeholders = ",".join("?" for _ in names)
        conn.execute(
            f"DELETE FROM meta.factor_definition WHERE name IN ({placeholders})",
            names,
        )
        now = __import__("datetime").datetime.now()
        conn.executemany(
            "INSERT INTO meta.factor_definition VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [catalog._spec_row(spec, "implemented", now) for spec in specs],
        )
        catalog.refresh_coverage(conn, coverage)
        catalog.refresh_views(conn, factor_dir)
        catalog.record_build(
            conn,
            {
                "run_id": batch_id,
                "started_at": now,
                "finished_at": now,
                "families": MINED_FAMILY,
                "years": f"{start[:4]}-{end[:4]}",
                "factor_count": len(specs),
                "rows_written": sum(item["rows"] for item in coverage),
                "status": "ok",
                "message": "",
            },
        )
        catalog.refresh_manifest(conn)


def prepare_publication(
    *,
    batch_id: str,
    run_id: str,
    candidates: tuple[Mapping[str, Any], ...],
    approval_ids: tuple[str, ...],
    context: FrozenContext,
    staging_root: Path,
    db_file: Path,
    source_catalog: Path,
    factor_dir: Path,
    tolerance: float = 1e-5,
) -> StagedPublication:
    """构建并校验完整 mined 族；此时尚未修改正式路径。"""

    if staging_root.exists():
        raise PublicationConflictError(f"发布暂存目录已存在：{staging_root}")
    staging_root.mkdir(parents=True)
    previous_manifest = factors.published.load_manifest(factor_dir)
    previous_specs = factors.published.published_specs(factor_dir)
    new_specs = tuple(
        _candidate_spec(candidate, context)
        for candidate in candidates
    )
    existing_names = {spec.name for spec in factors.all_factors()}
    existing_names.update(spec.name for spec in previous_specs)
    for spec in new_specs:
        if spec.name in existing_names:
            raise PublicationConflictError(f"正式因子名称冲突：{spec.name}")
        existing_names.add(spec.name)
    semantic_hashes = {
        item.get("semantic_hash")
        for item in previous_manifest["factors"]
        if item.get("semantic_hash")
    }
    for candidate, spec in zip(candidates, new_specs, strict=True):
        if candidate["semantic_hash"] in semantic_hashes:
            raise PublicationConflictError(
                f"正式因子语义重复：{spec.name}"
            )
        semantic_hashes.add(candidate["semantic_hash"])

    all_specs = tuple(
        sorted(previous_specs + new_specs, key=lambda spec: spec.name)
    )
    family_root = staging_root / MINED_FAMILY
    conn = duckdb.connect()
    try:
        conn.execute(f"ATTACH '{str(db_file).replace(chr(39), chr(39) * 2)}' AS src (READ_ONLY)")
        _write_year_files(
            conn,
            family_root=family_root,
            specs=all_specs,
            start=FULL_FACTOR_START,
            end=context.holdout_end,
        )
        validate_full_tree(
            conn=conn,
            family_root=family_root,
            specs=all_specs,
            start=FULL_FACTOR_START,
            end=context.holdout_end,
            min_coverage=context.budget.min_coverage,
        )
        validate_no_lookahead(
            conn=conn,
            family_root=family_root,
            specs=all_specs,
            start=FULL_FACTOR_START,
            end=context.holdout_end,
            tolerance=tolerance,
        )
        coverage = _coverage_rows(conn, all_specs, family_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    finally:
        conn.close()

    entries = {
        item["name"]: dict(item) for item in previous_manifest["factors"]
    }
    for candidate, spec, approval_id in zip(
        candidates,
        new_specs,
        approval_ids,
        strict=True,
    ):
        entries[spec.name] = _entry(
            spec,
            candidate,
            run_id=run_id,
            approval_id=approval_id,
        )
    manifest = {
        "schema_version": PUBLISHED_SCHEMA_VERSION,
        "updated_at": utc_now_iso(),
        "batch_id": batch_id,
        "factors": [entries[name] for name in sorted(entries)],
    }
    definitions_path = family_root / "definitions.json"
    write_json(definitions_path, manifest)
    manifest_hash = hashlib.sha256(definitions_path.read_bytes()).hexdigest()
    files = tuple(sorted(family_root.glob("year=*/part.parquet")))
    artifact_sha256 = tree_hash(files, family_root)
    catalog_path = staging_root / "factors.staged.duckdb"
    _stage_catalog(
        source_catalog=source_catalog,
        target_catalog=catalog_path,
        specs=all_specs,
        coverage=coverage,
        factor_dir=factor_dir,
        batch_id=batch_id,
        start=FULL_FACTOR_START,
        end=context.holdout_end,
    )
    return StagedPublication(
        staging_root=staging_root,
        family_root=family_root,
        definitions_path=definitions_path,
        catalog_path=catalog_path,
        manifest_hash=manifest_hash,
        artifact_sha256=artifact_sha256,
        factor_names=tuple(spec.name for spec in new_specs),
        specs=all_specs,
        coverage=coverage,
    )
