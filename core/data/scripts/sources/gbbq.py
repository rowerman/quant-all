"""下载并校验公司行动辅助数据 gbbq。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import requests

from scripts.config import EXTRA_DIR
from scripts.errors import SourceError
from scripts.logging_conf import get_logger

logger = get_logger("sources.gbbq")

GBBQ_DIR = EXTRA_DIR / "gbbq"
GBBQ_FILE = GBBQ_DIR / "gbbq.parquet"
MANIFEST_FILE = GBBQ_DIR / "manifest.json"
SOURCE_COMMIT = "d16be984b73673c0db594cd33f777af1bff0babb"
SOURCE_URL = (
    "https://hf-mirror.com/datasets/yifishbossman/financial-analyst-data-full/"
    f"resolve/{SOURCE_COMMIT}/parquet/xdxr/gbbq.parquet?download=true"
)
EXPECTED_SHA256 = "0ca271a5797fd9c1389d76a344aafecfefa64d6d772ef31299361703577e67c9"
EXPECTED_BYTES = 3_445_873


def sha256_file(path: Path) -> str:
    """计算文件内容的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: Path = GBBQ_FILE,
    *,
    expected_sha256: str = EXPECTED_SHA256,
    expected_bytes: int = EXPECTED_BYTES,
) -> str:
    """校验 gbbq 文件大小和摘要，返回摘要。"""
    if not path.is_file():
        raise SourceError(f"缺少 gbbq 文件：{path}")
    size = path.stat().st_size
    if size != expected_bytes:
        raise SourceError(f"gbbq 文件大小不符：期望 {expected_bytes}，实际 {size}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise SourceError(
            f"gbbq 文件摘要不符：期望 {expected_sha256}，实际 {digest}"
        )
    return digest


def _write_manifest(digest: str, size: int, downloaded_at: str) -> None:
    """原子写入下载来源与校验信息。"""
    payload = {
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "sha256": digest,
        "bytes": int(size),
        "downloaded_at": downloaded_at,
    }
    temporary = MANIFEST_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, MANIFEST_FILE)


def download(force: bool = False) -> dict:
    """下载固定版本 gbbq；已有完整文件时直接复用。

    参数:
        force: True 时即使本地摘要正确也重新下载。
    """
    GBBQ_DIR.mkdir(parents=True, exist_ok=True)
    if GBBQ_FILE.exists() and not force:
        digest = verify_file()
        logger.info("gbbq 已存在且校验通过：%s", GBBQ_FILE)
        return {
            "path": str(GBBQ_FILE),
            "sha256": digest,
            "bytes": GBBQ_FILE.stat().st_size,
            "cached": True,
        }
    downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    temporary: Path | None = None
    try:
        with requests.get(SOURCE_URL, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                dir=GBBQ_DIR,
                prefix=".gbbq.",
                suffix=".parquet.tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        digest = verify_file(temporary)
        os.replace(temporary, GBBQ_FILE)
        temporary = None
        _write_manifest(digest, GBBQ_FILE.stat().st_size, downloaded_at)
        logger.info("gbbq 下载完成：%s，%d 字节", GBBQ_FILE, GBBQ_FILE.stat().st_size)
        return {
            "path": str(GBBQ_FILE),
            "sha256": digest,
            "bytes": GBBQ_FILE.stat().st_size,
            "cached": False,
        }
    except requests.RequestException as error:
        raise SourceError(f"gbbq 下载失败：{error}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    """命令行入口。"""
    result = download()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
