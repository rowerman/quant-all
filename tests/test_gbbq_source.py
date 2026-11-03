"""gbbq 固定版本文件的完整性校验。"""

import pytest

from scripts.errors import SourceError
from scripts.sources import gbbq


def test_verify_file_accepts_matching_size_and_digest(tmp_path):
    path = tmp_path / "gbbq.parquet"
    path.write_bytes(b"abc")
    digest = gbbq.sha256_file(path)
    assert gbbq.verify_file(path, expected_sha256=digest, expected_bytes=3) == digest


@pytest.mark.parametrize(
    ("content", "expected_sha256"),
    [("abd", "bad"), ("abcd", "bad")],
)
def test_verify_file_rejects_digest_or_size_mismatch(tmp_path, content, expected_sha256):
    path = tmp_path / "gbbq.parquet"
    path.write_bytes(content.encode())
    with pytest.raises(SourceError):
        gbbq.verify_file(path, expected_sha256=expected_sha256, expected_bytes=3)
