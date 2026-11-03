"""直接依赖与完整锁文件的一致性。"""

from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _pins(path: Path) -> dict[str, str]:
    """读取 ``名称==版本`` 行。"""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        name, pinned = line.split("==", 1)
        pinned = pinned.removesuffix("\\").strip()
        pins[name.lower()] = pinned
    return pins


def test_direct_dependencies_match_installed_and_locked_versions():
    """直接依赖必须与当前环境和完整锁一致，避免只改一处造成漂移。"""
    direct = _pins(ROOT / "requirements.txt")
    locked = _pins(ROOT / "requirements.lock")
    assert direct
    for name, pinned in direct.items():
        assert version(name) == pinned
        assert locked.get(name) == pinned
