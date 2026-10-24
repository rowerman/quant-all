"""把前后端包与仓库根加入导入路径。

前后端统一放在 ``apps/monitor/`` 下，``import backend`` 需要本目录，
而 ``backend`` 又依赖仓库根的 ``calculate``/``quant``/``backtest``：
放在这里的 conftest 会在收集 ``apps/monitor/**`` 的测试时自动生效。
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
for _path in (_PROJECT_ROOT, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
