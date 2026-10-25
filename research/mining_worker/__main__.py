"""Worker 容器入口。"""

from __future__ import annotations

import os

from mining_worker.service import app_from_environment


def main() -> None:
    """启动仅监听本机转发端口的 Worker 服务。"""

    app_from_environment().run(
        host=os.environ.get("MINING_WORKER_HOST", "0.0.0.0"),
        port=int(os.environ.get("MINING_WORKER_PORT", "8790")),
        threaded=True,
    )


if __name__ == "__main__":
    main()
