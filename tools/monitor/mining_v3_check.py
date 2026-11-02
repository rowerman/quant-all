"""V3 四类来源、审核入口和移动端浏览器验收。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server() -> tuple[subprocess.Popen, str]:
    port = _free_port()
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(root),
                str(root / "apps" / "monitor"),
                env.get("PYTHONPATH", ""),
            ),
        )
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tools.monitor.mining_v3_fixture:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("V3 浏览器夹具启动失败")
        try:
            with urllib.request.urlopen(
                f"{url}/api/mining/options",
                timeout=1,
            ) as response:
                if response.status == 200:
                    return process, url
        except OSError:
            time.sleep(0.2)
    process.terminate()
    raise RuntimeError("V3 浏览器夹具启动超时")


def check(url: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))

        page.goto(url + "/#/mining/new")
        page.wait_for_selector("#source-mode")
        options = page.locator("#source-mode option").all_text_contents()
        assert options == [
            "手工研究假设",
            "自由探索",
            "资料驱动",
            "CoSTEER 人工提升",
        ]
        page.locator("#source-mode").select_option("costeer")
        page.locator("[name=promotion_id]").wait_for(state="visible")
        assert "只接受服务端已封存" in page.locator("main").inner_text()
        page.screenshot(path=str(output / "v3-new.png"), full_page=True)

        page.goto(url + "/#/mining")
        page.wait_for_selector("#mining-runs tbody tr")
        text = page.locator("main").inner_text()
        assert "V3" in text or "v3" in text
        for label in ("手工研究假设", "自由探索", "资料驱动", "CoSTEER 人工提升"):
            assert label in text
        page.screenshot(path=str(output / "v3-runs.png"), full_page=True)

        row = page.locator("#mining-runs a", has_text="CoSTEER").first
        row.click()
        page.wait_for_selector("#search-candidates tbody tr")
        assert "V3 搜索 researchrun_" in page.locator("main").inner_text()
        page.locator("#search-candidates tbody tr").first.locator("a").click()
        page.wait_for_selector("text=V3 来源绑定")
        assert "审批证据哈希" in page.locator("main").inner_text()
        page.screenshot(
            path=str(output / "v3-candidate.png"),
            full_page=True,
        )
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(
            path=str(output / "v3-candidate-mobile.png"),
            full_page=True,
        )
        assert not errors, errors
        browser.close()
    print(f"V3 页面浏览器验收通过，截图在 {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 挖掘页面验收")
    parser.add_argument(
        "--output",
        default="AiRecord/evidence/mining_v3_web",
    )
    args = parser.parse_args()
    process, url = _start_server()
    try:
        check(url, Path(args.output))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
