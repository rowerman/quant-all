"""资料驱动 V2 页面和来源展示的浏览器验收。"""

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


def _start_server() -> tuple[subprocess.Popen, str, Path]:
    """启动带资料候选样例的临时监测站。"""

    port = _free_port()
    root = Path(__file__).resolve().parents[2]
    fixture_root = Path(tempfile.mkdtemp(prefix="quant-mining-research-web-"))
    env = dict(os.environ)
    env["MINING_SEARCH_FIXTURE_ROOT"] = str(fixture_root)
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
            "tools.monitor.mining_search_fixture:app",
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
            raise RuntimeError("资料驱动浏览器夹具启动失败")
        try:
            with urllib.request.urlopen(
                f"{url}/api/mining/options",
                timeout=1,
            ) as response:
                if response.status == 200:
                    return process, url, fixture_root
        except OSError:
            time.sleep(0.2)
    process.terminate()
    shutil.rmtree(fixture_root, ignore_errors=True)
    raise RuntimeError("资料驱动浏览器夹具启动超时")


def check(url: str, output: Path) -> None:
    """验收资料模式、运行摘要、候选来源和窄屏布局。"""

    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))

        page.goto(url + "/#/mining/new")
        page.wait_for_selector("#source-mode")
        page.locator("#source-mode").select_option("research")
        page.locator("[name=material_path]").wait_for(state="visible")
        assert "data/source/reports" in page.locator("main").inner_text()

        page.goto(url + "/#/mining")
        page.wait_for_selector("text=资料驱动研究：1 份研报资料")
        page.locator("text=资料驱动研究：1 份研报资料").click()
        page.wait_for_selector("#research-materials tbody tr")
        assert "资料来源" in page.locator("main").inner_text()
        assert "2026-10/short-reversal.pdf" in page.locator("main").inner_text()
        page.screenshot(
            path=str(output / "research-run-desktop.png"),
            full_page=True,
        )

        page.locator("#search-candidates tbody tr").first.locator("a").click()
        page.wait_for_selector("#research-source-evidence tbody tr")
        assert "研报来源依据" in page.locator("main").inner_text()
        assert "Equity research report" in page.locator("main").inner_text()
        page.screenshot(
            path=str(output / "research-candidate-desktop.png"),
            full_page=True,
        )

        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(
            path=str(output / "research-candidate-mobile.png"),
            full_page=True,
        )
        assert not errors, errors
        browser.close()
    print(f"资料驱动页面浏览器验收通过，截图在 {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="资料驱动 V2 页面验收")
    parser.add_argument("--url", help="使用已有服务；省略时启动临时夹具")
    parser.add_argument(
        "--output",
        default="AiRecord/evidence/mining_research_web",
    )
    args = parser.parse_args()
    process = None
    fixture_root = None
    target = args.url
    try:
        if target is None:
            process, target, fixture_root = _start_server()
        check(target.rstrip("/"), Path(args.output))
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if fixture_root is not None:
            shutil.rmtree(fixture_root, ignore_errors=True)
