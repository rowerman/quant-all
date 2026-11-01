"""V2 挖掘审批、发布失败重试和移动端浏览器验收。"""

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
    port = _free_port()
    root = Path(__file__).resolve().parents[2]
    fixture_root = Path(tempfile.mkdtemp(prefix="quant-mining-v2-browser-"))
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
            raise RuntimeError("V2 挖掘浏览器夹具启动失败")
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
    raise RuntimeError("V2 挖掘浏览器夹具启动超时")


def check(url: str, output: Path) -> None:
    """完成 V2 批准、拒绝、发布失败重试和移动端验收。"""

    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: dialog.accept())

        page.goto(url + "/#/mining")
        page.wait_for_selector("text=短期趋势可能带来开发期超额收益")
        page.locator(
            "#mining-runs a",
            has_text="短期趋势可能带来开发期超额收益",
        ).first.click()
        page.wait_for_selector("#search-candidates tbody tr")
        assert page.locator("#search-candidates tbody tr").count() == 3
        page.screenshot(
            path=str(output / "run-desktop.png"),
            full_page=True,
        )

        rows = page.locator("#search-candidates tbody tr")
        rows.nth(0).locator("a").click()
        page.wait_for_selector("#decision-form")
        assert "V2 评分卡" in page.locator("main").inner_text()
        assert "开发期证据" in page.locator("main").inner_text()
        assert "选择期证据" in page.locator("main").inner_text()
        page.locator("[name=note]").fill("五阶段证据完整，批准候选。")
        page.locator("button[data-decision=approved]").click()
        page.wait_for_selector("text=已批准")
        page.screenshot(
            path=str(output / "candidate-approved.png"),
            full_page=True,
        )

        page.locator("a.secondary-link").first.click()
        page.wait_for_selector("#search-candidates tbody tr")
        page.locator("#search-candidates tbody tr").nth(1).locator("a").click()
        page.wait_for_selector("#decision-form")
        page.locator("[name=note]").fill("同批候选证据完整，批准发布。")
        page.locator("button[data-decision=approved]").click()
        page.wait_for_selector("text=已批准")

        page.locator("a.secondary-link").first.click()
        page.wait_for_selector("#search-candidates tbody tr")
        page.locator("#search-candidates tbody tr").nth(2).locator("a").click()
        page.wait_for_selector("#decision-form")
        page.locator("[name=note]").fill("机制解释不足，拒绝进入发布。")
        page.locator("button[data-decision=rejected]").click()
        page.wait_for_selector("text=已拒绝")

        page.locator("a.secondary-link").first.click()
        page.wait_for_selector("#search-candidates tbody tr")
        page.locator("#search-candidates tbody tr").nth(0).locator("a").click()
        page.wait_for_selector("#publish-batch")
        page.locator("#publish-batch").click()
        page.wait_for_selector(
            "#publish-batch",
            state="visible",
            timeout=15000,
        )
        page.wait_for_function(
            "document.querySelector('#publish-batch')?.textContent.includes('重试')"
            " && document.querySelector('.page-heading .intro')?.textContent.includes('正式构建失败')"
        )
        assert "正式构建失败" in page.locator(".publication-card").inner_text()
        page.screenshot(
            path=str(output / "candidate-failed.png"),
            full_page=True,
        )

        page.locator("#publish-batch").click()
        page.wait_for_function(
            "document.querySelector('.page-heading .intro')?.textContent.includes('已发布')",
            timeout=15000,
        )
        assert "已发布" in page.locator(".publication-card").inner_text()
        page.screenshot(
            path=str(output / "candidate-published.png"),
            full_page=True,
        )

        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(
            path=str(output / "candidate-mobile.png"),
            full_page=True,
        )
        assert not errors, errors
        browser.close()
    print(f"V2 挖掘页面浏览器验收通过，截图在 {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 挖掘页面浏览器验收")
    parser.add_argument("--url", help="使用已有服务；省略时启动临时夹具")
    parser.add_argument(
        "--output",
        default="AiRecord/evidence/mining_v2_web",
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
