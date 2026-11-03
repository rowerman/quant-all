"""S7 挖掘页面浏览器验收：临时候选库、确定性 Worker 和完整审核流程。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
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
    fixture_root = Path(tempfile.mkdtemp(prefix="quant-mining-web-browser-"))
    env = dict(os.environ)
    env["MINING_FIXTURE_ROOT"] = str(fixture_root)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root), str(root / "apps" / "monitor"), env.get("PYTHONPATH", "")))
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tools.monitor.mining_fixture:app",
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
            raise RuntimeError("挖掘浏览器夹具启动失败")
        try:
            with urllib.request.urlopen(f"{url}/api/mining/options", timeout=1) as response:
                if response.status == 200:
                    return process, url, fixture_root
        except OSError:
            time.sleep(0.2)
    process.terminate()
    shutil.rmtree(fixture_root, ignore_errors=True)
    raise RuntimeError("挖掘浏览器夹具启动超时")


def check(url: str, output: Path) -> None:
    """走通创建、审核批准、拒绝和失败候选状态，并保存截图。"""

    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: dialog.accept())

        page.goto(url + "/#/mining")
        page.wait_for_selector("text=提交研究假设")
        page.locator("a[href='#/mining/new']").click()
        page.wait_for_selector("#mining-form")
        page.locator("[name=hypothesis]").fill(
            "短期量价变化可能预测未来五日收益，用于浏览器联合验收。"
        )
        page.locator("summary").click()
        assert page.locator("[name=max_rounds]").input_value() == "1"
        page.route(
            "**/api/mining/runs",
            lambda route: route.fulfill(
                status=503,
                content_type="application/json",
                body='{"code":"MINING-119","message":"Worker 尚未配置","request_id":"browser-worker"}',
            ),
        )
        page.locator("button[type=submit]").click()
        page.wait_for_function(
            "document.querySelector('#mining-form-status').textContent.includes('MINING-119')"
        )
        page.unroute("**/api/mining/runs")
        stream_attempts = {"count": 0}

        def interrupt_first_stream(route):
            if stream_attempts["count"] == 0:
                stream_attempts["count"] += 1
                route.abort()
                return
            route.continue_()

        stream_pattern = re.compile(r".*/api/mining/runs/.*/stream\?after=.*")
        page.route(stream_pattern, interrupt_first_stream)
        page.locator("button[type=submit]").click()
        page.wait_for_url(re.compile(r"#/mining/run_"))
        page.wait_for_function(
            "document.querySelector('#mining-run-status')?.textContent.includes('事件流中断')"
        )
        page.wait_for_selector("#run-candidates tbody tr")
        page.unroute(stream_pattern)
        page.wait_for_function(
            "document.querySelector('#mining-run-status')?.textContent.includes('待人工审核')"
        )
        page.wait_for_function(
            "document.querySelectorAll('.mining-event').length >= 5"
        )
        assert page.locator("#run-candidates tbody tr").count() == 4
        assert page.locator(".mining-event").count() >= 5
        page.screenshot(path=str(output / "run-desktop.png"), full_page=True)

        rows = page.locator("#run-candidates tbody tr")
        rows.nth(0).locator("a").click()
        page.wait_for_selector("#decision-form")
        assert page.locator("#full-evaluation canvas").count() >= 1
        assert "候选内容哈希" in page.locator("#decision-card .hash-grid").inner_text()
        page.screenshot(path=str(output / "candidate-desktop.png"), full_page=True)
        page.locator("[name=note]").fill("证据完整，研究与测试段结论符合模块审核要求。")
        page.route(
            "**/decision",
            lambda route: route.fulfill(
                status=409,
                content_type="application/json",
                body='{"code":"MINING-108","message":"审批证据已变化","request_id":"browser-review"}',
            ),
        )
        page.locator("button[data-decision=approved]").click()
        page.wait_for_function(
            "document.querySelector('#decision-status').textContent.includes('MINING-108')"
        )
        page.unroute("**/decision")
        page.locator("button[data-decision=approved]").click()
        page.wait_for_selector("text=已批准")
        page.screenshot(path=str(output / "candidate-approved.png"), full_page=True)

        page.locator("a[href^='#/mining/run_']").first.click()
        page.wait_for_selector("#run-candidates tbody tr")
        page.locator("#run-candidates tbody tr").nth(1).locator("a").click()
        page.wait_for_selector("#decision-form")
        page.locator("[name=note]").fill("证据完整，第二个候选同批批准。")
        page.locator("button[data-decision=approved]").click()
        page.wait_for_selector("text=已批准")

        page.locator("a[href^='#/mining/run_']").first.click()
        page.wait_for_selector("#run-candidates tbody tr")
        page.locator("#run-candidates tbody tr").nth(2).locator("a").click()
        page.wait_for_selector("#decision-form")
        page.locator("[name=note]").fill("研究机制解释不足，暂不进入正式入库。")
        page.locator("button[data-decision=rejected]").click()
        page.wait_for_selector("text=已拒绝")
        page.screenshot(path=str(output / "candidate-rejected.png"), full_page=True)

        page.locator("a[href^='#/mining/run_']").first.click()
        page.wait_for_selector("#run-candidates tbody tr")
        page.locator("#run-candidates tbody tr").nth(0).locator("a").click()
        page.wait_for_selector("#publish-batch")
        page.locator("#publish-batch").click()
        page.wait_for_selector("text=已发布")
        page.screenshot(path=str(output / "candidate-published.png"), full_page=True)

        page.locator("a[href^='#/mining/run_']").first.click()
        page.wait_for_selector("#run-candidates tbody tr")
        page.locator("#run-candidates tbody tr").nth(3).locator("a").click()
        page.wait_for_selector(".decision-card .warning-text")
        assert "不可审批" in page.locator(".decision-card .warning-text").inner_text()
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "candidate-failed-mobile.png"), full_page=True)
        assert not errors, errors
        browser.close()
    print(f"挖掘页面浏览器验收通过，截图在 {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S7 挖掘页面浏览器验收")
    parser.add_argument("--url", help="使用已有服务；省略时启动临时夹具")
    parser.add_argument("--output", default="AiRecord/evidence/mining_s7")
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
