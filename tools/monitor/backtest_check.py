"""回测页面真实浏览器验收：跑一次短区间回测并打开结果详情页。

用法::

    bash tools/monitor/serve.sh            # 另开一个终端
    .venv/bin/python tools/monitor/backtest_check.py --url http://127.0.0.1:8800
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(url, output, start, end):
    """验收回测列表页与详情页，参数为服务地址、截图目录和回测区间。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url + '/#/backtest/new')
        page.wait_for_selector('#strategy-form')
        page.locator('[name=signal_mode]').select_option('provider')
        page.wait_for_selector('#provider-fields [data-param=min_breadth]')
        assert page.locator('#factor-config').is_hidden()
        assert page.locator('#provider-fields [data-param=min_breadth]').input_value() == '50'
        page.locator('[name=provider_name]').select_option('momentum_trailing_stop')
        page.wait_for_selector('#provider-fields [data-param=stop_loss]')
        assert page.locator('#provider-fields [data-param=stop_loss]').input_value() == '10'
        assert page.locator('#provider-fields [data-param=trailing_stop]').input_value() == '15'
        page.locator('#validate-provider').click()
        page.wait_for_function(
            "document.querySelector('#strategy-status').textContent.startsWith('参数校验通过')",
            timeout=30000,
        )
        page.screenshot(path=str(output / 'backtest-new-provider.png'), full_page=True)
        page.goto(url + '/#/backtest')
        page.wait_for_selector('#backtest-form')
        assert page.locator('#backtest-form select[name=strategy_id] option').count() >= 2
        assert page.locator('#backtest-form select[name=universe] option').count() >= 2
        assert page.locator('#backtest-form select[name=benchmark] option').count() == 4
        assert page.locator('#backtest-form select[name=benchmark]').input_value() == 'H00300.CSI'
        page.locator('[name=start]').fill(start)
        page.locator('[name=end]').fill(end)
        page.locator('button[type=submit]').click()
        page.wait_for_function(
            "document.querySelector('#backtest-status').textContent.startsWith('回测完成')", timeout=600000
        )
        assert page.locator('#run-list tbody tr').count() >= 1
        page.screenshot(path=str(output / 'backtest-list.png'), full_page=True)
        page.locator('#backtest-status a').click()
        page.wait_for_selector('#run-metrics tbody tr')
        assert page.locator('#run-info tbody tr').count() == 1
        assert page.locator('#run-curves canvas').count() >= 2
        assert page.locator('#run-cost canvas').count() >= 2
        assert page.locator('#run-calibration tbody tr').count() >= 5
        # 区间太短时可能一笔都没成交，只要持仓区渲染出来就算通过
        assert page.locator('#run-holding').count() == 1
        # 退市强制清算区块（没有清算时显示提示文案）
        assert page.locator('#run-liquidation').count() == 1
        page.screenshot(path=str(output / 'backtest-detail.png'), full_page=True)
        browser.close()
    assert not errors, errors
    print('回测页面验收通过，截图在', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='回测页面浏览器验收')
    parser.add_argument('--url', default='http://127.0.0.1:8800')
    parser.add_argument('--output', default='AiRecord/evidence/monitor_browser')
    parser.add_argument('--start', default='2026-08-03')
    parser.add_argument('--end', default='2026-09-28')
    args = parser.parse_args()
    check(args.url, args.output, args.start, args.end)
