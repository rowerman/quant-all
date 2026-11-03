"""真实浏览器验收：启动服务后从项目根目录运行本脚本，需要Playwright/Chromium。"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(url, output):
    """验收三个页面与全指标渲染，参数为服务URL和截图输出目录。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url + '/#/factors')
        page.wait_for_selector('#factor-list tbody tr')
        assert page.locator('#factor-list tbody tr').count() == 29
        page.locator('#search').fill('MOM_20')
        assert page.locator('#factor-list tbody tr').count() == 1
        page.locator('#factor-list a').click()
        page.wait_for_selector('#evaluate-form')
        page.locator('[name=start]').fill('2024-05-06')
        page.locator('[name=end]').fill('2024-05-31')
        page.locator('button[type=submit]').click()
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('分析完成')", timeout=120000)
        assert page.locator('#results section').count() == 13
        assert page.locator('.skeleton').count() == 0
        assert page.locator('[data-metric=bin_returns] canvas').count() == 2
        assert page.locator('[data-metric=bin_dist] canvas').count() == 2
        assert page.locator('[data-metric=correlation] canvas').count() == 1
        page.screenshot(path=str(output / 'detail.png'), full_page=True)
        page.locator('nav a[href="#/compare"]').click()
        page.wait_for_selector('.selection input')
        for name in ('MOM_20', 'MOM_60', 'REV_5', 'RSI_14'):
            page.locator(f'.selection input[value="{name}"]').check()
        page.locator('[name=start]').fill('2024-05-06')
        page.locator('[name=end]').fill('2024-05-31')
        page.locator('[name=mode]').select_option('ic')
        page.locator('button[type=submit]').click()
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('分析完成')", timeout=120000)
        assert page.locator('#comparison tbody tr').count() == 4
        assert page.locator('#results section').count() == 12
        page.screenshot(path=str(output / 'compare.png'), full_page=True)
        page.locator('nav a[href="#/factors"]').click()
        page.wait_for_selector('#factor-list a')
        page.locator('#factor-list a[href="#/factor/MOM_20"]').click()
        page.wait_for_selector('#evaluate-form')
        page.locator('[name=start]').fill('2000-01-04')
        page.locator('button[type=submit]').click()
        page.wait_for_selector('#status.error')
        assert '2400' in page.locator('#status').inner_text()
        # 运行取消必须显示明确状态；不需要等整个三年分析结束。
        page.locator('[name=start]').fill('2024-01-01')
        page.locator('button[type=submit]').click()
        page.locator('#cancel').click()
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('已取消')")
        page.set_viewport_size({'width': 390, 'height': 844})
        page.screenshot(path=str(output / 'mobile.png'))
        assert not errors, errors
        browser.close()
    print('浏览器验收通过：列表搜索、全13指标、两套权重、四因子对比、超限、取消、窄屏；无JS错误')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8800')
    parser.add_argument('--output', default='/tmp/monitor-browser-check')
    args = parser.parse_args()
    check(args.url, args.output)
