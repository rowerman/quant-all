"""启动预检：仅验证本地依赖与数据，不取数、不建库。"""

from pathlib import Path

from backend import service
from calculate.config import DB_FILE, FACTOR_DB_FILE

# 前端与后端同在 apps/monitor 下，用文件位置定位，不再依赖仓库根路径
FRONTEND_DIR = Path(__file__).resolve().parents[1] / 'frontend'


def check():
    """核验依赖、数据库、已落盘因子及本地图表文件，失败向调用者反馈。"""
    for path in (DB_FILE, FACTOR_DB_FILE, FRONTEND_DIR / 'index.html',
                 FRONTEND_DIR / 'vendor/echarts.min.js'):
        if not Path(path).is_file():
            raise FileNotFoundError(f'启动预检：缺少 {path}')
    choices = service.options()
    factors = service.factor_list()
    if not factors:
        raise RuntimeError('没有可评估的已落盘因子')
    print(f"预检通过：{len(factors)} 因子，{choices['start']}～{choices['end']}，单worker/2计算线程/4等待")


if __name__ == '__main__':
    check()
