"""因子间相关性指标声明。"""

from calculate.correlation import compute_correlation
from calculate.metrics.base import Metric, MetricParam, register_metric

register_metric(Metric(key='correlation', label='与已有因子的相关性', stage='correlation',
                       compute=compute_correlation,
                       params=(MetricParam('top_n', '相关因子数量', 'int', 10),)))
