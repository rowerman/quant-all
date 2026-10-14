"""接口错误类型及 HTTP 映射。"""

from calculate.errors import CalculateError, EvalFactorError
from backtest.errors import BacktestError


class MonitorError(Exception):
    """接口可预期错误；参数为用户提示、HTTP状态和模块错误码。"""

    def __init__(self, message, status=400, code='MONITOR-101'):
        self.message, self.status, self.code = message, status, code
        super().__init__(message)


def error_payload(error, request_id):
    """将异常转为公开错误；参数 request_id 用于日志定位，内部堆栈不发到页面。"""
    if isinstance(error, MonitorError):
        status, code, message = error.status, error.code, error.message
    elif isinstance(error, CalculateError):
        status = 404 if isinstance(error, EvalFactorError) else 400
        code, message = error.code, error.message
    elif isinstance(error, FileNotFoundError):
        status, code, message = 503, 'MONITOR-103', '数据产物缺失，请检查启动预检与服务日志'
    elif isinstance(error, BacktestError):
        status, code, message = 400, error.code, error.message
    else:
        status, code, message = 500, 'MONITOR-104', '计算失败，请用请求编号查看服务日志'
    return status, {'code': code, 'message': message, 'request_id': request_id}
