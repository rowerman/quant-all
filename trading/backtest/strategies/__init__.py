"""用户目标权重提供者目录。

在此目录新增 ``*.py`` 模块并在导入时调用 :func:`backtest.providers.register_provider`，
回测启动时会自动发现。文件名以下划线开头的模块不会被自动导入。
"""
