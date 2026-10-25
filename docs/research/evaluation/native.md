# 原生计算阶段保护（验证中）

research/evaluation/calculate/native.py提供共享可重入锁及同步函数/事件生成器保护，由后端service和scheduler调用；使用jsonutil转换事件，不读写数据资产。

- native_section(cancelled=None)：保护一次原生操作；提供取消事件时分段等待锁，取消后不进入计算。
- serialized(function)：保护同步元数据调用，允许同线程嵌套。
- serialized_events(function)：锁内创建、推进、JSON转换和关闭，交付纯Python事件后释放锁。

两个任务可在阶段边界交替推进。事件队列背压时不持原生锁；取消不能中断正在执行的原生阶段。scheduler在预检前reserve，使等待元数据锁的计算请求也计入六任务容量。

修改注意：不能跨yield和网络/队列等待持锁，也不能只保护SQL而遗漏转换、close和对象释放；新服务元数据入口使用同一保护。锁只覆盖后端入口，不保证直接并行调用calculate.evaluate等底层函数安全。

SIGSEGV具体原生根因未知。阶段隔离会牺牲吞吐；当前仅有局部回归证据，真实重复双任务、最终全集和浏览器复验待续，不能声称稳定验收。
