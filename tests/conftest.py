import pytest

from app.core.api_security import reset_rate_limits


@pytest.fixture(autouse=True)
def reset_rate_limit_counters():
    """每个用例执行前清空限流计数。

    限流是「每 IP 每分钟 60 次」，本来是对线上流量设计的。
    全量跑测试时，同一个进程里几百个请求都来自同一个测试客户端，
    跑到后面就会莫名其妙收到 429——
    表现为「单独跑这个文件是绿的，全量跑就红」，非常难查。
    """

    reset_rate_limits()

    yield
