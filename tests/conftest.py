import pytest

from app.core.api_security import reset_rate_limits
from app.core.db import get_connection


@pytest.fixture(autouse=True)
def reset_rate_limit_counters():
    """每个用例执行前清空限流计数。

    限流是「每 IP 每分钟 60 次」，本来是对线上流量设计的。
    全量跑测试时，同一个进程里几百个请求都来自同一个测试客户端，
    跑到后面就会莫名其妙收到 429——
    表现为「单独跑这个文件是绿的，全量跑就红」，非常难查。

    放在 conftest 里而不是各个测试文件里：
        这是环境问题不是业务问题，
        要求每个写用例的人都记得调一次不现实。
    """

    reset_rate_limits()

    yield


@pytest.fixture(autouse=True)
def clean_ticket_tables():
    """每个用例执行前清空工单表、幂等键表与草稿表。

    工单已经从内存字典改为落 SQLite，数据会跨用例累积。
    不清空的话，第二个用例对同一订单提交同类申请时，
    会被业务去重逻辑直接返回上一个用例建的工单，
    断言看到的就是「别人的」数据了。

    草稿表同理：
    一个用例留下的 open 草稿，会被后面用例的
    「同会话同订单」场景意外复用。
    """

    with get_connection() as connection:
        connection.execute("DELETE FROM after_sales_tickets")
        connection.execute("DELETE FROM idempotency_keys")
        connection.execute("DELETE FROM ticket_drafts")

    yield
