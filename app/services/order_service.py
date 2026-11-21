from app.core.demo_mode import resolve_demo_owner
from app.models.order import Order
from data.mock_data import ORDERS


def list_orders_by_user(user_id: str) -> list[Order]:
    """列出某个用户名下的所有订单，最近下单的在前。

    和 get_order_by_id 用同一套归属规则（含演示模式的访客映射），
    否则会出现「列表里看得到、点进去查不到」的矛盾。

    只认订单自己的 user_id，不看工单：
        工单是「对订单的售后动作」，订单归属不会因为有人申请售后而改变。
    """

    if not user_id:
        return []

    lookup_user = resolve_demo_owner(user_id)

    owned = [order for order in ORDERS.values() if order.user_id == lookup_user]

    # 下单时间是 mock 数据里按 now - N 天算出来的，
    # 排序用它而不是字典顺序：字典顺序只是录入顺序，没有业务含义。
    owned.sort(key=lambda item: item.order_time, reverse=True)

    return owned


def get_order_by_id(
    order_id: str,
    user_id: str,
) -> Order | None:
    """根据订单编号查询订单，并校验订单归属。

    参数：
        order_id: 订单编号
        user_id: 当前用户编号，必须是该订单的归属人

    返回：
        订单对象；订单不存在、或不属于 user_id 时返回 None。

    为什么两种情况都返回 None：
        如果「不存在」和「无权访问」返回不同结果，
        攻击者可以用订单号逐个试探，从而判断系统里有哪些订单存在。
        统一返回值可以避免这种枚举。
    """

    if not order_id or not user_id:
        return None

    order = ORDERS.get(order_id)

    if order is None:
        return None

    # 演示模式：访客共用演示账号的订单。
    #
    # 不这么做的话，别人打开链接后查什么都是「订单不存在或无权访问」，
    # 根本没法测。所以把 guest-xxx 映射到演示账号。
    #
    # 注意这里只映射「查订单的账号」，工单仍然按访客自己的 id 存，
    # 所以 A 建的工单不会出现在 B 的列表里。
    #
    # 越权保护还在：guest 查 USER002 的 ORD003 依然会被拒，
    # 这个分支仍然走下面那句 user_id != order.user_id。
    lookup_user = resolve_demo_owner(user_id)

    if order.user_id != lookup_user:
        return None

    return order
