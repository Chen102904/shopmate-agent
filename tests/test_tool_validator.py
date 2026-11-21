"""工具参数校验与执行层测试。"""

import json

from app.agent.tool_executor import execute_tool
from app.agent.tool_validator import validate_tool_arguments


def test_valid_arguments_pass():
    result = validate_tool_arguments(
        "get_order",
        {"order_id": "ORD001"},
    )

    assert result.ok is True
    assert result.cleaned == {"order_id": "ORD001"}
    assert result.errors == []


def test_missing_required_argument_fails():
    result = validate_tool_arguments("get_order", {})

    assert result.ok is False
    assert any("order_id" in e for e in result.errors)


def test_empty_string_treated_as_missing():
    result = validate_tool_arguments(
        "get_order",
        {"order_id": "   "},
    )

    assert result.ok is False


def test_wrong_type_fails():
    result = validate_tool_arguments(
        "get_order",
        {"order_id": 123},
    )

    assert result.ok is False
    assert any("类型" in e for e in result.errors)


def test_extra_argument_is_dropped_not_failed():
    """多余参数只剔除并提示，不判失败。

    理由：模型多带一个无关键不改变业务结果，
    判失败会让本来正确的调用白跑一轮。
    """

    result = validate_tool_arguments(
        "get_order",
        {"order_id": "ORD001", "user_id": "USER001"},
    )

    assert result.ok is True
    assert result.cleaned == {"order_id": "ORD001"}
    assert any("user_id" in e for e in result.errors)


def test_boolean_is_not_accepted_as_integer():
    """bool 是 int 的子类，必须单独拦住。"""

    result = validate_tool_arguments(
        "get_order",
        {"order_id": True},
    )

    assert result.ok is False


def test_execute_tool_returns_readable_error_for_missing_argument():
    """模型应收到能看懂的错误，而不是 TypeError 文本。"""

    raw = execute_tool(
        "get_order",
        {},
        user_id="USER001",
    )

    payload = json.loads(raw)

    assert payload["error"] == "工具参数不合法"
    assert payload["details"]
    assert "hint" in payload


def test_execute_tool_still_works_with_valid_arguments():
    raw = execute_tool(
        "get_order",
        {"order_id": "ORD001"},
        user_id="USER001",
    )

    payload = json.loads(raw)

    assert payload["order_id"] == "ORD001"


def test_execute_tool_denies_other_users_order():
    raw = execute_tool(
        "get_order",
        {"order_id": "ORD003"},
        user_id="USER001",
    )

    payload = json.loads(raw)

    assert payload["status"] == "not_accessible"


def test_extra_argument_does_not_break_execution():
    raw = execute_tool(
        "get_order",
        {"order_id": "ORD001", "user_id": "USER999"},
        user_id="USER001",
    )

    payload = json.loads(raw)

    # 多余参数被剔除，仍按请求头身份执行
    assert payload["order_id"] == "ORD001"
