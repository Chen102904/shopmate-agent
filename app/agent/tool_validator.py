"""工具参数校验。

解决的问题：
    大模型返回的 arguments 是一段自由 JSON，它可能
        - 少传必需参数（{"order_id": ...} 漏掉）
        - 多传不存在的参数（{"order_id": "ORD001", "user": "我"}）
        - 类型不对（{"order_id": 123}）
        - 传空值（{"order_id": ""}）

    不校验的后果：
        1. 缺参数 → Python 抛 TypeError，错误信息是
           "get_order_by_id() missing 1 required positional argument"
           这种给开发者看的话，模型读不懂，也没法修正。
        2. 多参数 → 同样 TypeError。
        3. 类型不对 → 可能在业务层深处才炸，错误位置远离源头。

    所以要在调用业务函数之前，先按工具的 JSON Schema 校验一遍，
    并把错误翻译成模型能理解、能据此重试的话。
"""

from dataclasses import dataclass, field

from app.agent.tools import TOOLS


# 把 TOOLS 列表转成 {工具名: 参数 schema}，方便按名字查
TOOL_PARAMETER_SCHEMAS: dict[str, dict] = {
    tool["function"]["name"]: tool["function"].get("parameters", {})
    for tool in TOOLS
}


# JSON Schema 类型 -> Python 类型
_JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


@dataclass
class ValidationResult:
    """参数校验结果。"""

    ok: bool
    errors: list[str] = field(default_factory=list)
    cleaned: dict = field(default_factory=dict)


def _type_name(value: object) -> str:
    """返回给模型看的类型名。"""

    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def validate_tool_arguments(
    tool_name: str,
    arguments: dict,
) -> ValidationResult:
    """按工具的 JSON Schema 校验参数。

    返回：
        ValidationResult(
            ok=是否通过,
            errors=[人类/模型可读的错误说明],
            cleaned=剔除多余参数后的参数
        )

    设计取舍：
        多余参数不直接判失败，而是剔除后继续执行。
        原因：多传一个无关键（例如模型习惯性带上 user_id）
        不改变业务结果，判失败会让本来正确的调用白跑一轮。
        但会在 errors 里留下记录，方便观测模型的行为。
    """

    schema = TOOL_PARAMETER_SCHEMAS.get(tool_name)

    if schema is None:
        # 未知工具交给上层处理（会返回「未知工具」）
        return ValidationResult(ok=True, cleaned=dict(arguments))

    if not isinstance(arguments, dict):
        return ValidationResult(
            ok=False,
            errors=["工具参数必须是一个 JSON 对象"],
            cleaned={},
        )

    properties = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []

    errors: list[str] = []
    cleaned: dict = {}

    # 1. 缺失必需参数
    missing = [
        name
        for name in required
        if name not in arguments or arguments[name] in (None, "")
    ]

    if missing:
        errors.append(
            "缺少必需参数："
            + "、".join(missing)
            + "。请补充后重新调用。"
        )

    # 2. 多余参数：剔除并记录，不判失败
    extra = [name for name in arguments if name not in properties]

    for name in extra:
        errors.append(
            f"参数 {name} 不属于该工具，已忽略。"
            f"可用参数：{'、'.join(properties) or '无'}。"
        )

    # 3. 逐个检查已提供参数的类型
    for name, value in arguments.items():

        if name not in properties:
            continue

        expected_type = properties[name].get("type")

        if not expected_type:
            cleaned[name] = value
            continue

        allowed = _JSON_TYPE_MAP.get(expected_type)

        if allowed is None:
            cleaned[name] = value
            continue

        # bool 是 int 的子类，要单独排除，避免 True 被当成 integer
        if expected_type in ("integer", "number") and isinstance(value, bool):
            errors.append(
                f"参数 {name} 类型错误：应为 {expected_type}，实际为 boolean。"
            )
            continue

        if not isinstance(value, allowed):
            errors.append(
                f"参数 {name} 类型错误：应为 {expected_type}，"
                f"实际为 {_type_name(value)}。"
            )
            continue

        cleaned[name] = value

    # 4. 字符串参数不允许空串（模型有时会传 "" 占位）
    for name in required:
        value = cleaned.get(name)

        if isinstance(value, str) and not value.strip():
            errors.append(f"参数 {name} 不能为空字符串。")

    # 只有「缺失/类型/空值」才算失败；
    # 「多余参数」只记录提示，不影响执行。
    blocking = [
        error
        for error in errors
        if "不属于该工具" not in error
    ]

    return ValidationResult(
        ok=not blocking,
        errors=errors,
        cleaned=cleaned,
    )
