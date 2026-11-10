import json

from openai import OpenAIError

from app.core.request_context import with_request_context
from app.core.execution_trace import record_execution_event

from app.agent.tools import TOOLS
from app.agent.tool_executor import execute_tool

from app.agent.conversation_store import (
    get_conversation,
    add_message,
    get_pending_clarification,
    set_pending_clarification,
    clear_pending_clarification,
    with_conversation_lock,
)

from app.agent.input_guard import validate_user_input
from app.agent.intent_classifier import classify_intent

from app.core.llm_client import (
    DEEPSEEK_MODEL,
    create_chat_completion_with_fallback,
)


SYSTEM_PROMPT = """
你是 ShopMate 智能电商售后助手。

你的身份：
- 你是 AI 客服助手，不是真人客服。
- 你只能介绍当前项目已经实现的功能。
- 你不能假装自己拥有不存在的业务接口或操作权限。
- 回答时使用简洁、清晰、友好的中文。

一、当前已经实现的功能

目前你只能通过以下四个工具提供业务服务：

1. 订单查询：通过 get_order 查询订单信息。
2. 物流查询：通过 get_logistics 查询物流信息。
3. 退款资格检查：通过 check_refund_eligibility 检查订单是否符合项目模拟退款规则。
4. 售后政策查询：通过 search_after_sales_policy 检索售后知识库。

如果用户询问你能做什么，只能介绍以上四项已实现能力。

二、尚未实现的功能

以下功能目前没有对应的业务工具或接口：

- 直接执行退款或支付操作。
- 修改订单信息。
- 直接取消订单。
- 直接创建或提交退货、换货申请。
- 查询真实退款进度。
- 修改账户信息或密码。
- 申请、修改或重开发票。
- 直接创建售后工单。
- 直接联系商家或人工客服。

用户询问以上功能时：
- 明确说明当前系统暂不支持直接执行该操作。
- 可以根据已有知识库提供相关政策说明。
- 必要时建议用户联系人工客服或使用平台官方入口。
- 绝对不能声称已经替用户完成操作。

三、工具调用规则

1. 订单查询
- 用户询问订单状态、金额、下单时间等信息时，调用 get_order。
- 必须根据工具返回的真实数据回答。
- 不得编造订单信息。

2. 物流查询
- 用户询问快递状态、物流单号、物流进度时，调用 get_logistics。
- 必须根据工具返回的真实数据回答。
- 不得将模拟数据描述成真实物流平台的实时数据。

3. 退款资格检查
- 用户询问自己的某个订单是否可以退款时，调用 check_refund_eligibility。
- 退款资格必须以业务工具返回的结果为准。
- 不允许仅凭知识库政策自行判断某个订单是否符合退款资格。
- 不得声称已经执行退款。

4. 售后政策查询
- 用户询问退款政策、退货条件、换货规则、物流异常处理、人工审核规则时，调用 search_after_sales_policy。
- 用户询问签收超过7天是否还能退款等一般政策问题时，也应调用该工具。
- 回答政策问题时，必须以知识库检索结果为依据。
- 如果知识库要求人工审核，必须明确告知用户。
- 不要编造知识库中没有的信息。
- 不要自行编造文件名、Chunk ID 或相似度。

5. 综合问题
- 如果用户既询问具体订单的退款资格，又询问相关政策，可以同时调用 check_refund_eligibility 和 search_after_sales_policy。
- 综合回答时，要区分订单实际业务判断和知识库政策说明。
- 如果两者存在差异，不允许自行编造结论，应说明情况并建议联系人工客服。

四、退款规则特别说明

- 项目知识库中的签收后7天规则属于 ShopMate 项目的模拟业务规则。
- 不得将其描述成所有电商平台通用的法律规定或行业统一政策。
- 退款资格最终以业务工具返回结果和实际审核为准。
- 商品签收超过7天时，如果知识库要求人工审核，应明确建议用户联系人工客服。

五、通用安全规则

- 不允许编造订单、物流、退款和政策信息。
- 必须根据工具返回的结果回答。
- 如果工具返回错误，要如实告诉用户。
- 不要自行创建退款或售后工单。
- 创建售后工单等有副作用的操作，必须先取得用户明确确认；当前系统尚未实现工单创建工具，因此不能声称能够创建工单。
- 不要索要用户密码、短信验证码或支付验证码。
- 不要把用户没有提供的信息当成事实。
- 结合当前会话的历史消息理解用户的追问。
- 如果用户的问题超出当前系统能力，诚实说明限制，不要自行扩展能力范围。
"""


MODEL_UNAVAILABLE_MESSAGE = (
    "抱歉，ShopMate 的智能服务暂时不可用，"
    "可能是模型服务连接超时或暂时发生故障。"
    "请稍后重试，你的当前会话不会因为这次失败而被清空。"
)


def _record_final_answer(
    answer: str,
    response_type: str,
) -> str:
    """记录最终返回给用户的回答，不保存回答正文。"""

    record_execution_event(
        event="final_answer",
        status="success",
        details={
            "response_type": response_type,
            "answer_length": len(answer),
        },
    )

    return answer


def _save_direct_response(
    conversation_id: str,
    user_message: str,
    answer: str,
    response_type: str = "direct",
) -> str:
    """保存不需要调用业务工具的对话。"""

    add_message(
        conversation_id,
        {
            "role": "user",
            "content": user_message,
        },
    )

    add_message(
        conversation_id,
        {
            "role": "assistant",
            "content": answer,
        },
    )

    return _record_final_answer(
        answer,
        response_type,
    )


def _save_model_error_response(
    conversation_id: str,
) -> str:
    """保存模型不可用时的友好降级回复。"""

    add_message(
        conversation_id,
        {
            "role": "assistant",
            "content": MODEL_UNAVAILABLE_MESSAGE,
        },
    )

    return _record_final_answer(
        MODEL_UNAVAILABLE_MESSAGE,
        "model_unavailable_fallback",
    )


@with_request_context
@with_conversation_lock
def run_agent(
    user_message: str,
    conversation_id: str = "default",
) -> str:
    """运行 ShopMate Agent，支持多轮对话和并发安全。"""

    # 第一步：校验用户输入
    input_result = validate_user_input(user_message)

    if not input_result.valid:
        answer = input_result.error or "输入内容不符合要求。"

        return _record_final_answer(
            answer,
            "input_validation_error",
        )

    user_message = input_result.message

    # 第二步：获取当前会话历史和待澄清状态
    history = get_conversation(conversation_id)

    pending_clarification = get_pending_clarification(
        conversation_id
    )

    # 第三步：结合待澄清状态识别用户意图
    intent_result = classify_intent(
        user_message,
        pending_clarification=pending_clarification,
    )

    intent = intent_result.intent

    # 第四步：处理仍然不明确的问题
    if intent == "unclear":
        answer = (
            intent_result.clarification_question
            or "请问你的订单具体遇到了什么情况？"
        )

        _save_direct_response(
            conversation_id,
            user_message,
            answer,
            response_type=(
                "intent_classification_error"
                if intent_result.error
                else "clarification"
            ),
        )

        # 意图识别服务异常时，不创建错误的待澄清状态
        if intent_result.error is None:
            original_user_message = user_message

            if pending_clarification:
                original_user_message = (
                    pending_clarification.get(
                        "original_user_message",
                        user_message,
                    )
                )

            set_pending_clarification(
                conversation_id=conversation_id,
                original_user_message=original_user_message,
                clarification_question=answer,
            )

        return answer

    # 第五步：用户已经明确表达意图，清除旧的待澄清状态
    if pending_clarification is not None:
        clear_pending_clarification(conversation_id)

    # 第六步：处理不需要进入业务 Agent 的意图
    if intent == "smalltalk":
        return _save_direct_response(
            conversation_id,
            user_message,
            "你好！我是 ShopMate 智能电商售后助手。"
            "目前可以帮你查询订单、查询物流、检查退款资格，"
            "以及检索售后政策。你可以告诉我订单编号或遇到的问题。",
            response_type="smalltalk",
        )

    if intent == "nonsense":
        return _save_direct_response(
            conversation_id,
            user_message,
            "抱歉，我没有理解你的问题。"
            "请重新描述一下你遇到的订单或售后问题。",
            response_type="nonsense",
        )

    if intent == "out_of_scope":
        message = user_message.lower()

        if "发票" in message:
            response = (
                "抱歉，ShopMate 目前暂不支持申请、修改或重开发票。"
                "建议你通过平台官方入口申请发票，或联系人工客服处理。"
            )

        elif (
            "直接退款" in message
            or "帮我退款" in message
            or "给我退款" in message
        ):
            response = (
                "抱歉，ShopMate 目前只能检查订单的退款资格，"
                "暂不支持直接执行退款操作。"
                "如需实际提交退款，请通过平台官方入口操作，或联系人工客服。"
            )

        elif "修改订单" in message or "改订单" in message:
            response = (
                "抱歉，ShopMate 目前暂不支持修改订单。"
                "建议你通过平台官方入口操作，或联系人工客服处理。"
            )

        else:
            response = (
                "抱歉，我是 ShopMate 智能电商售后助手，"
                "目前主要支持订单查询、物流查询、退款资格检查和售后政策查询。"
                "你询问的操作暂未实现，建议联系平台人工客服处理。"
            )

        return _save_direct_response(
            conversation_id,
            user_message,
            response,
            response_type="out_of_scope",
        )

    # 第七步：进入 Agent 工具调用流程
    add_message(
        conversation_id,
        {
            "role": "user",
            "content": user_message,
        },
    )

    # 关键修改：
    # 不再依赖之前读取的内存列表自动追加消息。
    # 每次写入后都从 SQLite 重新读取完整历史。
    history = get_conversation(conversation_id)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *history,
    ]

    collected_sources = []

    max_tool_rounds = 5

    try:
        for _ in range(max_tool_rounds):

            # 优先调用 DeepSeek，失败后自动尝试智谱和千问
            response = create_chat_completion_with_fallback(
                model=DEEPSEEK_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                extra_body={
                    "thinking": {
                        "type": "disabled"
                    }
                },
            )

            message = response.choices[0].message

            # 模型不需要调用工具，直接返回最终回答
            if not message.tool_calls:
                answer = (
                    message.content
                    or "暂时无法生成回复，请稍后重试。"
                )

                if collected_sources:
                    source_lines = []

                    for source in collected_sources:
                        source_lines.append(
                            f"- 文件：{source['source']}；"
                            f"Chunk ID：{source['chunk_id']}；"
                            f"相似度：{source['score']:.4f}"
                        )

                    answer += (
                        "\n\n参考来源（系统检索）：\n"
                        + "\n".join(source_lines)
                    )

                add_message(
                    conversation_id,
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                )

                return _record_final_answer(
                    answer,
                    "agent_response",
                )

            # 保存模型的工具调用消息
            messages.append(
                message.model_dump(exclude_none=True)
            )

            # 执行模型请求的工具
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments_text = tool_call.function.arguments

                try:
                    arguments = json.loads(arguments_text)

                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "工具参数必须是 JSON 对象"
                        )

                    tool_result = execute_tool(
                        tool_name=tool_name,
                        arguments=arguments,
                    )

                    # 提取 RAG 的真实检索来源
                    if tool_name == "search_after_sales_policy":
                        result_data = json.loads(tool_result)

                        rag_result = result_data.get("answer")

                        if isinstance(rag_result, dict):
                            sources = rag_result.get("sources", [])

                            for source in sources:
                                if not isinstance(source, dict):
                                    continue

                                source_name = source.get("source")
                                chunk_id = source.get("chunk_id")
                                score = source.get("score")

                                if (
                                    not source_name
                                    or not chunk_id
                                    or not isinstance(score, (int, float))
                                ):
                                    continue

                                source_item = {
                                    "source": source_name,
                                    "chunk_id": chunk_id,
                                    "score": float(score),
                                }

                                if not any(
                                    item["source"] == source_name
                                    and item["chunk_id"] == chunk_id
                                    for item in collected_sources
                                ):
                                    collected_sources.append(source_item)

                except (json.JSONDecodeError, ValueError) as exc:
                    tool_result = json.dumps(
                        {"error": f"工具参数或结果无效：{exc}"},
                        ensure_ascii=False,
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        answer = "工具调用次数达到上限，请稍后重试。"

        return _record_final_answer(
            answer,
            "tool_round_limit",
        )

    except OpenAIError:
        # 主模型和备用模型均不可用时，友好降级
        return _save_model_error_response(conversation_id)