import json
import logging
import re

from openai import OpenAIError

from app.core.circuit_breaker import AllProvidersCircuitOpen
from app.core.request_context import with_request_context
from app.core.execution_trace import record_execution_event
from app.core.identity import with_user_identity

from app.agent.tools import TOOLS
from app.agent.tool_executor import execute_tool
from app.agent.tool_cache import (
    build_cache_key,
    get_cached_result,
    store_tool_result,
)

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
from app.agent.memory import build_memory_context

from app.core.llm_client import (
    DEEPSEEK_MODEL,
    create_chat_completion_with_fallback,
)


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
你是 ShopMate 智能电商售后助手。

你的身份：
- 你是 AI 客服助手，不是真人客服。
- 你只能介绍当前项目已经实现的功能。
- 你不能假装自己拥有不存在的业务接口或操作权限。
- 回答时使用简洁、清晰、友好的中文。

一、当前已经实现的功能

目前已实现以下八个工具：

1. 订单查询：通过 get_order 查询订单信息。
2. 物流查询：通过 get_logistics 查询物流信息。
3. 退款资格检查：通过 check_refund_eligibility 检查订单是否符合项目模拟退款规则。
4. 售后政策查询：通过 search_after_sales_policy 检索售后知识库。
5. 售后申请草稿：通过 draft_after_sales_ticket 生成待确认的申请草稿（不落地）。
6. 提交售后申请：通过 confirm_after_sales_ticket 正式创建工单（会落地，进人工审核）。
7. 查询售后进度：通过 get_after_sales_tickets 查已提交工单的状态、
   以及有没有还没提交的草稿。
8. 撤销申请：通过 cancel_after_sales_ticket 撤销售后申请（有副作用）。

如果用户询问你能做什么，可以介绍这几项能力，
但要说明「提交申请后需要人工审核，不能保证通过」。

二、尚未实现的功能

以下功能目前没有对应的业务工具或接口：

- 直接执行退款或打款到账。
- 修改订单信息。
- 直接取消订单。
- 修改账户信息或密码。
- 申请、修改或重开发票。
- 直接联系商家或人工客服。
- 修改已提交工单的状态或撤销工单。

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

5. 综合问题与多诉求

- 用户一条消息里提出多件事时，每一件都要做，不能只做最后一件。
  例如「查 ORD001 订单，再看能不能退款」：
  先用 get_order 拿到订单详情，再用 check_refund_eligibility 判断资格，
  回答里两部分都要体现。
- 注意 get_order 描述里的「检查退款资格不需要先调用本工具」，
  只适用于用户单纯问「这单能不能退」的场景。
  用户明确说要查订单信息时，必须调用 get_order，
  否则用户会拿不到金额、下单时间这些他要的东西。
- 如果用户既询问具体订单的退款资格，又询问相关政策，可以同时调用 check_refund_eligibility 和 search_after_sales_policy。
- 综合回答时，要区分订单实际业务判断和知识库政策说明。
- 如果两者存在差异，不允许自行编造结论，应说明情况并建议联系人工客服。

6. 提交售后申请（重要：两步走）

用户要求申请退款、退货、换货时，必须严格按以下两步执行：

第一步：调用 draft_after_sales_ticket 生成草稿。
- 需要三个参数：订单号、申请类型（refund/return/exchange）、理由。
- 理由必须来自用户原话或已确认的事实，不允许自行编造。
- 如果用户没说清楚要退款还是退货换货，或者没给理由，先用文字追问，不要猜。
- 这个工具不会创建工单，只是把内容冻结下来。

第一步之后，你必须把草稿内容完整告诉用户：
- 订单号
- 申请类型
- 申请理由
- 然后明确问一句「是否确认提交」。

第二步：只有用户明确表示同意后，才调用 confirm_after_sales_ticket。
- 通常不需要传 draft_id：省略时系统自动使用本会话最近一张待确认草稿。
- 千万不要自己编造 draft_id，也不要为了拿到编号重新调用 draft 工具。
- 用户说「确认」「可以」「提交吧」这类明确同意才算数。
- 用户说「先不」「算了」「再想想」，就不要调用，并告知草稿未提交。

绝对禁止：
- 禁止在同一轮里连续调用这两个工具。用户必须有机会看到内容再决定。
- 禁止用户只是询问流程、耗时、条件时就去建单。
- 禁止没拿到 draft_id 就调用 confirm，也不要自己编造 draft_id。
- 禁止声称已经提交成功，除非 confirm 工具真的返回了工单号。
- **禁止在没有成功调用 draft_after_sales_ticket 的情况下，说「已为你生成草稿」
  或直接列出草稿内容。**
  只有工具返回了 draft_id，才存在草稿；没调用就描述草稿，用户下一轮说「确认」
  会因为没有草稿而失败，他会以为申请丢了。
  如果这一轮先查了订单或退款资格、还没来得及生成草稿，
  就如实说「我先帮你确认了 X，接下来生成草稿」，然后在本轮内调用 draft 工具。

7. 查询售后进度

用户提交申请之后，一定会追问进度：
「我那个退款怎么样了」「审核到哪一步了」「我的工单是什么状态」。

- 这类问题一律调用 get_after_sales_tickets，不要凭记忆作答，
  也不要重复走一遍 draft → confirm。
- 用户给了订单号（ORD001）或工单号（UUID）就原样传进 keyword；
  什么都没给就不传，系统会返回最近的工单。
- 结果里 pending_draft 不为空，说明用户有一张**生成了但没提交**的草稿：
  要明确告诉他「这张申请还没提交」，把内容复述一遍并问是否现在提交。
  绝对不能说成「您已经申请过了」——那会让他以为在等审核。
- 查不到工单时如实说没查到，不要编造工单号、审核结果或处理时间。
- 工单状态含义：pending 待人工审核、approved 审核通过、
  rejected 审核未通过、processing 处理中、completed 已完成、
  cancelled 已撤销。
  回答时用中文说法，不要说 pending / approved 这种枚举值。
- 订单状态同理，一律说中文：
  pending → 待付款，paid → 已付款，shipped → 已发货，
  delivered → 已送达，cancelled → 已取消。
  严禁输出「已支付（paid）」这种中英并列的写法，括号里的枚举值对用户没有意义。

8. 撤销已提交的申请

- 用户明确表示不想申请了（「算了不退了」「帮我撤销」「取消这个申请」）时，
  调用 cancel_after_sales_ticket。
- 只有「待人工审核」的申请能撤销。已审核完成或已经撤销过的不能撤。
- 如果工具返回 confirmation_required，说明用户还没明确确认：
  把「要撤销的是哪张单、撤销后就不会进入审核」讲清楚，
  等用户回复「确认」「撤销吧」之后再调用一次。
- **用户只是催进度、抱怨太慢、问什么时候能好——这些都不是撤销。**
  这种情况用 get_after_sales_tickets 查状态，绝对不能撤销。
  把「催单」当成「撤销」，等于把用户的申请弄丢了。
- 撤销成功后可以告诉用户「如果还需要，可以重新申请」。
- 不要说成「已退款」或「钱会退给你」：撤销申请不等于退款。

四、退款规则特别说明

- 项目知识库中的签收后7天规则属于 ShopMate 项目的模拟业务规则。
- 不得将其描述成所有电商平台通用的法律规定或行业统一政策。
- 退款资格最终以业务工具返回结果和实际审核为准。
- 商品签收超过7天时，如果知识库要求人工审核，应明确建议用户联系人工客服。

五、通用安全规则

- 不允许编造订单、物流、退款和政策信息。
- 必须根据工具返回的结果回答。
- 如果工具返回错误，要如实告诉用户。
- 只能查询当前用户本人的订单。
- 当工具返回「订单不存在或无权访问」时：
  - 直接按这个结论回答，不要猜测订单是否真实存在。
  - 不要反复更换订单号试探。
  - 不要暗示该订单可能属于其他用户。
  - 可以请用户核对订单号，或建议联系人工客服。
- 有副作用的操作（提交售后申请）必须先取得用户明确确认，
  并且只能通过 draft → confirm 两步完成。
- 不要索要用户密码、短信验证码或支付验证码。
- 不要把用户没有提供的信息当成事实。
- 结合当前会话的历史消息理解用户的追问。
- 如果用户的问题超出当前系统能力，诚实说明限制，不要自行扩展能力范围。

六、回答风格

- 只调用回答问题所必需的工具。
  历史对话里已经查过、且信息不会变化的订单或物流，
  不要重复调用同一个工具。
  例如用户先查了订单状态，接着问「能不能退」，
  只需要调用退款资格检查，不必再查一次订单。
- 不得给出绝对承诺。
  即使工具判断符合退款条件，也只能说「符合退款条件，可以申请」，
  不能说「一定能退成功」——实际仍需平台审核。
  禁止「一定可以」「肯定能」「保证能通过」这类表述。
- 涉及天数、金额、次数时写出具体数值，不要用「一段时间」「几天内」含糊带过。
- 回答按「结论 — 条件 — 下一步」组织，先说能不能，再说条件，最后说怎么做。
"""


MODEL_UNAVAILABLE_MESSAGE = (
    "抱歉，ShopMate 的智能服务暂时不可用，"
    "可能是模型服务连接超时或暂时发生故障。"
    "请稍后重试，你的当前会话不会因为这次失败而被清空。"
)


# 订单号格式：ORD + 数字
ORDER_ID_PATTERN = re.compile(r"ORD\d+", re.IGNORECASE)


def _find_referenced_order_id(
    history: list[dict],
    current_message: str,
) -> str | None:
    """找出用户上文提到的订单号，用于补全多轮对话里的指代。

    场景：
        第 1 轮：帮我查 ORD001
        第 2 轮：那这个能退吗     ← 消息里没有订单号

    如果不补，模型会追问「请问是哪个订单」，用户已经说过一遍。
    这里从历史里取出最近提到的订单号，作为提示交给模型。

    只做「提示」，不做「强制替换」：
        如果当前问题与订单无关（例如问一般政策），
        模型可以忽略这个提示，避免把政策问题错误绑定到某个订单。
    """

    if ORDER_ID_PATTERN.search(current_message or ""):
        return None

    for message in reversed(history or []):

        if not isinstance(message, dict):
            continue

        content = message.get("content") or ""

        # 跳过与当前消息内容相同的那条（历史末尾通常是当前消息）
        if content.strip() == (current_message or "").strip():
            continue

        matches = ORDER_ID_PATTERN.findall(content)

        if matches:
            return matches[-1].upper()

    return None


def _build_order_hint_text(order_id: str) -> str:
    """构造附加在用户消息后方的系统提示。"""

    return (
        f"\n\n（系统提示：用户在本次对话上文提到的订单号是 {order_id}。"
        "如果当前问题需要订单号，直接使用它，不要再向用户索要。"
        "这只是补全参数，不是让你去查订单："
        "除非用户明确问订单状态或金额，否则不要调用 get_order；"
        "历史里已经查过的订单信息也不用重复查询。"
        "如果当前问题明显与订单无关，请忽略本提示。）"
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
    """保存模型不可用时的友好降级回复。

    注意这里自己也可能失败：
        模型不可用的常见原因之一就是整个服务不健康，
        这时候连「写一条消息进数据库」都可能抛异常。
        如果不在内部兜住，降级逻辑本身会变成新的异常源，
        等于兜了个寂寞。
    """

    try:
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

    except Exception:
        logger.warning(
            "降级回复写入失败，直接返回文案 | conversation=%s",
            conversation_id,
            exc_info=True,
        )

        return MODEL_UNAVAILABLE_MESSAGE


# 未预期异常的对外文案。
#
# 措辞刻意不写「服务器错误」：
#   用户不关心是谁的问题，只想知道「是不是我操作错了」。
#   明确说「不是你输入的问题」，能减少他反复重试和自我怀疑。
UNEXPECTED_ERROR_MESSAGE = (
    "抱歉，处理你的问题时出了点状况，可以稍后再试一次。"
    "如果反复出现，建议联系人工客服处理。"
)


def _save_unexpected_error_response(
    conversation_id: str,
) -> str:
    """保存「未预期异常」的降级回复。

    和 _save_model_error_response 一样，内部再兜一层：
    能走到这里说明服务已经不太健康，落库未必成功。
    """

    try:
        add_message(
            conversation_id,
            {
                "role": "assistant",
                "content": UNEXPECTED_ERROR_MESSAGE,
            },
        )

        return _record_final_answer(
            UNEXPECTED_ERROR_MESSAGE,
            "unexpected_error_fallback",
        )

    except Exception:
        logger.warning(
            "未预期异常的降级回复写入失败，直接返回文案 | conversation=%s",
            conversation_id,
            exc_info=True,
        )

        return UNEXPECTED_ERROR_MESSAGE


# 装饰器顺序有讲究：
#
#   with_request_context 必须在最外层。
#   它会 pop 掉内部调用用的 _include_request_metadata 参数，
#   而内层 with_user_identity 会按被包装函数的签名做 bind_partial——
#   如果顺序反了，bind_partial 拿到一个签名里没有的关键字，
#   直接抛 TypeError，整个 /chat 接口不可用。
#
#   这个顺序错误真实存在过：所有 /chat 测试都 mock 了 run_agent，
#   所以从没跑到过这条链路，直到手工调接口才炸出来。
#
#   with_user_identity 放在 request_context 里面、锁的外面：
#   这样从拿锁开始的所有步骤都在正确的用户上下文里。
@with_request_context
@with_user_identity
@with_conversation_lock
def run_agent(
    user_message: str,
    conversation_id: str = "default",
    user_id: str | None = None,
) -> str:
    """运行 ShopMate Agent，支持多轮对话和并发安全。

    这一层只做一件事：**保证没有异常能穿透到 API**。

    为什么需要最外层兜底：
        内部的 try 只包住了工具调用循环（从读历史之前就开始了），
        而「取会话历史」「写用户消息」「构造摘要」这些数据库操作
        在 try 外面。数据库一抖动，异常就直冲到 FastAPI 的
        全局处理器，用户收到裸的 500「服务器内部错误」，
        前面已经花掉的模型调用也白费。

        对用户来说，「服务暂时不可用」和「服务器内部错误」是同一件事，
        但前者至少告诉他「不是你说错了」，而不是让他怀疑自己。

    参数：
        user_message: 用户消息
        conversation_id: 会话 ID
        user_id: 当前用户编号。
                 由 API 层从请求头读取后传入；
                 不传时回落到演示账号（见 app/core/identity.py）。
                 工具执行时会用它做订单归属校验，
                 所以模型无法通过订单号查到别人的数据。
    """

    try:
        return _run_agent_impl(user_message, conversation_id, user_id)

    except (OpenAIError, AllProvidersCircuitOpen):
        # 内部已经处理过一次，这里是防御：
        # 万一某条新路径漏了，也不能让用户看到 500。
        return _save_model_error_response(conversation_id)

    except Exception as exc:
        # 数据库异常、类型错误、第三方库 bug 都落在这里。
        #
        # 不把 str(exc) 交给用户：内部信息可能包含文件路径、
        # SQL 片段、甚至密钥片段。只记日志，给用户一句能理解的说明。
        logger.exception(
            "Agent 主流程出现未预期异常 | conversation=%s | error_type=%s",
            conversation_id,
            type(exc).__name__,
        )

        record_execution_event(
            event="agent_run",
            status="error",
            details={
                "error_type": type(exc).__name__,
                "stage": "unhandled",
            },
        )

        return _save_unexpected_error_response(conversation_id)


def _run_agent_impl(
    user_message: str,
    conversation_id: str,
    user_id: str | None,
) -> str:
    """Agent 主流程实现。

    异常不在这里统一捕获——交给外层 run_agent。
    这样新增步骤时不用每加一段就多写一个 try。
    """

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

    # 第三步：结合待澄清状态和历史识别用户意图
    # history 让分类器能理解「这个」「那单」指向上一轮提到的订单，
    # 否则第二轮追问会被判成 unclear。
    intent_result = classify_intent(
        user_message,
        pending_clarification=pending_clarification,
        history=history,
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
            "目前可以帮你查询订单、查询物流、检查退款资格、检索售后政策，"
            "也可以帮你整理并提交售后申请（提交前会先给你确认）。"
            "你可以告诉我订单编号或遇到的问题。",
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

        elif any(
            keyword in message
            for keyword in (
                "直接退款",
                "帮我退款",
                "给我退款",
                "退到我卡",
                "退我卡",
                "打款",
                "把钱退",
                "退钱",
            )
        ):
            # 注意这里不能简单说「不支持」：
            # 系统现在能提交退款申请工单了，
            # 拒绝得太干净会把用户推走，也是能力描述不准确。
            # 要区分「不能直接打款」和「能提交申请」两件事。
            response = (
                "ShopMate 不能直接执行退款打款，退款需要人工审核。"
                "但我可以帮你提交一张退款申请工单："
                "请告诉我订单号和退款理由，"
                "我先把申请内容整理出来给你确认，你确认后才会正式提交。"
            )

        elif "修改订单" in message or "改订单" in message:
            response = (
                "抱歉，ShopMate 目前暂不支持修改订单。"
                "建议你通过平台官方入口操作，或联系人工客服处理。"
            )

        else:
            response = (
                "抱歉，我是 ShopMate 智能电商售后助手。"
                "我可以帮你查询订单、查询物流、检查退款资格、检索售后政策，"
                "也可以帮你整理并提交售后申请（提交前会让你确认）。"
                "你询问的这个操作暂时做不到，建议联系平台人工客服处理。"
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

    # 长期记忆：历史过长时，把较早部分压缩成摘要，
    # 只带「摘要 + 最近若干条」。
    # 否则第 15 轮时，第 1 轮说过的订单号已经被挤出上下文。
    summary, recent_history = build_memory_context(
        conversation_id,
        history,
    )

    if summary:
        messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\n\n【本次会话前文摘要】\n"
                    + summary
                    + "\n（这是更早对话的摘要，用于理解上下文，"
                    "不要原样复述给用户。）"
                ),
            },
            *recent_history,
        ]

    # 多轮指代补全：当前消息没写订单号，但上文提过，就给模型一个提示。
    # 只改发给模型的内容，不改数据库里存的原始消息，
    # 保证对话记录还原的是用户真实说过的话。
    referenced_order_id = _find_referenced_order_id(
        history,
        user_message,
    )

    if (
        referenced_order_id
        and intent
        in {
            "order_query",
            "logistics_query",
            "refund_query",
            "return_query",
            "exchange_query",
        }
        and messages
        and messages[-1].get("role") == "user"
    ):
        messages[-1] = {
            "role": "user",
            "content": (
                user_message
                + _build_order_hint_text(referenced_order_id)
            ),
        }

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

                    # 会话级结果缓存：
                    # 模型重复查同一个订单时，直接返回上次结果，
                    # 不再真的查一遍（多轮对话里很常见）。
                    cache_key = build_cache_key(
                        tool_name,
                        arguments,
                    )

                    cached_result = get_cached_result(
                        conversation_id,
                        cache_key,
                    )

                    if cached_result is not None:
                        tool_result = cached_result

                        record_execution_event(
                            event="tool_cache_hit",
                            status="success",
                            details={"tool": tool_name},
                        )

                        logger.info(
                            "工具结果命中会话缓存 | tool=%s | "
                            "conversation=%s",
                            tool_name,
                            conversation_id,
                        )

                    else:
                        tool_result = execute_tool(
                            tool_name=tool_name,
                            arguments=arguments,
                            conversation_id=conversation_id,
                        )

                        store_tool_result(
                            conversation_id,
                            cache_key,
                            tool_result,
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
                    # 不带异常原文：JSONDecodeError 会把出错的片段
                    # 原样带出来（例如 "Expecting ',' delimiter: line 1 column 87"），
                    # 塞进模型上下文等于给它一段看不懂的噪声，
                    # 它只会回一句「数据格式异常」敷衍用户。
                    logger.warning(
                        "工具结果解析失败 | tool=%s | error=%s",
                        tool_name,
                        type(exc).__name__,
                    )

                    tool_result = json.dumps(
                        {
                            "error": "这次查询没有返回可用的结果",
                            "status": "unparsable_result",
                            "hint": (
                                "如实告诉用户这次没查到，建议稍后重试；"
                                "不要编造数据，也不要把内部错误细节说给用户。"
                            ),
                        },
                        ensure_ascii=False,
                    )

                except Exception as exc:
                    # 工具层意外异常（网络中断、数据库故障、代码 bug）
                    # 不能穿透到调用方：
                    # 一旦穿透，整轮对话直接 500，用户什么都拿不到，
                    # 前面已经花掉的模型调用也白费了。
                    # 这里转成模型能理解的错误，让它继续或如实告知用户。
                    #
                    # 注意：不能把 str(exc) 交给模型或用户，
                    # 内部异常信息可能包含路径、SQL、密钥片段。
                    logger.exception(
                        "工具执行出现未预期异常 | tool=%s | error_type=%s",
                        tool_name,
                        type(exc).__name__,
                    )

                    record_execution_event(
                        event="tool_result",
                        status="error",
                        details={
                            "tool": tool_name,
                            "error_type": type(exc).__name__,
                            "unexpected": True,
                        },
                    )

                    tool_result = json.dumps(
                        {
                            "error": "工具暂时不可用",
                            "hint": "请稍后重试，或建议用户联系人工客服。",
                        },
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

        # 这条回复也要落库。
        # 不落库的话，用户下一轮追问时，模型看不到自己上一轮说过
        # 「达到上限」，会以为对话是空的，很可能再走一遍同样的工具调用，
        # 又一次撞上限。
        add_message(
            conversation_id,
            {
                "role": "assistant",
                "content": answer,
            },
        )

        return _record_final_answer(
            answer,
            "tool_round_limit",
        )

    except (OpenAIError, AllProvidersCircuitOpen):
        # 主模型和备用模型均不可用（或全部处于熔断）时，友好降级
        return _save_model_error_response(conversation_id)