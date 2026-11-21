"""会话长期记忆（滚动摘要）。

解决的问题：
    上下文窗口有限，不能把整段对话都塞给模型。
    原来只取最近 6 条（意图分类）或全部历史（Agent 主链路），
    长对话里早期信息会直接丢失：
        「我的订单号是 ORD001」在第 1 轮说的，
         第 15 轮问「能退吗」时模型已经看不见了。

做法：
    消息条数超过阈值后，把较早的部分压缩成一段摘要，
    存进会话状态；之后每次只带「摘要 + 最近若干条原文」。

    这样早期的关键事实（订单号、诉求、已给过的结论）不会丢，
    同时上下文长度保持有界。

为什么不用简单截断：
    截断会丢掉「用户第一轮说过订单号」这类信息，
    表现为「聊久了就失忆」，是客服场景里最影响体验的问题之一。

代价：
    每次摘要要调一次模型（约 0.5-1 秒）。
    所以设置成「攒够一批才摘要」，不是每条消息都摘要。
"""

import logging

from typing import Any

from app.agent.conversation_store import (
    get_conversation_state,
    update_conversation_state,
)

logger = logging.getLogger(__name__)


# 消息总数超过这个值才启用摘要
SUMMARY_TRIGGER_MESSAGES = 16

# 保留最近多少条原文
SUMMARY_KEEP_RECENT = 8

# 摘要文本长度上限（字符）
SUMMARY_MAX_CHARS = 400

# 摘要覆盖到第几条（存在状态里，避免重复摘要同一段）
STATE_KEY = "conversation_summary"


SUMMARY_PROMPT = """
你是客服对话的摘要助手。

请把下面这段对话压缩成不超过 200 字的中文摘要，保留以下信息：
1. 用户提到的订单编号（例如 ORD001）。
2. 用户的核心诉求（查订单、问物流、要退款、问政策等）。
3. 已经给出的结论或建议。
4. 尚未解决的问题。

要求：
- 不要编造用户没有说过的信息。
- 不要写「用户说」「助手说」这类叙述，直接写事实。
- 如果对话里没有订单号，就不要提订单号。
- 只输出摘要正文，不要标题，不要解释。
"""


# 增量摘要时追加的指令。
#
# 为什么必须合并而不是重写：
#     长对话是分段摘要的（覆盖 0-8、8-20、20-28……）。
#     如果每段摘要都直接覆盖上一次的结果，
#     那么 20 轮之后，第 1 轮说的订单号就彻底消失了——
#     这恰恰是这个模块要解决的问题，不能反过来制造它。
SUMMARY_MERGE_PROMPT = """
你是客服对话的摘要助手。

下面是【已有的对话摘要】和【新增的一段对话】。
请把它们合并成一份不超过 200 字的新摘要。

合并要求：
1. 保留已有摘要里的关键事实，尤其是订单编号和用户诉求，
   即使新增对话里没有再提到，也不能丢。
2. 如果新增对话里的信息与已有摘要冲突（比如订单号变了、诉求变了），
   以新增对话为准，但要保留变化痕迹（例如「订单由 ORD001 改为 ORD002」）。
3. 已经解决的问题可以压缩成一句，尚未解决的问题要写清楚。
4. 不要编造。不要写「用户说」「助手说」。只输出摘要正文。
"""


def _message_text(message: dict) -> str:
    """取出消息的文本内容。"""

    if not isinstance(message, dict):
        return ""

    content = message.get("content")

    return content if isinstance(content, str) else ""


def _render_for_summary(messages: list[dict]) -> str:
    """把消息列表渲染成给摘要模型看的文本。"""

    lines: list[str] = []

    for message in messages:

        role = message.get("role") if isinstance(message, dict) else None
        text = _message_text(message).strip()

        if not text:
            continue

        label = {"user": "用户", "assistant": "助手"}.get(role)

        if not label:
            continue

        lines.append(f"{label}：{text}")

    return "\n".join(lines)


def _generate_summary(
    messages: list[dict],
    previous_summary: str = "",
) -> str:
    """调用模型生成摘要。失败返回空字符串。

    previous_summary 非空时走「合并」而不是「重写」：
        这个函数会被反复调用（每次攒够一批消息就调一次），
        不合并的话，每次都会丢掉上一次摘要里的内容，
        最终只剩最近一批消息的摘要。
    """

    transcript = _render_for_summary(messages)

    if not transcript.strip():
        return previous_summary

    try:
        from app.core.llm_client import (
            DEEPSEEK_MODEL,
            create_chat_completion_with_fallback,
        )

        if previous_summary.strip():

            system_prompt = SUMMARY_MERGE_PROMPT

            user_content = (
                "【已有的对话摘要】\n"
                + previous_summary.strip()
                + "\n\n【新增的一段对话】\n"
                + transcript
            )

        else:

            system_prompt = SUMMARY_PROMPT

            user_content = transcript

        response = create_chat_completion_with_fallback(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )

        summary = (
            response.choices[0].message.content or ""
        ).strip()

        return summary[:SUMMARY_MAX_CHARS]

    except Exception:
        # 摘要失败不影响主流程。
        # 但不要退化成「没有摘要」——那样会丢掉全部早期信息，
        # 不如继续用上一次的摘要（它至少覆盖了更早的部分）。
        #
        # 记日志：摘要一直失败，对话越长模型越「失忆」，
        # 表现是「用户重复说订单号」「助手重复问已答过的问题」，
        # 这类问题从用户侧看像模型变笨了，从日志才能看出是摘要挂了。
        logger.warning("对话摘要生成失败，沿用上一次摘要", exc_info=True)
        return previous_summary


def get_stored_summary(conversation_id: str) -> dict[str, Any] | None:
    """读取已存的摘要及其覆盖范围。"""

    state = get_conversation_state(conversation_id)

    stored = state.get(STATE_KEY)

    if not isinstance(stored, dict):
        return None

    if not isinstance(stored.get("text"), str):
        return None

    return stored


def build_memory_context(
    conversation_id: str,
    history: list[dict],
) -> tuple[str | None, list[dict]]:
    """构造「摘要 + 最近消息」。

    返回：
        (summary, recent_messages)
        summary 为 None 表示历史还短，不需要摘要。

    覆盖判定：
        记录摘要覆盖到第几条（covered）。
        如果 covered 已经覆盖到「需要摘要的边界」，
        就直接复用，不重复调用模型。
    """

    if not history:
        return None, []

    # 历史不够长，直接全量返回
    if len(history) <= SUMMARY_TRIGGER_MESSAGES:
        return None, history

    boundary = len(history) - SUMMARY_KEEP_RECENT

    if boundary <= 0:
        return None, history

    stored = get_stored_summary(conversation_id)

    previous_summary = ""

    if stored is not None and isinstance(stored.get("covered"), int):
        covered = int(stored["covered"])
        previous_summary = (stored.get("text") or "").strip()

        # 已有摘要且覆盖到位，直接复用
        if covered >= boundary:
            return stored.get("text") or None, history[boundary:]

        # 覆盖不够：从上次覆盖处继续往后摘要，避免重复处理老内容
        start = max(0, covered)
    else:
        start = 0

    to_summarize = history[start:boundary]

    if not to_summarize:
        return (
            previous_summary or None,
            history[boundary:],
        )

    # 传入上一次的摘要，让模型「合并」而不是「重写」。
    #
    # 这里再包一层 try：_generate_summary 内部已经有兜底，
    # 但那层兜的是「模型调用失败」。
    # 如果这个函数本身出问题（改了签名、返回类型不对、
    # 依赖的模块导入失败），异常就会绕过它的 except 直接上抛，
    # 一路穿到 run_agent 最外层，用户的整轮对话变成「出了点状况」。
    # 摘要是锦上添花的功能，不值得让整轮对话失败。
    try:
        new_summary = _generate_summary(
            to_summarize,
            previous_summary=previous_summary,
        )
    except Exception:
        logger.warning(
            "对话摘要生成异常，沿用上一次摘要",
            exc_info=True,
        )
        new_summary = ""

    if not new_summary:
        # 摘要失败：退回上一次的摘要（如果有），只带最近消息。
        # 关键是不能把已有的摘要一起丢掉。
        return previous_summary or None, history[boundary:]

    # 只有真的合并成功，才推进 covered。
    # 否则下次会从同一个位置重来，不会漏掉这段对话。
    update_conversation_state(
        conversation_id,
        {
            STATE_KEY: {
                "text": new_summary,
                "covered": boundary,
            }
        },
    )

    return new_summary, history[boundary:]


def clear_memory(conversation_id: str) -> None:
    """清除会话摘要。"""

    update_conversation_state(
        conversation_id,
        {},
        remove_keys=[STATE_KEY],
    )
