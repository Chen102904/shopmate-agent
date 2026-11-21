
import logging

from app.core.llm_client import (
    DEEPSEEK_MODEL,
    create_chat_completion_with_fallback,
)

logger = logging.getLogger(__name__)


REWRITE_PROMPT = """
你是一个搜索查询优化助手，负责改写电商售后领域的用户问题。

你的任务：
将用户的口语化问题改写成适合检索售后政策知识库的简洁查询。

改写规则：
1. 保留用户原本的问题意图，不得改变含义。
2. 将口语表达转换成清晰、规范的书面表达。
3. 尽量补充明确的业务关键词，例如退款、退货、换货、物流。
4. 不要回答用户的问题。
5. 不要添加用户没有提供的订单信息或事实。
6. 只输出改写后的查询，不要解释，不要添加引号。

示例：

用户问题：快递怎么一直不动了？
改写结果：物流长时间没有更新应该如何处理？

用户问题：东西坏了想换一个怎么办？
改写结果：商品存在质量问题时如何申请换货？

用户问题：超过七天了还能退钱不？
改写结果：商品签收超过7天是否还能申请退款？
"""


def rewrite_query(query: str) -> str:
    """
    使用大模型将用户的口语化问题改写成适合知识库检索的查询。

    如果改写失败，则返回用户原始问题，保证检索流程可以继续。
    """

    if not query or not query.strip():
        raise ValueError("查询内容不能为空")

    original_query = query.strip()

    try:
        # 走统一的三模型 fallback：
        # 改写只是辅助步骤，主模型不可用时用备用模型即可，
        # 不应该因为一次改写失败就退化成原始查询。
        response = create_chat_completion_with_fallback(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": REWRITE_PROMPT,
                },
                {
                    "role": "user",
                    "content": original_query,
                },
            ],
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )

        rewritten_query = (
            response.choices[0].message.content or ""
        ).strip()

        # 防止模型返回空内容
        if not rewritten_query:
            return original_query

        return rewritten_query

    except Exception:
        # 查询改写失败时，降级使用原始查询。
        # 降级本身对用户无感，但必须留日志：
        # 改写关掉后口语化提问的命中率会下降，
        # 不记日志就查不出「为什么最近检索变差」。
        logger.warning("查询改写失败，改用原始查询", exc_info=True)
        return original_query
