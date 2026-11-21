
import os

from app.rag.retriever import KnowledgeRetriever
from app.rag.query_rewriter import rewrite_query
from app.core.llm_client import (
    DEEPSEEK_MODEL,
    create_chat_completion_with_fallback,
)


# 是否启用查询改写。
#
# 收益：把「东西坏了想换一个」这类口语改写成规范表述，提升检索命中。
# 代价：每次政策问答多一次模型调用（约 0.5-1 秒）和一份 token。
#
# 默认开启；如果延迟敏感或有成本限制，
# 设置环境变量 SHOPMATE_RAG_REWRITE=0 即可关闭，退回原始查询。
ENABLE_QUERY_REWRITE = os.getenv(
    "SHOPMATE_RAG_REWRITE",
    "1",
).strip().lower() not in {"0", "false", "off", "no"}


# 回答生成单独使用的采样温度。
#
# 为什么允许覆盖全局的 0：
#     「回答」和「分类、抽取」对随机性的容忍度不一样。
#     意图分类错一个类别、工具参数错一个字段，整条链路就废了，必须 0。
#     但回答是要给人读的，温度稍高语气更自然，不会每句都像模板。
#
# 不设时（None）沿用统一入口的默认值 0，行为和改造前完全一致。
# 想放宽语气：SHOPMATE_ANSWER_TEMPERATURE=0.4
ANSWER_TEMPERATURE = (
    os.getenv("SHOPMATE_ANSWER_TEMPERATURE") or ""
).strip() or None


SYSTEM_PROMPT = """
你是 ShopMate 智能电商售后助手。

你需要根据系统提供的知识库检索结果回答用户问题。

回答规则：
1. 优先依据知识库内容回答。
2. 不允许编造知识库中没有的售后政策。
3. 如果知识库没有相关信息，明确告诉用户暂时无法确认。
4. 如果政策要求人工审核，必须明确告知用户。
5. 回答要简洁、清晰、友好。
6. 不要自行编造参考来源、文件名、文本块 ID 或相似度。
7. 不要把模拟政策描述成所有电商平台通用的真实规则。

8. 涉及数字时写出具体值。
   知识库写的是「7 天」「1 至 3 个工作日」「48 小时」，
   就写具体数字，不要含糊成「一段时间」「几天内」「很快」。

9. 不得给出绝对承诺。
   即使知识库说「可以申请换货」，也不等于「一定能换成功」——
   因为还受库存、审核、商品状态影响。
   禁止使用「一定可以换」「肯定能退」「保证能通过」这类表述。
   正确说法：「如果符合……条件，可以申请，最终以审核结果为准」。

10. 区分「明确不能」和「无法确认」。
    - 知识库明确写「不支持」「无法」时，直接告诉用户不能，不要含糊成「可能不行」。
    - 知识库没写时，说「暂时无法确认」，并建议联系人工客服。
    两者不能混用：把「明确不能」说成「可能」会让用户白跑一趟；
    把「不知道」说成「不能」会错误拒绝用户的合理诉求。

11. 回答按「结论 — 条件 — 下一步」组织。
    先直接回答能不能，再说适用条件，最后说用户该怎么做。
"""


class RAGService:
    """ShopMate 基础 RAG 问答服务。"""

    def __init__(self):
        self.retriever = KnowledgeRetriever()
        self.retriever.build_index()

    def answer(
        self,
        query: str,
        top_k: int = 3,
        use_rewrite: bool | None = None,
        temperature: float | None = None,
    ) -> dict:
        """
        检索知识库，并让大模型根据检索结果回答用户问题。

        参数：
            query: 用户原始问题
            top_k: 检索条数
            use_rewrite: 是否改写查询。
                         None 表示读环境变量 SHOPMATE_RAG_REWRITE（默认开）。
                         评测时可以显式关掉，做「改写有没有用」的对照实验。
            temperature: 回答生成温度。
                         None 表示读 SHOPMATE_ANSWER_TEMPERATURE，
                         再没有就沿用全局默认 0。
                         只有回答这一步受影响，意图分类和工具抽取始终是 0。

        返回：
        {
            "answer": "大模型生成的回答",
            "sources": [{"source","chunk_id","score"}],
            "search_query": "实际用于检索的查询（改写后或原始）"
        }
        """

        if not query or not query.strip():
            return {
                "answer": "请输入你的问题。",
                "sources": [],
                "search_query": "",
                "kb_version": self.retriever.version,
            }

        # 第一步：按需改写查询
        # 改写失败会自动返回原始问题（见 query_rewriter），不会中断流程。
        should_rewrite = (
            ENABLE_QUERY_REWRITE
            if use_rewrite is None
            else use_rewrite
        )

        rewritten = (
            rewrite_query(query)
            if should_rewrite
            else query
        )

        # 双路检索：原始问题 + 改写问题。
        # 改写有收益时它加分，改写跑偏时原始问题兜底。
        search_queries = [query]

        if rewritten and rewritten.strip() and rewritten.strip() != query.strip():
            search_queries.append(rewritten)

        # 第二步：检索知识库
        results = self.retriever.search_multi(
            search_queries,
            top_k=top_k,
        )

        search_query = " || ".join(search_queries)

        # 第三步：没有检索到相关内容时，直接拒答
        if not results:
            return {
                "answer": "知识库中暂时没有找到相关信息。",
                "sources": [],
                "search_query": search_query,
                "kb_version": self.retriever.version,
            }

        # 第四步：整理知识库上下文
        context_parts = []

        for item in results:

            # 降级检索（BM25 兜底）的分数是「关键词相对分」，
            # 不是语义相似度。
            # 标成「相似度」会误导模型：它会以为这是语义匹配，
            # 回答时更笃定；实际可能只是碰巧命中了同一个词。
            # 如实标注，模型才会在表述上保守一些。
            score_label = (
                "关键词匹配度"
                if item.get("retrieval_mode") == "bm25_only"
                else "相似度"
            )

            context_parts.append(
                f"""
来源文件：{item['source']}
文本块ID：{item['chunk_id']}
{score_label}：{item['score']:.4f}
内容：
{item['content']}
"""
            )

        context = "\n\n".join(context_parts)

        # 第四步：构造模型消息
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
请根据以下知识库内容回答我的问题。

【知识库检索结果】
{context}

【用户问题】
{query}
""",
            },
        ]

        # 第五步：调用统一的三模型 Fallback
        #
        # 温度只在这一步生效：回答要给人读，语气可以略有变化。
        # 意图分类、查询改写、工具参数抽取都不走这里，始终用全局的 0。
        llm_kwargs: dict = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "extra_body": {
                "thinking": {
                    "type": "disabled"
                }
            },
        }

        effective_temperature = (
            temperature
            if temperature is not None
            else ANSWER_TEMPERATURE
        )

        if effective_temperature is not None:
            llm_kwargs["temperature"] = float(effective_temperature)

        response = create_chat_completion_with_fallback(
            **llm_kwargs,
        )

        answer = (
            response.choices[0].message.content
            or "暂时无法生成回答，请稍后重试。"
        )

        # 第六步：由 Python 程序生成可信的引用信息
        sources = []

        for item in results:
            sources.append(
                {
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "score": round(float(item["score"]), 4),
                }
            )

        # 第八步：返回回答和引用
        return {
            "answer": answer,
            "sources": sources,
            "search_query": search_query,
            "rewritten": search_query != query,
            # 这次回答是基于哪一版知识库检索出来的。
            # 排查「昨天说的和今天不一样」时，第一个要看的就是它。
            "kb_version": self.retriever.version,
        }
