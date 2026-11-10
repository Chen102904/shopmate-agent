
from app.rag.retriever import KnowledgeRetriever
from app.core.llm_client import (
    DEEPSEEK_MODEL,
    create_chat_completion_with_fallback,
)


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
    ) -> dict:
        """
        检索知识库，并让大模型根据检索结果回答用户问题。

        返回：
        {
            "answer": "大模型生成的回答",
            "sources": [
                {
                    "source": "知识库文件名",
                    "chunk_id": "文本块ID",
                    "score": 0.8
                }
            ]
        }
        """

        if not query or not query.strip():
            return {
                "answer": "请输入你的问题。",
                "sources": [],
            }

        # 第一步：检索知识库
        results = self.retriever.search(
            query=query,
            top_k=top_k,
        )

        # 第二步：没有检索到相关内容时，直接拒答
        if not results:
            return {
                "answer": "知识库中暂时没有找到相关信息。",
                "sources": [],
            }

        # 第三步：整理知识库上下文
        context_parts = []

        for item in results:
            context_parts.append(
                f"""
来源文件：{item['source']}
文本块ID：{item['chunk_id']}
相似度：{item['score']:.4f}
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
        response = create_chat_completion_with_fallback(
            model=DEEPSEEK_MODEL,
            messages=messages,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
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

        # 第七步：返回回答和引用
        return {
            "answer": answer,
            "sources": sources,
        }
