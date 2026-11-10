
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.knowledge_pipeline import build_knowledge_chunks


# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# 本地 Embedding 模型目录
MODEL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"

# 实验性最低相似度阈值
DEFAULT_SCORE_THRESHOLD = 0.60


class KnowledgeRetriever:
    """ShopMate 知识库向量检索器。"""

    def __init__(self):
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Embedding 模型不存在：{MODEL_DIR}"
            )

        # 加载本地 Embedding 模型
        self.model = SentenceTransformer(
            str(MODEL_DIR),
            local_files_only=True,
        )

        # 保存知识库文本块
        self.chunks = []

        # 保存文本块对应的向量
        self.embeddings = None

    def build_index(self):
        """加载知识库并为所有文本块生成向量。"""

        self.chunks = build_knowledge_chunks()

        if not self.chunks:
            raise ValueError("知识库为空，没有可检索的文本")

        texts = [
            chunk["content"]
            for chunk in self.chunks
        ]

        # 一次性生成所有文本块的向量
        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> list[dict]:
        """检索相关文本块，并过滤低于相似度阈值的结果。"""

        if not query or not query.strip():
            raise ValueError("查询内容不能为空")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not -1 <= score_threshold <= 1:
            raise ValueError("score_threshold 必须在 -1 到 1 之间")

        if self.embeddings is None:
            raise RuntimeError("请先调用 build_index() 建立索引")

        # 将用户问题转换成向量
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        # 计算问题向量与所有文本块的余弦相似度
        scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        # 按照相似度从高到低排序
        ranked_indices = np.argsort(scores)[::-1]

        # 获取 Top-K 结果，同时过滤低于相似度阈值的文本
        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < score_threshold:
                continue

            chunk = self.chunks[int(index)]

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "content": chunk["content"],
                    "score": score,
                }
            )

            if len(results) >= top_k:
                break

        return results
