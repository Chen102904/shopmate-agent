"""BM25 关键词检索与混合检索。

为什么需要向量之外再加一路：
    向量检索依赖语义相似度，对「换了说法」比较友好，
    但对「精确词命中」反而不敏感。
    例如用户问「对处理结果不满意」，
    知识库里写的是「用户对处理结果存在异议」，
    向量能召回；但如果用户问「运费谁出」，
    向量有时会把「退款到账时间」排到前面，
    而 BM25 靠词频能准确把「运费承担」那一段顶上来。

融合方式：RRF（Reciprocal Rank Fusion，倒数排名融合）
    score = Σ 1 / (k + rank)，k 取 60。
    只看名次不看分数，因此不需要把「余弦相似度」和「BM25 分数」
    两种不同尺度的数值强行对齐——这也是 RRF 在工业界常用的原因。

关键设计：BM25 只做「重排」，不做「扩召回」
    召回集合完全由向量检索（含相似度阈值）决定，
    BM25 只调整已召回内容的先后顺序。
    这样做的好处是：
        - 不会因为关键词误命中而把无关内容塞进上下文；
        - 无关问题的拒答率不会被破坏。
"""

import re

import jieba
import numpy as np
from rank_bm25 import BM25Okapi


# RRF 的平滑常数，取自原论文推荐值
RRF_K = 60


# 中文停用词：这些词每个问题里都有，对区分内容没有帮助，
# 留在分词结果里会让 BM25 分数被稀释。
STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
    "吗", "呢", "吧", "啊", "呀", "怎么", "什么", "如何", "请问",
    "可以", "能", "能不能", "是否", "还", "又", "把", "被",
}


def tokenize(text: str) -> list[str]:
    """中文分词，并去掉停用词和空白。

    使用 jieba 的精确模式（默认），
    对售后政策这类短句足够，不需要自定义词典。
    """

    if not text:
        return []

    tokens = jieba.lcut(text)

    cleaned: list[str] = []

    for token in tokens:
        token = token.strip()

        if not token:
            continue

        # 纯标点直接丢弃
        if not re.search(r"[\w\u4e00-\u9fff]", token):
            continue

        if token in STOPWORDS:
            continue

        cleaned.append(token)

    return cleaned


class BM25Index:
    """对知识库文本块建立 BM25 索引。"""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        if not chunks:
            raise ValueError("知识库为空，无法建立 BM25 索引")

        self.tokenized_corpus = [
            tokenize(chunk["content"])
            for chunk in chunks
        ]

        # 极端情况：所有内容分词后都是空的，BM25 无法工作
        if not any(self.tokenized_corpus):
            raise ValueError("知识库分词结果为空，无法建立 BM25 索引")

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def scores(self, query: str) -> np.ndarray:
        """返回查询与每个文本块的 BM25 分数。"""

        tokens = tokenize(query)

        if not tokens:
            return np.zeros(len(self.chunks))

        return self.bm25.get_scores(tokens)


def fuse_with_rrf(
    dense_results: list[dict],
    bm25_scores: dict[str, float],
    top_k: int,
) -> list[dict]:
    """用 RRF 把向量召回结果重新排序。

    参数：
        dense_results: 向量检索返回的候选（已按相似度降序，已过阈值）
        bm25_scores: {chunk_id: bm25 分数}
        top_k: 返回条数

    返回：
        重新排序后的结果，每个元素附带
        dense_score（向量分数）和 rrf_score（融合分）。

    注意：BM25 分数不直接参与加权求和，
         而是转成「在候选集合内的名次」再融合。
         原因是 BM25 分数没有上界，和余弦相似度不可比。
    """

    if not dense_results:
        return []

    # 候选集合内的 BM25 名次
    ordered_by_bm25 = sorted(
        dense_results,
        key=lambda item: bm25_scores.get(item["chunk_id"], 0.0),
        reverse=True,
    )

    bm25_rank = {
        item["chunk_id"]: rank
        for rank, item in enumerate(ordered_by_bm25)
    }

    fused: list[dict] = []

    for dense_rank, item in enumerate(dense_results):

        chunk_id = item["chunk_id"]

        rrf_score = 1.0 / (RRF_K + dense_rank + 1) + 1.0 / (
            RRF_K + bm25_rank.get(chunk_id, len(dense_results)) + 1
        )

        fused.append(
            {
                **item,
                "dense_score": float(item["score"]),
                "bm25_score": float(bm25_scores.get(chunk_id, 0.0)),
                "rrf_score": rrf_score,
            }
        )

    fused.sort(key=lambda item: item["rrf_score"], reverse=True)

    return fused[:top_k]
