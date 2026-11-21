
import logging
import os
import threading
import time

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.bm25_retriever import BM25Index, fuse_with_rrf
from app.rag.knowledge_pipeline import build_knowledge_chunks
from app.rag.knowledge_version import (
    compute_knowledge_version,
    ensure_snapshot,
)
from app.rag.reranker import Reranker


logger = logging.getLogger(__name__)


# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# 本地 Embedding 模型目录
MODEL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"


# 知识库热重载开关。
#
# 开启后：每次检索前检查知识库指纹，变了就重建索引。
# 关闭后：只在进程启动时建一次，改文件必须重启。
#
# 为什么默认开：
#     改一条政策要重启服务，在生产上是不可接受的；
#     而这个知识库只有 15 个文本块，重建不到 0.2 秒。
#
# 什么情况下该关：
#     知识库极大（几千 chunk）时重建要几十秒，
#     应该在发布流程里重建，而不是在用户请求里触发。
HOT_RELOAD_ENABLED = os.getenv(
    "SHOPMATE_KB_HOT_RELOAD",
    "1",
).strip().lower() not in {"0", "false", "off", "no"}


# 两次指纹检查之间的最小间隔（秒）。
#
# 为什么需要冷却：
#     一次问答会走 search_multi → search 两条路，
#     如果每条都 stat 一遍文件、算一遍 sha256，
#     在知识库变大后就是纯粹的浪费。
#     2 秒足够覆盖一次用户请求，又能保证改完文件很快生效。
RELOAD_CHECK_INTERVAL = float(
    os.getenv("SHOPMATE_KB_CHECK_INTERVAL", "2.0")
)

# 最低相似度阈值。
#
# 2026-11-20 做过两轮阈值敏感性实验（30 条检索样本）：
#
# 第一轮（知识库 5 章节 / 40 行）：
#     0.45 → Hit@3 0.92，无关拒答 80%
#     0.55 → Hit@3 0.88，无关拒答 100%
#     0.60 → Hit@3 0.68，无关拒答 100%
#
# 第二轮（知识库 12 章节 / 扩写后）：
#     0.45 → Hit@3 0.760，无关拒答 60%
#     0.55 → Hit@3 0.760，无关拒答 80%
#     0.60 → Hit@3 0.760，无关拒答 100%
#     0.65 → Hit@3 0.560，无关拒答 100%
#
# 结论：知识库内容越多，无关问题越容易撞上中低相似度区间。
# 0.60 命中率最高（0.92），但会有 1 条无关问题（公司成立年份）以 0.6066 混进来。
# 单纯提高阈值到 0.62 会掉 12 个点命中率，不划算，
# 因此改用「阈值 + 领域门控」两层：
#   - 阈值 0.60 负责过滤语义不相关；
#   - 领域门控负责拦住「完全不属于售后领域」的问题。
DEFAULT_SCORE_THRESHOLD = 0.60


# 售后领域关键词。
# 用户问题里一个都不出现时，说明大概率不属于售后咨询，
# 这时要求更高的相似度才允许返回结果。
DOMAIN_KEYWORDS = {
    "订单", "退款", "退货", "换货", "物流", "快递", "签收", "售后",
    "商品", "发货", "收货", "付款", "支付", "运费", "破损", "丢件",
    "发票", "客服", "优惠", "积分", "赠品", "生鲜", "定制", "退货",
    "拒收", "补发", "保修", "质量问题", "退款率",
    # 口语表达：用户不会说「我要取消尚未付款的订单」，
    # 他说的是「刚下单就不想要了，还没给钱那种」。
    #
    # 漏掉这些词的后果不是「答得不好」，而是「完全答不出来」——
    # 问题被判定为「不含售后领域词」，阈值从 0.60 提到 0.65，
    # 正确答案的 0.56 分被直接筛掉，用户收到「没找到相关政策」。
    # 实测样本 R22 就是这样失败的。
    "下单", "买了", "购买", "不想要", "没给钱", "未给钱",
    "没付款", "未付款", "不买了", "退钱",
    # 元问题也需要放行：用户问「答不上来怎么办」「你们知识库有什么」，
    # 答案就在「人工客服政策」章节里，不能因为没出现业务词就拦掉。
    "知识库", "人工", "答不上来", "无法回答", "答不了",
}

# 无领域关键词时的更高门槛。
# 例：「ShopMate公司成立于哪一年」最高分 0.6066，
# 过了 0.60 但过不了 0.65，属于典型的「字面有点像、实则无关」。
DOMAIN_GUARD_THRESHOLD = 0.65

# 向量召回的候选倍数。
# 混合检索需要比 top_k 更多的候选来做重排，
# 取 3 倍是为了给 BM25 留出调整空间，同时不显著增加计算量。
CANDIDATE_MULTIPLIER = 3


def _has_domain_keyword(query: str) -> bool:
    """判断问题里是否出现售后领域关键词。"""

    if not query:
        return False

    return any(keyword in query for keyword in DOMAIN_KEYWORDS)


class KnowledgeRetriever:
    """ShopMate 知识库检索器。

    支持三种模式组合：
        dense  ：只用向量检索（原始行为）
        hybrid ：向量召回 + BM25 融合（默认）

    排序链路（默认全开）：
        1. 向量召回候选（过阈值）
        2. 领域门控：问句里没有售后词时要求更高相似度
        3. BM25 + RRF 融合，按名次重排
        4. Cross-Encoder 重排，按「问题 + 文本块」的实际相关性排序

    为什么默认开混合：
        知识库是政策条款，用户问法和条文措辞常常不一致。
        向量负责「听懂换了说法的意思」，
        BM25 负责「把精确命中的条款顶上来」，
        两者互补。

    为什么还要 Cross-Encoder：
        RRF 只看名次不看内容，能把对的选进候选但排不准。
        Cross-Encoder 直接把问题和文本块一起过模型，
        对「否定关系」「多条件」这类语义更敏感。
    """

    def __init__(
        self,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ):
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Embedding 模型不存在：{MODEL_DIR}"
            )

        # 加载本地 Embedding 模型
        self.model = SentenceTransformer(
            str(MODEL_DIR),
            local_files_only=True,
        )

        # Cross-Encoder 重排器（懒加载，模型缺失时自动降级）
        self.reranker = Reranker(enabled=use_rerank)

        # 保存知识库文本块
        self.chunks = []

        # 保存文本块对应的向量
        self.embeddings = None

        # BM25 索引（建立失败时为 None，自动降级为纯向量检索）
        self.bm25_index: BM25Index | None = None

        self.use_hybrid = use_hybrid

        # 当前索引对应的知识库版本（内容指纹）。
        # 空字符串表示还没建索引。
        self.version: str = ""

        # 重载锁：避免两个请求同时发现变更、同时重建索引
        self._reload_lock = threading.Lock()

        # 上次检查指纹的单调时钟时间，用于冷却
        self._last_check_time: float = 0.0

    def build_index(self):
        """加载知识库并为所有文本块生成向量。

        实现上分「先构建、后替换」两步：
            所有中间结果先放在局部变量里，
            全部成功之后才一次性赋给 self。
        这样构建中途失败时，旧索引仍然可用——
        对线上服务来说，「用略微过时的知识回答」
        远好于「检索直接不可用」。
        """

        chunks = build_knowledge_chunks()

        if not chunks:
            raise ValueError("知识库为空，没有可检索的文本")

        texts = [
            chunk["content"]
            for chunk in chunks
        ]

        # 一次性生成所有文本块的向量
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        # 建立 BM25 索引。
        # 失败不影响主流程：降级为纯向量检索，
        # 比整个服务起不来要好。
        bm25_index: BM25Index | None = None

        if self.use_hybrid:
            try:
                bm25_index = BM25Index(chunks)
            except Exception:
                logger.warning("BM25 索引构建失败，降级为纯向量检索", exc_info=True)
                bm25_index = None

        # ---- 到这里全部成功，才替换 ----
        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25_index = bm25_index

        # 记录这一份索引对应的知识库版本。
        # force=True：必须真算，不能用可能过期的缓存，
        # 否则「索引是新的、版本号是旧的」，追溯就失去意义。
        self.version = compute_knowledge_version(force=True)

        # 顺手存一份快照：
        # 下次知识库变更后，这一份就是「上一版」，
        # 没它就没法做差异对比和回滚。
        ensure_snapshot(self.version)

        return len(self.chunks)

    def maybe_reload(self) -> bool:
        """检查知识库是否变了，变了就重建索引。

        返回：
            True  → 本次真的重建了
            False → 没变化、被冷却挡住、或重载失败

        为什么失败要返回 False 而不是抛异常：
            调用方是 search()，在用户请求路径上。
            知识库被改坏是运维问题，不该让用户收到 500。
            这里记日志、继续用旧索引，等运维修文件后再自动重载。
        """

        if not HOT_RELOAD_ENABLED:
            return False

        now = time.monotonic()

        # 冷却：一次请求内会调多次 search，不必每次都验指纹
        if now - self._last_check_time < RELOAD_CHECK_INTERVAL:
            return False

        self._last_check_time = now

        current = compute_knowledge_version(force=True)

        if current == self.version:
            return False

        with self._reload_lock:

            # 双重检查：等锁期间别的线程可能已经重建完了
            current = compute_knowledge_version(force=True)

            if current == self.version:
                return False

            try:
                chunk_count = self.build_index()

                logger.info(
                    "知识库已热重载 | version=%s → %s | chunks=%s",
                    self.version,
                    compute_knowledge_version(),
                    chunk_count,
                )

                return True

            except Exception:
                # 关键：不抛出、不清空旧索引。
                # 一个格式写坏的文件不该让整个检索服务不可用。
                logger.exception(
                    "知识库热重载失败，继续使用旧索引 | old_version=%s",
                    self.version,
                )

                # 把冷却时间推后，避免坏文件导致每次请求都重试重建
                self._last_check_time = time.monotonic()

                return False

    def _dense_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> list[dict]:
        """纯向量检索，返回候选集合（已过滤低相似度）。"""

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        ranked_indices = np.argsort(scores)[::-1]

        results: list[dict] = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < score_threshold:
                break

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

    def _bm25_only_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        """向量模型不可用时的兜底检索：只用 BM25 关键词匹配。

        为什么这条通道要独立存在：
            向量检索依赖 embedding 模型（本地 1.1 GB 或远程 API）。
            模型文件损坏、显存不够、服务超时，
            整条检索就断了。而 BM25 只是词频统计，
            不依赖任何模型，完全可以顶上。

        代价要说清楚：
            BM25 只认字面词。「退款」能命中「退款」，
            但命中不了「退钱」「把钱退回来」。
            所以降级后召回会变差，尤其口语化提问。
            这是有意识的取舍：宁可答得不够全，
            也不要在这个时刻直接告诉用户「查不到」。

        阈值语义不同（容易写错的地方）：
            BM25 分数没有上界，不是 0-1 的相似度，
            不能拿 score_threshold（0.55）去比。
            这里只过滤「分数为 0」——一个关键词都没命中的块。
            为了让下游按 0-1 使用时不出错，
            score 字段填的是「相对于本批最高分」的比值，
            原始分数保留在 bm25_score 里供排查。
        """

        if self.bm25_index is None:
            return []

        scores = self.bm25_index.scores(query)

        if len(scores) == 0:
            return []

        ranked_indices = np.argsort(scores)[::-1]

        max_score = float(scores[ranked_indices[0]]) if len(ranked_indices) else 0.0

        results: list[dict] = []

        for index in ranked_indices:
            score = float(scores[index])

            # 完全不命中任何关键词，后面的更不可能命中
            if score <= 0:
                break

            chunk = self.chunks[int(index)]

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "content": chunk["content"],
                    "score": score / max_score if max_score > 0 else 0.0,
                    "bm25_score": score,
                    # 下游（RAG 回答、日志、评测）能看出这次是降级结果，
                    # 不至于把「关键词命中」当成「语义匹配」来解释。
                    "retrieval_mode": "bm25_only",
                }
            )

            if len(results) >= top_k:
                break

        return results

    def search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        _apply_rerank: bool = True,
    ) -> list[dict]:
        """检索相关文本块。

        返回的每个元素都含 score（向量相似度），
        混合模式下额外带 dense_score / bm25_score / rrf_score，
        开启重排时还有 rerank_score，方便排查「为什么这一段排前面」。

        _apply_rerank：
            内部参数。多路检索（search_multi）会关掉单路重排，
            改成合并后统一排一次，避免重复计算。
        """

        if not query or not query.strip():
            raise ValueError("查询内容不能为空")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if not -1 <= score_threshold <= 1:
            raise ValueError("score_threshold 必须在 -1 到 1 之间")

        if self.embeddings is None:
            raise RuntimeError("请先调用 build_index() 建立索引")

        # 知识库可能在运行期间被改过（改了条文、加了章节）。
        # 这里检查一次指纹，变了就重建，避免用旧条文回答新问题。
        # 内部有冷却和失败降级，不会拖慢正常请求，也不会因为
        # 文件写坏就让检索不可用。
        self.maybe_reload()

        # 1. 向量召回：多取候选，供 BM25 重排
        candidate_k = max(top_k * CANDIDATE_MULTIPLIER, top_k)

        try:
            candidates = self._dense_search(
                query,
                top_k=candidate_k,
                score_threshold=score_threshold,
            )

        except Exception:
            # 向量通道断了：embedding 模型文件损坏、显存不足、
            # 远程 embedding 服务超时，都可能走到这里。
            #
            # 这时候不能直接报错。BM25 是纯词频统计，
            # 完全不依赖模型，可以把检索接住——
            # 「答得差」远好过「答不出来」。
            logger.warning(
                "向量检索失败，降级为 BM25 关键词检索 | version=%s",
                self.version,
                exc_info=True,
            )

            if self.bm25_index is None:
                # 两条通道都不可用，这时再报错是诚实的
                raise

            return self._bm25_only_search(
                query,
                top_k=top_k,
            )

        if not candidates:
            return []

        # 领域门控：
        # 问题里完全没有售后相关词时，要求更高的相似度。
        # 这不是为了「让指标好看」，而是防止
        # 「公司成立年份」这类问题被硬塞一段政策条文回去——
        # 客服场景里，答错比答不出来更糟。
        if (
            not _has_domain_keyword(query)
            and candidates[0]["score"] < DOMAIN_GUARD_THRESHOLD
        ):
            return []

        # 2. 纯向量模式：直接交给重排（或不重排）
        if not self.use_hybrid or self.bm25_index is None:

            if not _apply_rerank:
                return candidates[:top_k]

            return self.reranker.rerank(query, candidates, top_k)

        # 3. BM25 + RRF 融合
        #    BM25 分数按知识库全体计算，这里只取候选集合内的名次。
        #    多留一倍候选给重排器，否则重排没有腾挪空间
        #    （候选只有 3 条时，怎么排都是这 3 条）。
        raw_scores = self.bm25_index.scores(query)

        bm25_scores = {
            chunk["chunk_id"]: float(raw_scores[index])
            for index, chunk in enumerate(self.chunks)
        }

        fused = fuse_with_rrf(
            candidates,
            bm25_scores,
            top_k * 2,
        )

        # 4. Cross-Encoder 重排
        #    只重排 fused（已经过阈值和门控），不会引入新结果，
        #    所以拒答行为完全不变。
        if not _apply_rerank:
            return fused[:top_k]

        return self.reranker.rerank(query, fused, top_k)

    def search_multi(
        self,
        queries: list[str],
        top_k: int = 3,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> list[dict]:
        """多路检索后合并结果。

        为什么需要多路：
            查询改写不是稳定操作。
            它能把「钱付了货还没发」变规范，提升命中；
            也可能把「退款资格有争议」改写成「应联系哪个部门」，
            反而丢掉原文里的关键词，导致检索不到。

            只信任改写结果，等于把检索质量押在一次模型输出上。
            双路检索（原始 + 改写）合并后，
            改写只能「加分」，不会「减分」。

        合并规则：
            同一 chunk 在多路结果中出现时，保留最高分。
            最后用原始问题做一次 Cross-Encoder 重排。

        为什么要「合并后再重排」：
            如果每路各排一次，一是成本翻倍（两遍 cross-encoder），
            二是合并时又按向量分排序，把两路的排序结果都打乱了。
            统一在合并后用原始问题重排一次，
            既省一半时间，也更准——重排要对着用户真正问的那句话。
        """

        if not queries:
            return []

        # 领域门控只认「用户原话」，不认改写结果。
        #
        # 原因：查询改写是为了提升检索而做的优化，
        # 它可能给「美国总统是谁」加上「政策」「售后」这类词，
        # 从而绕过门控、把无关问题强行塞进售后知识库。
        # 判断「这是不是一个售后问题」必须看用户真正问了什么。
        original_query = queries[0]

        effective_threshold = score_threshold

        if not _has_domain_keyword(original_query):
            effective_threshold = max(
                score_threshold,
                DOMAIN_GUARD_THRESHOLD,
            )

        merged: dict[str, dict] = {}

        for query in queries:

            if not query or not query.strip():
                continue

            # 每路多取一些，给合并留余量。
            # 这里关掉单路内部的重排，合并后统一做一次。
            hits = self.search(
                query,
                top_k=top_k * 2,
                score_threshold=effective_threshold,
                _apply_rerank=False,
            )

            for hit in hits:
                chunk_id = hit["chunk_id"]

                existing = merged.get(chunk_id)

                if existing is None or hit["score"] > existing["score"]:
                    merged[chunk_id] = hit

        ordered = sorted(
            merged.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        return self.reranker.rerank(
            original_query,
            ordered,
            top_k,
        )
