"""
Cross-Encoder 重排（Rerank）。

为什么需要它：
    现在链路是「向量召回 → BM25 融合（RRF）」。
    RRF 只看名次，不看内容——它能让对的结果进候选集，
    但排不准顺序。

    向量检索把问题和文本块各自压成一个向量再算余弦相似度，
    压缩过程中丢了词序和否定关系。
    「不支持下单」和「支持下单后取消」向量上可能很近，
    但对用户来说是相反的答案。

    Cross-Encoder 把「问题 + 文本块」拼在一起过一遍模型，
    直接输出相关性分数，不吃向量压缩的损耗。
    代价是慢：不能预计算，每个候选都要跑一次前向。

    所以只对候选集重排（6–9 条），不做全库检索。

设计上的两个约束：

    1. 只重排，不扩召回。
       rerank 绝对不能把原本被阈值过滤掉的结果捞回来，
       否则拒答率会崩——用户问「今天天气」也能被塞一段政策。

    2. 不改 score 字段。
       score 是向量余弦相似度，阈值 0.60 和领域门控都按它判断。
       如果拿 cross-encoder 的原始打分覆盖它，阈值语义就全乱了
       （cross-encoder 分数没有固定上界，不是 0–1）。
       所以重排结果只改顺序，另加 rerank_score 供排查。
"""

import logging
import os

from pathlib import Path

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[2]

RERANK_MODEL_DIR = BASE_DIR / "models" / "bge-reranker-base"

# 关闭开关：设为 1 时完全跳过重排，回到纯 RRF 排序。
# 留着是为了能跑「有/无 rerank」的对照评测。
DISABLE_RERANK = os.getenv("SHOPMATE_RERANK_DISABLE", "0") == "1"

# 输入最大长度。
# 知识库的 chunk 是按 300 字切的，512 token 足够覆盖，
# 再长只会拖慢速度。
MAX_LENGTH = 512


class Reranker:
    """懒加载的 Cross-Encoder 重排器。

    加载失败或模型缺失时自动降级为「不重排」，
    不让整个 RAG 因为一个可选优化起不来。
    """

    def __init__(
        self,
        model_dir: Path = RERANK_MODEL_DIR,
        enabled: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.enabled = enabled and not DISABLE_RERANK

        self.model = None

        # 加载失败只尝试一次，避免每次请求都重试一遍
        self._load_failed = False

    def _ensure_model(self):
        """首次使用时加载模型。"""

        if self.model is not None:
            return self.model

        if not self.enabled or self._load_failed:
            return None

        if not self.model_dir.exists():
            self._load_failed = True
            return None

        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                str(self.model_dir),
                max_length=MAX_LENGTH,
            )

        except Exception:
            # 模型损坏、依赖缺失、内存不够……
            # 都不应该让检索直接失败，降级即可。
            #
            # 但这一条必须打 warning 且只打一次（_load_failed 保证）：
            # 加载失败意味着此后所有检索都不经过重排，
            # 是部署事故级别的降级，不能悄悄发生。
            logger.warning(
                "重排模型加载失败，此后检索将跳过重排 | model_dir=%s",
                self.model_dir,
                exc_info=True,
            )
            self._load_failed = True
            return None

        return self.model

    @property
    def available(self) -> bool:
        return self._ensure_model() is not None

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """按 cross-encoder 分数重排候选，返回前 top_k 条。

        候选集必须已经过阈值和领域门控，
        本方法不会引入任何新结果。
        """

        if not candidates:
            return []

        # 候选不多，排序没有收益，直接返回
        if len(candidates) <= 1:
            return candidates[:top_k]

        model = self._ensure_model()

        if model is None:
            return candidates[:top_k]

        try:
            scores = model.predict(
                [
                    (query, candidate["content"])
                    for candidate in candidates
                ]
            )

        except Exception:
            # 推理失败（显存、输入异常）时退回原顺序，
            # 不能因为重排出错就把检索结果丢掉。
            #
            # 必须留日志：降级对用户是无感的，
            # 但运维需要知道「这批回答没有经过重排」。
            # 否则线上检索质量悄悄下滑，只能靠人工发现。
            logger.warning(
                "重排推理失败，降级为原顺序 | candidates=%s",
                len(candidates),
                exc_info=True,
            )
            return candidates[:top_k]

        reranked = []

        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = round(float(score), 6)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]
