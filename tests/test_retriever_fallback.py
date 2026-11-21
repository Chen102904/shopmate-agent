"""检索层降级测试。

守的核心：

    **embedding 模型挂掉，检索不能跟着挂。**

    向量检索依赖 embedding 模型（本地 1.1 GB 或远程 API）。
    模型文件损坏、显存不足、服务超时，
    原来整条 RAG 就断了，用户收到「查询失败」。
    但 BM25 是纯词频统计，不依赖任何模型，可以顶上。

    取舍是有意识的：BM25 只认字面词，
    「退钱」命中不了「退款」，召回会变差。
    但「答得差」远好过「答不出来」。
"""

import pytest


@pytest.fixture(scope="module")
def retriever():
    """真实检索器，模块内复用。

    不用 mock 整个检索器：这个文件要验的正是
    「真实的向量通道坏掉时，真实的 BM25 能不能接住」。
    """

    from app.rag.retriever import KnowledgeRetriever

    instance = KnowledgeRetriever(use_rerank=False)
    instance.build_index()

    return instance


def _break_vector(monkeypatch, retriever):
    """让 embedding 编码抛异常，模拟模型文件损坏。"""

    def _explode(*args, **kwargs):
        raise RuntimeError("模型文件损坏")

    monkeypatch.setattr(retriever.model, "encode", _explode)


def test_dense_failure_falls_back_to_bm25(monkeypatch, retriever):
    """向量通道断了，检索仍要返回结果。"""

    _break_vector(monkeypatch, retriever)

    hits = retriever.search("退款多久到账", top_k=3)

    assert hits, "向量挂了不能直接返回空，应该由 BM25 接住"
    assert all(hit["retrieval_mode"] == "bm25_only" for hit in hits)


def test_fallback_keeps_normal_path_untouched(retriever):
    """正常时必须是 dense 结果，不能因为加了兜底就全走 BM25。"""

    hits = retriever.search("退款多久到账", top_k=3)

    assert hits
    assert all(hit.get("retrieval_mode") != "bm25_only" for hit in hits)


def test_fallback_scores_are_normalized(monkeypatch, retriever):
    """降级结果也要给 0-1 的 score。

    下游（领域门控、日志、评测）都按 0-1 使用 score。
    BM25 原始分数没有上界，直接塞进去会让阈值判断失效。
    """

    _break_vector(monkeypatch, retriever)

    hits = retriever.search("退款多久到账", top_k=3)

    assert hits

    for hit in hits:
        assert 0.0 <= hit["score"] <= 1.0
        # 原始分数要留着，否则排查「为什么排第一」时没有依据
        assert "bm25_score" in hit

    # 最高分归一化成 1.0
    assert hits[0]["score"] == pytest.approx(1.0)


def test_fallback_returns_empty_when_no_keyword_match(monkeypatch, retriever):
    """一个关键词都命中不了时，返回空而不是硬塞无关条文。"""

    _break_vector(monkeypatch, retriever)

    hits = retriever.search("zzzzqqqq", top_k=3)

    assert hits == []


def test_raises_when_both_channels_down(monkeypatch, retriever):
    """向量和 BM25 都不可用时，如实报错。

    这时候已经没有任何检索能力，
    返回空数组会让上层说「没找到相关政策」，
    掩盖了「检索服务坏了」这个事实。
    """

    _break_vector(monkeypatch, retriever)
    monkeypatch.setattr(retriever, "bm25_index", None)

    with pytest.raises(RuntimeError):
        retriever.search("退款多久到账", top_k=3)


def test_recovers_after_model_restored(retriever):
    """故障恢复后自动回到向量检索。

    这个用例在 monkeypatch 作用域外跑，
    所以模型是正常的——用来确认降级不是「一次性写死」。
    """

    hits = retriever.search("退款多久到账", top_k=3)

    assert hits
    assert all(hit.get("retrieval_mode") != "bm25_only" for hit in hits)


def test_search_multi_also_falls_back(monkeypatch, retriever):
    """多路检索（原始 + 改写）也要能降级。

    多路是主链路实际用的入口，
    单路能降级但多路不能，等于没修。
    """

    _break_vector(monkeypatch, retriever)

    hits = retriever.search_multi(
        ["退款多久到账", "退款到账时间"],
        top_k=3,
    )

    assert hits
    # 合并后可能经过 rerank，但来源必须是降级的那批
    assert all(hit.get("retrieval_mode") == "bm25_only" for hit in hits)
